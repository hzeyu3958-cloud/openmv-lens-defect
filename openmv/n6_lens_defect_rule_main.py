# OpenMV N6 lens defect rule detector.
#
# This script does not need a trained model. It is useful before the camera
# dataset is ready: tune LENS_ROI and thresholds in OpenMV IDE, then use the
# same JSON format as the Windows host.

import csi
import image
import time
import ujson
import pyb

try:
    from machine import UART
except Exception:
    UART = None


# ---------------------------------------------------------------------------
# Camera and output settings
# ---------------------------------------------------------------------------

PIXFORMAT = csi.GRAYSCALE
FRAMESIZE = csi.QVGA
STARTUP_STABLE_MS = 2000

# QVGA is 320 x 240. Adjust after the fixture and lighting are fixed.
LENS_ROI = (40, 30, 240, 180)

# Real-time lens tracking. When enabled, defect detection follows the tracked
# lens rectangle instead of using the fixed LENS_ROI above.
ENABLE_LENS_TRACKING = True
TRACK_SEARCH_ROI = None          # None means full frame, or set (x, y, w, h).
TRACK_SEARCH_MARGIN = 56         # Expand the previous ROI by this many pixels.
TRACK_ROI_PADDING = 12           # Expand the detected lens blob before detect.
TRACK_SMOOTHING = 0.35           # 0..1, larger follows motion faster.
TRACK_HOLD_FRAMES = 8            # Keep last ROI for short tracking dropouts.

# Tune these after the camera arrives. Black background + side/ring light works
# best because the transparent lens edge becomes a bright/dark contour.
TRACK_STD_FACTOR = 1.15
TRACK_MIN_CONTRAST_DELTA = 12
TRACK_MIN_PIXELS = 28
TRACK_MIN_AREA = 900
TRACK_MIN_WIDTH = 32
TRACK_MIN_HEIGHT = 24
TRACK_MAX_AREA_RATIO = 0.80
TRACK_MAX_ASPECT_RATIO = 3.2
TRACK_MERGE_MARGIN = 10
TRACK_EDGE_CANNY_LOW = 45
TRACK_EDGE_CANNY_HIGH = 85
TRACK_EDGE_BINARY_THRESHOLD = 180
TRACK_MIN_CONFIDENCE = 0.16

SEND_INTERVAL_MS = 300
PRINT_JSON_TO_IDE = False

# USB 发给上位机的实时画面。上位机识别 IMG_BEGIN/IMG_END 后显示。
ENABLE_USB_IMAGE_STREAM = True
USB_IMAGE_INTERVAL_MS = 600
USB_IMAGE_JPEG_QUALITY = 70

# 预留给单片机的口：开启后会把同一条检测 JSON 从 UART 发出。
# 接线示例：OpenMV TX -> 单片机 RX，OpenMV GND -> 单片机 GND。
ENABLE_UART_OUTPUT = False
UART_ID = 3
UART_BAUDRATE = 115200

DISABLE_AUTO_GAIN_AFTER_START = True
DISABLE_AUTO_WHITEBAL_AFTER_START = True


# ---------------------------------------------------------------------------
# Image-processing settings
# ---------------------------------------------------------------------------

ENABLE_HISTEQ = True
ENABLE_DENOISE = True
ENABLE_EDGE_CANDIDATES = True

STD_FACTOR = 1.35
MIN_CONTRAST_DELTA = 18
MIN_PIXELS = 8
MIN_AREA = 8
BLOB_MERGE_MARGIN = 4
MAX_DEFECTS = 20
IOU_DUPLICATE_THRESHOLD = 0.45

CANNY_LOW = 50
CANNY_HIGH = 90
EDGE_BINARY_THRESHOLD = 180
EDGE_MIN_PIXELS = 12

EDGE_MARGIN = 10
DUST_MAX_PIXELS = 45
DUST_MAX_ASPECT_RATIO = 2.0
SCRATCH_MIN_ASPECT_RATIO = 5.0
SCRATCH_MIN_LENGTH = 24
STAIN_MIN_PIXELS = 60
STAIN_MAX_ASPECT_RATIO = 4.5
SERIOUS_PIXELS = 900
MEDIUM_PIXELS = 220

DEFECT_TYPES = (
    "scratch",
    "dust",
    "stain",
)

LEVEL_SCORE = {
    "normal": 0,
    "light": 1,
    "medium": 2,
    "serious": 3,
}

COLOR_MAP = {
    "scratch": 255,
    "dust": 210,
    "stain": 170,
}


usb = pyb.USB_VCP()
uart = None
tracked_roi = None
track_lost_frames = 0
last_track_confidence = 0.0


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def safe_round(value, digits):
    scale = 1
    for _ in range(digits):
        scale *= 10
    return int(value * scale + 0.5) / scale


def blob_pixels(blob):
    try:
        return blob.pixels()
    except Exception:
        return blob.w() * blob.h()


def blob_rect_area(blob):
    return blob.w() * blob.h()


def blob_density(blob):
    rect_area = blob_rect_area(blob)
    if rect_area <= 0:
        return 0.0
    return float(blob_pixels(blob)) / float(rect_area)


def clamp_rect(rect, frame_w, frame_h):
    x, y, w, h = rect
    x = clamp(int(x), 0, frame_w - 1)
    y = clamp(int(y), 0, frame_h - 1)
    w = clamp(int(w), 1, frame_w - x)
    h = clamp(int(h), 1, frame_h - y)
    return x, y, w, h


def expand_rect(rect, margin, frame_w, frame_h):
    x, y, w, h = rect
    return clamp_rect((x - margin, y - margin, w + margin * 2, h + margin * 2), frame_w, frame_h)


def smooth_rect(old_rect, new_rect, alpha):
    if old_rect is None:
        return new_rect
    ox, oy, ow, oh = old_rect
    nx, ny, nw, nh = new_rect
    return (
        int(ox + (nx - ox) * alpha),
        int(oy + (ny - oy) * alpha),
        int(ow + (nw - ow) * alpha),
        int(oh + (nh - oh) * alpha),
    )


def rect_center(rect):
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def tracking_search_roi(img):
    if tracked_roi is not None and track_lost_frames <= TRACK_HOLD_FRAMES:
        return expand_rect(tracked_roi, TRACK_SEARCH_MARGIN, img.width(), img.height())
    if TRACK_SEARCH_ROI is not None:
        return clamp_rect(TRACK_SEARCH_ROI, img.width(), img.height())
    return (0, 0, img.width(), img.height())


def build_tracking_thresholds(img, roi):
    stats = img.get_statistics(roi=roi)
    mean = stats.mean()
    stdev = stats.stdev()
    delta = int(max(TRACK_MIN_CONTRAST_DELTA, stdev * TRACK_STD_FACTOR))

    dark_high = clamp(int(mean - delta), 0, 255)
    bright_low = clamp(int(mean + delta), 0, 255)

    thresholds = []
    if dark_high > 3:
        thresholds.append((0, dark_high))
    if bright_low < 252:
        thresholds.append((bright_low, 255))
    return thresholds


def add_tracking_blob(blob_list, blob):
    rect = blob.rect()
    for old_blob in blob_list:
        if rect_iou(rect, old_blob.rect()) >= 0.55:
            if blob_rect_area(blob) > blob_rect_area(old_blob):
                blob_list.remove(old_blob)
                blob_list.append(blob)
            return
    blob_list.append(blob)


def find_tracking_blobs(img, roi):
    blobs = []
    thresholds = build_tracking_thresholds(img, roi)
    if thresholds:
        try:
            for blob in img.find_blobs(
                thresholds,
                roi=roi,
                pixels_threshold=TRACK_MIN_PIXELS,
                area_threshold=TRACK_MIN_AREA,
                merge=True,
                margin=TRACK_MERGE_MARGIN,
            ):
                add_tracking_blob(blobs, blob)
        except Exception:
            pass

    try:
        edge_img = img.copy()
        edge_img.find_edges(image.EDGE_CANNY, threshold=(TRACK_EDGE_CANNY_LOW, TRACK_EDGE_CANNY_HIGH))
        for blob in edge_img.find_blobs(
            [(TRACK_EDGE_BINARY_THRESHOLD, 255)],
            roi=roi,
            pixels_threshold=TRACK_MIN_PIXELS,
            area_threshold=TRACK_MIN_AREA,
            merge=True,
            margin=TRACK_MERGE_MARGIN,
        ):
            add_tracking_blob(blobs, blob)
    except Exception:
        pass

    return blobs


def tracking_blob_score(blob, frame_w, frame_h):
    x, y, w, h = blob.rect()
    if w < TRACK_MIN_WIDTH or h < TRACK_MIN_HEIGHT:
        return 0.0

    rect_area = w * h
    frame_area = frame_w * frame_h
    if rect_area < TRACK_MIN_AREA or rect_area > frame_area * TRACK_MAX_AREA_RATIO:
        return 0.0

    aspect = float(max(w, h)) / float(max(1, min(w, h)))
    if aspect > TRACK_MAX_ASPECT_RATIO:
        return 0.0

    pixels = blob_pixels(blob)
    area_score = min(1.0, rect_area / float(frame_area * 0.28))
    pixel_score = min(1.0, pixels / float(frame_area * 0.018))
    aspect_score = 1.0 - min(1.0, abs(aspect - 1.55) / 2.2)
    density_score = 1.0 - min(1.0, abs(blob_density(blob) - 0.22) / 0.55)

    distance_score = 0.65
    if tracked_roi is not None:
        cx, cy = rect_center((x, y, w, h))
        pcx, pcy = rect_center(tracked_roi)
        dx = cx - pcx
        dy = cy - pcy
        distance = (dx * dx + dy * dy) ** 0.5
        diagonal = (frame_w * frame_w + frame_h * frame_h) ** 0.5
        distance_score = 1.0 - min(1.0, distance / max(1.0, diagonal))

    return (
        area_score * 0.30 +
        pixel_score * 0.24 +
        aspect_score * 0.18 +
        density_score * 0.10 +
        distance_score * 0.18
    )


def choose_lens_blob(blobs, frame_w, frame_h):
    best_blob = None
    best_score = 0.0
    for blob in blobs:
        score = tracking_blob_score(blob, frame_w, frame_h)
        if score > best_score:
            best_score = score
            best_blob = blob
    return best_blob, best_score


def lens_info(found, roi, confidence, source):
    x, y, w, h = roi
    cx, cy = rect_center(roi)
    return {
        "found": found,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "confidence": safe_round(confidence, 2),
        "lost_frames": track_lost_frames,
        "source": source,
    }


def track_lens(img):
    global tracked_roi, track_lost_frames, last_track_confidence

    if not ENABLE_LENS_TRACKING:
        tracked_roi = clamp_rect(LENS_ROI, img.width(), img.height())
        track_lost_frames = 0
        last_track_confidence = 1.0
        return lens_info(True, tracked_roi, 1.0, "fixed")

    search_roi = tracking_search_roi(img)
    blobs = find_tracking_blobs(img, search_roi)
    best_blob, confidence = choose_lens_blob(blobs, img.width(), img.height())

    if best_blob is not None and confidence >= TRACK_MIN_CONFIDENCE:
        raw_roi = expand_rect(best_blob.rect(), TRACK_ROI_PADDING, img.width(), img.height())
        tracked_roi = clamp_rect(smooth_rect(tracked_roi, raw_roi, TRACK_SMOOTHING), img.width(), img.height())
        track_lost_frames = 0
        last_track_confidence = confidence
        return lens_info(True, tracked_roi, confidence, "tracked")

    if tracked_roi is not None and track_lost_frames < TRACK_HOLD_FRAMES:
        track_lost_frames += 1
        return lens_info(True, tracked_roi, last_track_confidence * 0.7, "hold")

    track_lost_frames += 1
    fallback_roi = clamp_rect(LENS_ROI, img.width(), img.height())
    return lens_info(False, fallback_roi, 0.0, "fallback")


def is_near_roi_edge(x, y, w, h, roi):
    rx, ry, rw, rh = roi
    return (
        x <= rx + EDGE_MARGIN or
        y <= ry + EDGE_MARGIN or
        x + w >= rx + rw - EDGE_MARGIN or
        y + h >= ry + rh - EDGE_MARGIN
    )


def get_roi_brightness_delta(img, defect_rect, lens_roi):
    try:
        defect_stats = img.get_statistics(roi=defect_rect)
        lens_stats = img.get_statistics(roi=lens_roi)
        return abs(defect_stats.mean() - lens_stats.mean())
    except Exception:
        return 0


def estimate_level(defect_type, area, length, brightness_delta):
    if defect_type == "scratch":
        if length >= 70 or area >= MEDIUM_PIXELS:
            return "medium"
        return "light"

    if defect_type == "dust":
        return "light"

    if defect_type == "stain":
        if area >= MEDIUM_PIXELS:
            return "medium"
        return "light"

    if area >= SERIOUS_PIXELS:
        return "serious"
    if area >= MEDIUM_PIXELS:
        return "medium"
    if area > 0:
        return "light"
    return "normal"


def rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter_w = x2 - x1
    inter_h = y2 - y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def add_blob_if_unique(blob_list, blob):
    rect = blob.rect()
    for old_blob in blob_list:
        if rect_iou(rect, old_blob.rect()) >= IOU_DUPLICATE_THRESHOLD:
            if blob_pixels(blob) > blob_pixels(old_blob):
                blob_list.remove(old_blob)
                blob_list.append(blob)
            return
    blob_list.append(blob)


def sort_blobs_by_size(blobs):
    result = []
    temp = list(blobs)
    while temp:
        max_index = 0
        max_pixels = blob_pixels(temp[0])
        for i in range(1, len(temp)):
            pixels = blob_pixels(temp[i])
            if pixels > max_pixels:
                max_pixels = pixels
                max_index = i
        result.append(temp.pop(max_index))
    return result


def preprocess_image(img):
    if ENABLE_HISTEQ:
        try:
            img.histeq(adaptive=True, clip_limit=3)
        except Exception:
            try:
                img.histeq()
            except Exception:
                pass

    if ENABLE_DENOISE:
        try:
            img.median(1)
        except Exception:
            pass

    return img


def build_anomaly_thresholds(img, roi):
    stats = img.get_statistics(roi=roi)
    mean = stats.mean()
    stdev = stats.stdev()
    delta = int(max(MIN_CONTRAST_DELTA, stdev * STD_FACTOR))

    dark_high = clamp(int(mean - delta), 0, 255)
    bright_low = clamp(int(mean + delta), 0, 255)

    thresholds = []
    if dark_high > 3:
        thresholds.append((0, dark_high))
    if bright_low < 252:
        thresholds.append((bright_low, 255))
    return thresholds


def find_intensity_blobs(img, roi):
    thresholds = build_anomaly_thresholds(img, roi)
    if not thresholds:
        return []
    return img.find_blobs(
        thresholds,
        roi=roi,
        pixels_threshold=MIN_PIXELS,
        area_threshold=MIN_AREA,
        merge=True,
        margin=BLOB_MERGE_MARGIN,
    )


def find_edge_blobs(img, roi):
    if not ENABLE_EDGE_CANDIDATES:
        return []

    try:
        edge_img = img.copy()
        edge_img.find_edges(image.EDGE_CANNY, threshold=(CANNY_LOW, CANNY_HIGH))
        return edge_img.find_blobs(
            [(EDGE_BINARY_THRESHOLD, 255)],
            roi=roi,
            pixels_threshold=EDGE_MIN_PIXELS,
            area_threshold=MIN_AREA,
            merge=True,
            margin=BLOB_MERGE_MARGIN,
        )
    except Exception:
        return []


def find_defect_candidates(img, roi):
    candidates = []

    for blob in find_intensity_blobs(img, roi):
        add_blob_if_unique(candidates, blob)

    for blob in find_edge_blobs(img, roi):
        add_blob_if_unique(candidates, blob)

    candidates = sort_blobs_by_size(candidates)
    if len(candidates) > MAX_DEFECTS:
        candidates = candidates[:MAX_DEFECTS]
    return candidates


def classify_defect(blob, roi, img):
    x = blob.x()
    y = blob.y()
    w = blob.w()
    h = blob.h()
    area = blob_pixels(blob)
    length = max(w, h)
    short_side = max(1, min(w, h))
    aspect_ratio = float(length) / float(short_side)
    density = blob_density(blob)
    brightness_delta = get_roi_brightness_delta(img, (x, y, w, h), roi)

    defect_type = None
    confidence = 0.0

    if aspect_ratio >= SCRATCH_MIN_ASPECT_RATIO and length >= SCRATCH_MIN_LENGTH:
        defect_type = "scratch"
        confidence = 0.76 + min(0.16, (aspect_ratio - SCRATCH_MIN_ASPECT_RATIO) / 25.0)
    elif area <= DUST_MAX_PIXELS and aspect_ratio <= DUST_MAX_ASPECT_RATIO:
        defect_type = "dust"
        confidence = 0.70 + min(0.18, (DUST_MAX_PIXELS - area) / float(DUST_MAX_PIXELS))
    elif area >= STAIN_MIN_PIXELS and aspect_ratio <= STAIN_MAX_ASPECT_RATIO:
        defect_type = "stain"
        confidence = 0.66 + min(0.18, abs(0.55 - density) + brightness_delta / 255.0)

    if defect_type is None:
        return None

    confidence = clamp(confidence, 0.50, 0.98)
    level = estimate_level(defect_type, area, length, brightness_delta)

    return {
        "type": defect_type,
        "confidence": safe_round(confidence, 2),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "length": length,
        "aspect_ratio": safe_round(aspect_ratio, 2),
        "level": level,
    }


def build_result(defects, frame_w, frame_h, active_roi, lens):
    summary = {}
    for defect_type in DEFECT_TYPES:
        summary[defect_type] = 0

    overall_level = "normal"
    for defect in defects:
        defect_type = defect["type"]
        if defect_type in summary:
            summary[defect_type] += 1

        if LEVEL_SCORE[defect["level"]] > LEVEL_SCORE[overall_level]:
            overall_level = defect["level"]

    return {
        "has_defect": len(defects) > 0,
        "defect_count": len(defects),
        "summary": summary,
        "overall_level": overall_level,
        "defects": defects,
        "timestamp": time.ticks_ms(),
        "model": "rule_detector",
        "frame": {"w": frame_w, "h": frame_h},
        "roi": {
            "x": active_roi[0],
            "y": active_roi[1],
            "w": active_roi[2],
            "h": active_roi[3],
        },
        "lens": lens,
    }


def draw_debug(img, roi, defects, lens):
    lens_color = 180 if lens.get("found") else 70
    img.draw_rectangle(roi, color=lens_color)
    img.draw_cross(lens.get("cx", roi[0] + roi[2] // 2), lens.get("cy", roi[1] + roi[3] // 2), color=lens_color)
    img.draw_string(4, 4, "lens:%s %s" % (lens.get("source", ""), lens.get("confidence", 0)), color=lens_color)
    for defect in defects:
        defect_type = defect["type"]
        color = COLOR_MAP.get(defect_type, 255)
        rect = (defect["x"], defect["y"], defect["w"], defect["h"])
        img.draw_rectangle(rect, color=color)
        img.draw_string(defect["x"], max(0, defect["y"] - 10), defect_type, color=color)


def send_line(text):
    try:
        usb.write((text + "\n").encode("utf-8"))
    except Exception:
        pass

    if uart is not None:
        try:
            uart.write(text + "\n")
        except Exception:
            pass

    if PRINT_JSON_TO_IDE:
        print(text)


def send_usb_image(img):
    if not ENABLE_USB_IMAGE_STREAM:
        return

    try:
        width = img.width()
        height = img.height()
        jpg = img.compress(quality=USB_IMAGE_JPEG_QUALITY)
        usb.write(("IMG_BEGIN %d %d %d\n" % (jpg.size(), width, height)).encode("utf-8"))
        usb.write(jpg)
        usb.write(b"IMG_END\n")
    except Exception as exc:
        if PRINT_JSON_TO_IDE:
            print("USB image stream failed:", exc)


if ENABLE_UART_OUTPUT and UART is not None:
    try:
        uart = UART(UART_ID, baudrate=UART_BAUDRATE)
    except Exception as exc:
        print("UART init failed:", exc)
        uart = None

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(PIXFORMAT)
csi0.framesize(FRAMESIZE)
csi0.snapshot(time=STARTUP_STABLE_MS)

if DISABLE_AUTO_GAIN_AFTER_START:
    csi0.auto_gain(False)
if DISABLE_AUTO_WHITEBAL_AFTER_START:
    csi0.auto_whitebal(False)

clock = time.clock()
last_send_ms = time.ticks_ms()
last_image_ms = time.ticks_ms()

while True:
    clock.tick()
    img = csi0.snapshot()
    lens = track_lens(img)
    active_roi = clamp_rect((lens["x"], lens["y"], lens["w"], lens["h"]), img.width(), img.height())
    preprocess_image(img)

    blobs = find_defect_candidates(img, active_roi) if lens["found"] else []
    defects = []
    for blob in blobs:
        defect = classify_defect(blob, active_roi, img)
        if defect is not None:
            defects.append(defect)

    result = build_result(defects, img.width(), img.height(), active_roi, lens)
    draw_debug(img, active_roi, defects, lens)

    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
        send_line(ujson.dumps(result))
        last_send_ms = now_ms
    if time.ticks_diff(now_ms, last_image_ms) >= USB_IMAGE_INTERVAL_MS:
        send_usb_image(img)
        last_image_ms = now_ms
