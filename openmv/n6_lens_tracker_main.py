# OpenMV N6 real-time lens tracker.
#
# Run this first after the camera arrives. It only tracks the lens rectangle and
# emits JSON coordinates; no trained model is needed.

import csi
import image
import time
import ujson
import pyb

try:
    from machine import UART
except Exception:
    UART = None


PIXFORMAT = csi.GRAYSCALE
FRAMESIZE = csi.QVGA
STARTUP_STABLE_MS = 2000

# Fallback ROI when tracking is lost. Adjust this after you see the IDE image.
LENS_ROI = (40, 30, 240, 180)

TRACK_SEARCH_ROI = None
TRACK_SEARCH_MARGIN = 56
TRACK_ROI_PADDING = 12
TRACK_SMOOTHING = 0.35
TRACK_HOLD_FRAMES = 8

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

SEND_INTERVAL_MS = 100
PRINT_JSON_TO_IDE = False

ENABLE_UART_OUTPUT = False
UART_ID = 3
UART_BAUDRATE = 115200

DISABLE_AUTO_GAIN_AFTER_START = True
DISABLE_AUTO_WHITEBAL_AFTER_START = True


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


def build_result(lens, fps, frame_w, frame_h):
    return {
        "has_defect": False,
        "defect_count": 0,
        "summary": {
            "scratch": 0,
            "stain": 0,
        },
        "overall_level": "normal",
        "defects": [],
        "timestamp": time.ticks_ms(),
        "model": "lens_tracker",
        "frame": {"w": frame_w, "h": frame_h},
        "fps": safe_round(fps, 1),
        "lens": lens,
    }


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


def draw_tracking(img, lens):
    color = 220 if lens["found"] else 80
    roi = (lens["x"], lens["y"], lens["w"], lens["h"])
    img.draw_rectangle(roi, color=color)
    img.draw_cross(lens["cx"], lens["cy"], color=color)
    img.draw_string(4, 4, "lens:%s %.2f" % (lens["source"], lens["confidence"]), color=color)


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

while True:
    clock.tick()
    img = csi0.snapshot()
    lens = track_lens(img)
    draw_tracking(img, lens)

    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
        send_line(ujson.dumps(build_result(lens, clock.fps(), img.width(), img.height())))
        last_send_ms = now_ms
