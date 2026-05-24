# OpenMV N6 lens defect rule detector.
#
# This script does not need a trained model. It is useful before the camera
# dataset is ready: tune LENS_ROI and thresholds in OpenMV IDE, then use the
# same JSON format as the Windows host.

import csi
import image
import sys
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

PIXFORMAT = csi.RGB565
PREFERRED_FRAMESIZES = (csi.XGA, csi.VGA, csi.QVGA)
STARTUP_STABLE_MS = 2000

# Current N6 preview falls back to about 640 x 400. Keep this ROI tight around
# the real lens so frame/background edges do not become false defects.
LENS_ROI = (150, 90, 380, 260)

# Real-time lens tracking. When enabled, defect detection follows the tracked
# lens rectangle instead of using the fixed LENS_ROI above.
ENABLE_LENS_TRACKING = False
TRACK_SEARCH_ROI = None          # None means full frame, or set (x, y, w, h).
TRACK_SEARCH_MARGIN = 180        # Expand the previous ROI by this many pixels.
TRACK_ROI_PADDING = 38           # Expand the detected lens blob before detect.
TRACK_SMOOTHING = 0.35           # 0..1, larger follows motion faster.
TRACK_HOLD_FRAMES = 8            # Keep last ROI for short tracking dropouts.

# Tune these after the camera arrives. Black background + side/ring light works
# best because the transparent lens edge becomes a bright/dark contour.
TRACK_STD_FACTOR = 1.15
TRACK_MIN_CONTRAST_DELTA = 12
TRACK_MIN_PIXELS = 287
TRACK_MIN_AREA = 9216
TRACK_MIN_WIDTH = 102
TRACK_MIN_HEIGHT = 77
TRACK_MAX_AREA_RATIO = 0.80
TRACK_MAX_ASPECT_RATIO = 3.2
TRACK_MERGE_MARGIN = 32
TRACK_EDGE_CANNY_LOW = 45
TRACK_EDGE_CANNY_HIGH = 85
TRACK_EDGE_BINARY_THRESHOLD = 180
TRACK_MIN_CONFIDENCE = 0.16

SEND_INTERVAL_MS = 120
PRINT_JSON_TO_IDE = False
DRAW_DEBUG = False

# Temporal stability: one-frame glare/noise must not flip the final result.
STABLE_DEFECT_CONFIRM_FRAMES = 1
STABLE_DEFECT_SWITCH_FRAMES = 2
STABLE_NORMAL_CONFIRM_FRAMES = 3
STABLE_DEFECT_HOLD_FRAMES = 4

# USB 发给上位机的实时画面。上位机识别 IMG_BEGIN/IMG_END 后显示。
ENABLE_USB_IMAGE_STREAM = True
USB_IMAGE_INTERVAL_MS = 300
USB_IMAGE_JPEG_QUALITY = 38
USB_IMAGE_JPEG_SUBSAMPLING = None
SEND_PREVIEW_ROI_ONLY = True

# 预留给单片机的口：开启后会把同一条检测 JSON 从 UART 发出。
# 接线示例：OpenMV TX -> 单片机 RX，OpenMV GND -> 单片机 GND。
ENABLE_UART_OUTPUT = False
UART_ID = 3
UART_BAUDRATE = 115200

DISABLE_AUTO_GAIN_AFTER_START = False
DISABLE_AUTO_WHITEBAL_AFTER_START = False


# ---------------------------------------------------------------------------
# Image-processing settings
# ---------------------------------------------------------------------------

ENABLE_HISTEQ = False
ENABLE_DENOISE = False
ENABLE_EDGE_CANDIDATES = True

STD_FACTOR = 1.20
MIN_CONTRAST_DELTA = 18
MIN_PIXELS = 45
MIN_AREA = 45
BLOB_MERGE_MARGIN = 3
MAX_DEFECTS = 20
IOU_DUPLICATE_THRESHOLD = 0.45

CANNY_LOW = 50
CANNY_HIGH = 90
EDGE_BINARY_THRESHOLD = 180
EDGE_MIN_PIXELS = 35

EDGE_MARGIN = 32
SCRATCH_MIN_ASPECT_RATIO = 2.8
SCRATCH_MIN_LENGTH = 45
SCRATCH_MAX_SHORT_SIDE = 20
SCRATCH_MAX_PIXELS = 2200
SCRATCH_MAX_AREA_RATIO = 0.05
SCRATCH_EDGE_MAX_AREA_RATIO = 0.0008
STAIN_MIN_PIXELS = 2800
STAIN_MIN_DENSITY = 0.24
STAIN_MIN_BRIGHTNESS_DELTA = 14
DARK_DEFECT_MIN_RECT_DELTA = 12
STAIN_MAX_AREA_RATIO = 0.14
STAIN_MAX_ASPECT_RATIO = 2.8
STAIN_CLUSTER_MIN_SCRATCHES = 2
STAIN_CLUSTER_MAX_GAP = 42
STAIN_CLUSTER_MIN_TOTAL_AREA = 900
STAIN_CLUSTER_MIN_TOTAL_LENGTH = 150
STAIN_CLUSTER_MIN_BOX_FILL = 0.055
STAIN_CLUSTER_MIN_BOX_SHORT_SIDE = 26
STAIN_CLUSTER_MIN_BRIGHTNESS_DELTA = 34
SERIOUS_PIXELS = 9216
MEDIUM_PIXELS = 2253

DEFECT_TYPES = (
    "scratch",
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
    "stain": 170,
}


class StdOutPort:
    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif hasattr(data, "bytearray"):
            data = data.bytearray()
        sys.stdout.buffer.write(data)


def open_usb_output_port():
    try:
        return pyb.USB_VCP()
    except Exception:
        return StdOutPort()


usb = open_usb_output_port()
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


def stats_value(stats, name):
    value = getattr(stats, name)
    try:
        if callable(value):
            return value()
    except Exception:
        pass
    return value


def object_value(obj, name):
    value = getattr(obj, name)
    try:
        if callable(value):
            return value()
    except Exception:
        pass
    return value


def blob_rect(blob):
    return object_value(blob, "rect")


def blob_x(blob):
    return object_value(blob, "x")


def blob_y(blob):
    return object_value(blob, "y")


def blob_w(blob):
    return object_value(blob, "w")


def blob_h(blob):
    return object_value(blob, "h")


def blob_pixels(blob):
    try:
        return object_value(blob, "pixels")
    except Exception:
        return blob_w(blob) * blob_h(blob)


def blob_rect_area(blob):
    return blob_w(blob) * blob_h(blob)


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
    mean = stats_value(stats, "mean")
    stdev = stats_value(stats, "stdev")
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
    rect = blob_rect(blob)
    for old_blob in blob_list:
        if rect_iou(rect, blob_rect(old_blob)) >= 0.55:
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
    x, y, w, h = blob_rect(blob)
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
        raw_roi = expand_rect(blob_rect(best_blob), TRACK_ROI_PADDING, img.width(), img.height())
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
    return abs(get_roi_brightness_signed_delta(img, defect_rect, lens_roi))


def get_roi_brightness_signed_delta(img, defect_rect, lens_roi):
    try:
        defect_stats = img.get_statistics(roi=defect_rect)
        lens_stats = img.get_statistics(roi=lens_roi)
        return stats_value(defect_stats, "mean") - stats_value(lens_stats, "mean")
    except Exception:
        return 0


def estimate_level(defect_type, area, length, brightness_delta):
    if defect_type == "scratch":
        if length >= 70 or area >= MEDIUM_PIXELS:
            return "medium"
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
    rect = blob_rect(blob)
    for old_blob in blob_list:
        if rect_iou(rect, blob_rect(old_blob)) >= IOU_DUPLICATE_THRESHOLD:
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


def make_detection_image(img):
    try:
        return img.to_grayscale(copy=True)
    except Exception:
        pass

    try:
        gray = img.copy()
        gray.to_grayscale()
        return gray
    except Exception:
        return img


def build_anomaly_thresholds(img, roi):
    stats = img.get_statistics(roi=roi)
    mean = stats_value(stats, "mean")
    stdev = stats_value(stats, "stdev")
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
    x = blob_x(blob)
    y = blob_y(blob)
    w = blob_w(blob)
    h = blob_h(blob)
    area = blob_pixels(blob)
    length = max(w, h)
    short_side = max(1, min(w, h))
    aspect_ratio = float(length) / float(short_side)
    density = blob_density(blob)
    signed_brightness_delta = get_roi_brightness_signed_delta(img, (x, y, w, h), roi)
    brightness_delta = abs(signed_brightness_delta)
    is_dark_defect = signed_brightness_delta <= -DARK_DEFECT_MIN_RECT_DELTA
    roi_area = max(1, roi[2] * roi[3])
    area_ratio = float(area) / float(roi_area)

    near_edge = is_near_roi_edge(x, y, w, h, roi)
    if area_ratio >= 0.30:
        return None
    if near_edge and area_ratio >= 0.03 and not is_dark_defect:
        return None

    defect_type = None
    confidence = 0.0
    edge_scratch_noise = near_edge and not is_dark_defect and area_ratio >= SCRATCH_EDGE_MAX_AREA_RATIO
    stain_shape = (
        aspect_ratio <= STAIN_MAX_ASPECT_RATIO
        or short_side >= STAIN_CLUSTER_MIN_BOX_SHORT_SIDE
        or area > SCRATCH_MAX_PIXELS
    )

    if (
        not edge_scratch_noise
        and aspect_ratio >= SCRATCH_MIN_ASPECT_RATIO
        and length >= SCRATCH_MIN_LENGTH
        and short_side <= SCRATCH_MAX_SHORT_SIDE
        and area <= SCRATCH_MAX_PIXELS
        and area_ratio <= SCRATCH_MAX_AREA_RATIO
    ):
        defect_type = "scratch"
        confidence = 0.76 + min(0.16, (aspect_ratio - SCRATCH_MIN_ASPECT_RATIO) / 25.0)
    elif (
        area >= STAIN_MIN_PIXELS
        and area_ratio <= STAIN_MAX_AREA_RATIO
        and (density >= STAIN_MIN_DENSITY or is_dark_defect)
        and brightness_delta >= STAIN_MIN_BRIGHTNESS_DELTA
        and stain_shape
    ):
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
        "density": safe_round(density, 2),
        "area_ratio": safe_round(area_ratio, 3),
        "brightness_delta": safe_round(brightness_delta, 1),
        "brightness_signed_delta": safe_round(signed_brightness_delta, 1),
        "level": level,
    }


def defect_rect(defect):
    return defect["x"], defect["y"], defect["w"], defect["h"]


def defect_rect_gap(a, b):
    ax, ay, aw, ah = defect_rect(a)
    bx, by, bw, bh = defect_rect(b)
    dx = 0
    if ax + aw < bx:
        dx = bx - (ax + aw)
    elif bx + bw < ax:
        dx = ax - (bx + bw)

    dy = 0
    if ay + ah < by:
        dy = by - (ay + ah)
    elif by + bh < ay:
        dy = ay - (by + bh)

    if dx > dy:
        return dx
    return dy


def cluster_scratch_defects(scratches):
    clusters = []
    used = []
    for _ in scratches:
        used.append(False)

    for i in range(len(scratches)):
        if used[i]:
            continue
        used[i] = True
        cluster_indexes = [i]
        changed = True
        while changed:
            changed = False
            for j in range(len(scratches)):
                if used[j]:
                    continue
                for old_index in cluster_indexes:
                    if defect_rect_gap(scratches[j], scratches[old_index]) <= STAIN_CLUSTER_MAX_GAP:
                        used[j] = True
                        cluster_indexes.append(j)
                        changed = True
                        break

        cluster = []
        for index in cluster_indexes:
            cluster.append(scratches[index])
        clusters.append(cluster)

    return clusters


def scratch_cluster_to_stain(cluster, roi):
    if len(cluster) < STAIN_CLUSTER_MIN_SCRATCHES:
        return None

    left = cluster[0]["x"]
    top = cluster[0]["y"]
    right = cluster[0]["x"] + cluster[0]["w"]
    bottom = cluster[0]["y"] + cluster[0]["h"]
    total_area = 0
    total_length = 0
    max_confidence = 0.0
    max_delta = 0.0
    min_signed_delta = 255.0

    for defect in cluster:
        x, y, w, h = defect_rect(defect)
        if x < left:
            left = x
        if y < top:
            top = y
        if x + w > right:
            right = x + w
        if y + h > bottom:
            bottom = y + h
        total_area += int(defect.get("area", 0))
        total_length += int(defect.get("length", 0))
        confidence = float(defect.get("confidence", 0.0))
        if confidence > max_confidence:
            max_confidence = confidence
        delta = float(defect.get("brightness_delta", 0.0))
        if delta > max_delta:
            max_delta = delta
        signed_delta = float(defect.get("brightness_signed_delta", delta))
        if signed_delta < min_signed_delta:
            min_signed_delta = signed_delta

    width = max(1, right - left)
    height = max(1, bottom - top)
    length = max(width, height)
    short_side = max(1, min(width, height))
    box_area = max(1, width * height)
    density = float(total_area) / float(box_area)
    roi_area = max(1, roi[2] * roi[3])
    area_ratio = float(total_area) / float(roi_area)

    if area_ratio > STAIN_MAX_AREA_RATIO:
        return None
    if min_signed_delta > -DARK_DEFECT_MIN_RECT_DELTA:
        return None
    if max_delta < STAIN_CLUSTER_MIN_BRIGHTNESS_DELTA:
        return None
    if total_area < STAIN_CLUSTER_MIN_TOTAL_AREA and total_length < STAIN_CLUSTER_MIN_TOTAL_LENGTH:
        return None
    if density < STAIN_CLUSTER_MIN_BOX_FILL and short_side < STAIN_CLUSTER_MIN_BOX_SHORT_SIDE:
        return None

    confidence = clamp(max_confidence + 0.04 + min(0.10, density), 0.58, 0.96)
    return {
        "type": "stain",
        "confidence": safe_round(confidence, 2),
        "x": left,
        "y": top,
        "w": width,
        "h": height,
        "area": total_area,
        "length": length,
        "aspect_ratio": safe_round(float(length) / float(short_side), 2),
        "density": safe_round(density, 2),
        "area_ratio": safe_round(area_ratio, 3),
        "brightness_delta": safe_round(max_delta, 1),
        "brightness_signed_delta": safe_round(min_signed_delta, 1),
        "level": estimate_level("stain", total_area, length, max_delta),
        "clustered_from": "scratch",
        "cluster_size": len(cluster),
    }


def refine_defect_types(defects, roi):
    scratches = []
    refined = []
    for defect in defects:
        if defect.get("type") == "scratch":
            scratches.append(defect)
        else:
            refined.append(defect)

    for cluster in cluster_scratch_defects(scratches):
        stain = scratch_cluster_to_stain(cluster, roi)
        if stain is not None:
            refined.append(stain)
        else:
            for defect in cluster:
                refined.append(defect)

    refined.sort(key=lambda item: (LEVEL_SCORE.get(item.get("level", "normal"), 0), item.get("area", 0)), reverse=True)
    return refined


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


def primary_defect_type(defects):
    if not defects:
        return "normal"

    best = defects[0]
    best_rank = LEVEL_SCORE.get(best.get("level", "normal"), 0) * 100 + int(float(best.get("confidence", 0)) * 100)
    for defect in defects[1:]:
        rank = LEVEL_SCORE.get(defect.get("level", "normal"), 0) * 100 + int(float(defect.get("confidence", 0)) * 100)
        if rank > best_rank:
            best = defect
            best_rank = rank
    return best.get("type", "normal")


stable_defects = []
stable_class = "normal"
candidate_class = None
candidate_count = 0
defect_hold_frames = 0


def stabilize_defects(defects):
    global stable_defects, stable_class, candidate_class, candidate_count, defect_hold_frames

    raw_class = primary_defect_type(defects)
    if raw_class == stable_class:
        stable_defects = defects
        candidate_class = None
        candidate_count = 0
        if raw_class != "normal":
            defect_hold_frames = STABLE_DEFECT_HOLD_FRAMES
        return stable_defects

    if raw_class != candidate_class:
        candidate_class = raw_class
        candidate_count = 1
    else:
        candidate_count += 1

    if raw_class == "normal":
        if stable_class != "normal" and defect_hold_frames > 0:
            defect_hold_frames -= 1
        if candidate_count >= STABLE_NORMAL_CONFIRM_FRAMES and defect_hold_frames <= 0:
            stable_class = "normal"
            stable_defects = []
            candidate_class = None
            candidate_count = 0
        return stable_defects

    required = STABLE_DEFECT_CONFIRM_FRAMES if stable_class == "normal" else STABLE_DEFECT_SWITCH_FRAMES
    if candidate_count >= required:
        stable_class = raw_class
        stable_defects = defects
        defect_hold_frames = STABLE_DEFECT_HOLD_FRAMES
        candidate_class = None
        candidate_count = 0
    return stable_defects


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


def send_usb_image(img, roi=None):
    if not ENABLE_USB_IMAGE_STREAM:
        return

    try:
        preview = img
        if SEND_PREVIEW_ROI_ONLY and roi is not None:
            try:
                preview = img.copy(roi=roi, copy_to_fb=False)
            except Exception:
                preview = img.copy(roi=roi)
        width = preview.width()
        height = preview.height()
        try:
            if USB_IMAGE_JPEG_SUBSAMPLING is None:
                raise TypeError("JPEG subsampling is not available")
            jpg = preview.compress(quality=USB_IMAGE_JPEG_QUALITY, subsampling=USB_IMAGE_JPEG_SUBSAMPLING)
        except TypeError:
            jpg = preview.compress(quality=USB_IMAGE_JPEG_QUALITY)
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
framesize_ready = False
for framesize in PREFERRED_FRAMESIZES:
    try:
        csi0.framesize(framesize)
        framesize_ready = True
        break
    except Exception as exc:
        print("framesize failed:", framesize, exc)
if not framesize_ready:
    raise RuntimeError("No supported framesize")
csi0.snapshot(time=STARTUP_STABLE_MS)

if DISABLE_AUTO_GAIN_AFTER_START:
    try:
        csi0.auto_gain(False)
    except Exception as exc:
        print("auto_gain failed:", exc)
if DISABLE_AUTO_WHITEBAL_AFTER_START:
    try:
        csi0.auto_whitebal(False)
    except Exception as exc:
        print("auto_whitebal failed:", exc)

clock = time.clock()
last_send_ms = time.ticks_ms()
last_image_ms = time.ticks_ms()

while True:
    clock.tick()
    img = csi0.snapshot()
    detect_img = make_detection_image(img)
    lens = track_lens(detect_img)
    active_roi = clamp_rect((lens["x"], lens["y"], lens["w"], lens["h"]), img.width(), img.height())
    preprocess_image(detect_img)

    blobs = find_defect_candidates(detect_img, active_roi)
    defects = []
    for blob in blobs:
        defect = classify_defect(blob, active_roi, detect_img)
        if defect is not None:
            defects.append(defect)

    defects = refine_defect_types(defects, active_roi)
    stable_frame_defects = stabilize_defects(defects)
    result = build_result(stable_frame_defects, img.width(), img.height(), active_roi, lens)
    result["raw_defect_count"] = len(defects)
    result["stability"] = {
        "stable_class": stable_class,
        "candidate_class": candidate_class or "",
        "candidate_count": candidate_count,
    }
    now_ms = time.ticks_ms()
    if DRAW_DEBUG:
        draw_debug(img, active_roi, stable_frame_defects, lens)

    if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
        send_line(ujson.dumps(result))
        last_send_ms = now_ms
    if time.ticks_diff(now_ms, last_image_ms) >= USB_IMAGE_INTERVAL_MS:
        send_usb_image(img, active_roi)
        last_image_ms = now_ms
