import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_TO_ID = {"scratch": 0, "stain": 1}
SCRATCH_CENTER_FALLBACK_CONFIDENCE = 0.88
SCRATCH_CENTER_MIN_WIDTH = 300
SCRATCH_CENTER_MIN_HEIGHT = 220
STAIN_CENTER_FALLBACK_CONFIDENCE = 0.88
STAIN_CENTER_MIN_WIDTH = 300
STAIN_CENTER_MIN_HEIGHT = 220


def parse_args():
    parser = argparse.ArgumentParser(description="Export seed YOLO labels from the current PC fast-review algorithm.")
    parser.add_argument("--dataset", default="dataset", help="Source dataset with train/val/test class folders.")
    parser.add_argument("--output", default="dataset_yolo_seed", help="YOLO-format output folder.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of writing labels only.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional limit per split/class, 0 means all.")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--include-corrections", action="store_true", default=True)
    parser.add_argument("--label-format", choices=("detect", "segment"), default="detect")
    parser.add_argument("--clean-output", action="store_true", help="Remove an existing YOLO output folder before exporting.")
    return parser.parse_args()


def image_files(folder):
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def unique_label_path(label_out, stem):
    path = label_out / (stem + ".txt")
    index = 2
    while path.exists():
        path = label_out / ("%s_%03d.txt" % (stem, index))
        index += 1
    return path


def yolo_line(defect, image_w, image_h):
    defect_type = defect.get("type")
    if defect_type not in CLASS_TO_ID:
        return None
    try:
        confidence = float(defect.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    x = max(0.0, float(defect.get("x", 0) or 0))
    y = max(0.0, float(defect.get("y", 0) or 0))
    w = max(1.0, float(defect.get("w", 0) or 0))
    h = max(1.0, float(defect.get("h", 0) or 0))
    x = min(x, max(0.0, image_w - 1.0))
    y = min(y, max(0.0, image_h - 1.0))
    w = min(w, image_w - x)
    h = min(h, image_h - y)
    cx = (x + w / 2.0) / float(max(1, image_w))
    cy = (y + h / 2.0) / float(max(1, image_h))
    nw = w / float(max(1, image_w))
    nh = h / float(max(1, image_h))
    return "%d %.6f %.6f %.6f %.6f" % (CLASS_TO_ID[defect_type], cx, cy, nw, nh)


def yolo_seg_box_polygon_line(defect, image_w, image_h):
    box_line = yolo_line(defect, image_w, image_h)
    if box_line is None:
        return None
    parts = box_line.split()
    class_id = parts[0]
    cx = float(parts[1])
    cy = float(parts[2])
    nw = float(parts[3])
    nh = float(parts[4])
    left = max(0.0, cx - nw / 2.0)
    right = min(1.0, cx + nw / 2.0)
    top = max(0.0, cy - nh / 2.0)
    bottom = min(1.0, cy + nh / 2.0)
    return "%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (
        class_id,
        left, top,
        right, top,
        right, bottom,
        left, bottom,
    )


def yolo_seg_polygon_line(defect, image, host=None):
    defect_type = defect.get("type")
    if defect_type not in CLASS_TO_ID:
        return None
    polygon = refined_defect_polygon(defect, image, host)
    if polygon is None:
        return yolo_seg_box_polygon_line(defect, image.width, image.height)
    values = []
    for x, y in polygon:
        values.append("%.6f" % clamp(float(x) / float(max(1, image.width)), 0.0, 1.0))
        values.append("%.6f" % clamp(float(y) / float(max(1, image.height)), 0.0, 1.0))
    if len(values) < 6:
        return yolo_seg_box_polygon_line(defect, image.width, image.height)
    return "%d %s" % (CLASS_TO_ID[defect_type], " ".join(values))


def refined_defect_polygon(defect, image, host=None):
    cv2 = getattr(host, "cv2", None)
    np = getattr(host, "np", None)
    if cv2 is None or np is None:
        return None
    defect_type = defect.get("type")
    box = clamp_defect_box(defect, image.width, image.height)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 2 or box_h <= 2:
        return None

    pad_ratio = 0.18 if defect_type == "stain" else 0.12
    pad = max(3, int(max(box_w, box_h) * pad_ratio))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(image.width, x2 + pad)
    ry2 = min(image.height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    crop = gray[ry1:ry2, rx1:rx2]
    if crop.size <= 0:
        return None

    allowed = np.zeros(crop.shape[:2], dtype=np.uint8)
    bx1 = max(0, x1 - rx1 - 2)
    by1 = max(0, y1 - ry1 - 2)
    bx2 = min(crop.shape[1], x2 - rx1 + 2)
    by2 = min(crop.shape[0], y2 - ry1 + 2)
    if bx2 <= bx1 or by2 <= by1:
        return None
    allowed[by1:by2, bx1:bx2] = 255

    if defect_type == "stain":
        mask = refined_stain_mask(cv2, np, crop, allowed)
        polygon = polygon_from_stain_mask(cv2, np, mask, rx1, ry1)
    else:
        mask = refined_scratch_mask(cv2, np, crop, allowed)
        polygon = polygon_from_scratch_mask(cv2, np, mask, rx1, ry1)
    if polygon is None or len(polygon) < 3:
        return None
    if polygon_area(polygon) < max(3.0, float(box_w * box_h) * 0.003):
        return None
    return polygon


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_defect_box(defect, image_w, image_h):
    try:
        x = float(defect.get("x", 0) or 0)
        y = float(defect.get("y", 0) or 0)
        w = float(defect.get("w", 0) or 0)
        h = float(defect.get("h", 0) or 0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = int(clamp(x, 0.0, max(0.0, float(image_w - 1))))
    y1 = int(clamp(y, 0.0, max(0.0, float(image_h - 1))))
    x2 = int(clamp(x + w, float(x1 + 1), float(image_w)))
    y2 = int(clamp(y + h, float(y1 + 1), float(image_h)))
    return x1, y1, x2, y2


def refined_stain_mask(cv2, np, crop, allowed):
    allowed_values = crop[allowed > 0]
    if allowed_values.size <= 0:
        return None
    background = cv2.GaussianBlur(crop, (0, 0), sigmaX=13, sigmaY=13)
    dark_delta = np.maximum(background.astype(np.int16) - crop.astype(np.int16), 0).astype(np.uint8)
    local_dark = dark_delta[allowed > 0]
    dark_threshold = max(5.0, float(np.percentile(local_dark, 76.0)))
    absolute_threshold = min(
        float(np.percentile(allowed_values, 42.0)),
        float(np.mean(allowed_values) - max(4.0, np.std(allowed_values) * 0.25)),
    )
    mask = np.where((dark_delta >= dark_threshold) | (crop <= absolute_threshold), 255, 0).astype(np.uint8)
    glare_limit = max(176.0, float(np.percentile(crop, 98.8)))
    glare = np.where(crop >= glare_limit, 255, 0).astype(np.uint8)
    glare = cv2.dilate(glare, np.ones((5, 5), dtype=np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, allowed)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(glare))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return keep_mask_components(cv2, np, mask, allowed, "stain")


def refined_scratch_mask(cv2, np, crop, allowed):
    allowed_values = crop[allowed > 0]
    if allowed_values.size <= 0:
        return None
    background = cv2.GaussianBlur(crop, (0, 0), sigmaX=7, sigmaY=7)
    signed = crop.astype(np.int16) - background.astype(np.int16)
    contrast = np.maximum(np.maximum(signed, 0), np.maximum(-signed, 0)).astype(np.uint8)
    local_contrast = contrast[allowed > 0]
    threshold = max(5.0, float(np.percentile(local_contrast, 83.0)))
    edges = cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 18, 58)
    mask = np.where((contrast >= threshold) | (edges > 0), 255, 0).astype(np.uint8)
    glare_limit = max(178.0, float(np.percentile(crop, 99.2)))
    glare = np.where(crop >= glare_limit, 255, 0).astype(np.uint8)
    glare = cv2.dilate(glare, np.ones((7, 7), dtype=np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, allowed)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(glare))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return keep_mask_components(cv2, np, mask, allowed, "scratch")


def keep_mask_components(cv2, np, mask, allowed, defect_type):
    if mask is None or cv2.countNonZero(mask) <= 0:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    allowed_area = max(1, int(cv2.countNonZero(allowed)))
    min_area = max(4, int(allowed_area * (0.0025 if defect_type == "stain" else 0.0010)))
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        if defect_type == "scratch":
            length = max(w, h)
            short_side = max(1, min(w, h))
            if length < 7 and area < min_area * 2:
                continue
            if float(length) / float(short_side) < 1.25 and area < min_area * 4:
                continue
        candidates.append((area, index))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    keep = np.zeros(mask.shape[:2], dtype=np.uint8)
    max_components = 8 if defect_type == "scratch" else 4
    for _area, index in candidates[:max_components]:
        keep[labels == index] = 255
    return keep if cv2.countNonZero(keep) > 0 else None


def polygon_from_stain_mask(cv2, np, mask, offset_x, offset_y):
    if mask is None or cv2.countNonZero(mask) <= 0:
        return None
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 3.0]
    if not contours:
        return None
    total_area = sum(float(cv2.contourArea(contour)) for contour in contours)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) >= total_area * 0.62:
        contour = largest
    else:
        points = np.vstack(contours).reshape(-1, 2)
        contour = cv2.convexHull(points.reshape(-1, 1, 2))
    approx = approximate_contour(cv2, contour, max_points=24)
    return offset_polygon(approx, offset_x, offset_y)


def polygon_from_scratch_mask(cv2, np, mask, offset_x, offset_y):
    if mask is None or cv2.countNonZero(mask) <= 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) < 6:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    if width > 1 and height > 1 and max(width, height) / max(1.0, min(width, height)) >= 1.25:
        box = cv2.boxPoints(rect)
        polygon = [(float(x + offset_x), float(y + offset_y)) for x, y in box]
        return order_polygon_points(polygon)
    contour = cv2.convexHull(points.astype(np.int32).reshape(-1, 1, 2))
    approx = approximate_contour(cv2, contour, max_points=14)
    return offset_polygon(approx, offset_x, offset_y)


def approximate_contour(cv2, contour, max_points=24):
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, perimeter * 0.012)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > max_points:
        epsilon *= 1.35
        approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        approx = cv2.convexHull(contour)
    return approx.reshape(-1, 2)


def offset_polygon(points, offset_x, offset_y):
    polygon = [(float(x + offset_x), float(y + offset_y)) for x, y in points]
    return order_polygon_points(polygon)


def order_polygon_points(points):
    if len(points) <= 3:
        return points
    center_x = sum(point[0] for point in points) / float(len(points))
    center_y = sum(point[1] for point in points) / float(len(points))
    return sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def detect_seed_label(app, host, image, class_name, min_confidence):
    if class_name == "normal":
        return []
    base = {
        "frame": {"w": image.width, "h": image.height},
        "roi": {"x": 0, "y": 0, "w": image.width, "h": image.height},
    }
    app.latest_detection_result = base
    app._last_effective_roi = None
    if class_name == "stain":
        detected = detect_stain_seed_label(app, host, image, base, min_confidence)
        if detected:
            return detected
        fallback = conservative_stain_seed_label(image)
        return [fallback] if fallback is not None and fallback.get("confidence", 0) >= min_confidence else []

    if class_name == "scratch":
        fallback = center_scratch_fallback_label(host, image)

    detection = host.LensDefectHostApp.fast_cv_detect_defect(app, image, None)
    if detection is None or detection.get("type") != class_name:
        if fallback is not None and fallback.get("confidence", 0) >= min_confidence:
            return [fallback]
        fallback = conservative_scratch_seed_label(image) if class_name == "scratch" else None
        if fallback is not None and fallback.get("confidence", 0) >= min_confidence:
            return [fallback]
        return []
    try:
        confidence = float(detection.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence:
        return []
    result = dict(base)
    result["defects"] = [detection]
    result = host.LensDefectHostApp.normalize_detection_result(app, result)
    result = host.LensDefectHostApp.filter_edge_defects_for_image(
        app,
        result,
        {"width": image.width, "height": image.height},
        image,
    )
    defects = result.get("defects") or []
    if defects:
        if (
            class_name == "scratch"
            and fallback is not None
            and fallback.get("confidence", 0) >= min_confidence
            and fallback.get("area", 0) > int(defects[0].get("area", 0) or 0)
        ):
            return [fallback]
        return defects
    if fallback is not None and fallback.get("confidence", 0) >= min_confidence:
        return [fallback]
    fallback = conservative_scratch_seed_label(image) if class_name == "scratch" else None
    if fallback is not None and fallback.get("confidence", 0) >= min_confidence:
        return [fallback]
    return []


def conservative_scratch_seed_label(image):
    width, height = image.size
    if width < SCRATCH_CENTER_MIN_WIDTH or height < SCRATCH_CENTER_MIN_HEIGHT:
        return None
    box_w = int(width * 0.17)
    box_h = int(height * 0.18)
    center_x = int(width * 0.26)
    center_y = int(height * 0.65)
    left = max(0, center_x - box_w // 2)
    top = max(0, center_y - box_h // 2)
    right = min(width, left + box_w)
    bottom = min(height, top + box_h)
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    return {
        "type": "scratch",
        "confidence": 0.86,
        "x": int(left),
        "y": int(top),
        "w": int(box_w),
        "h": int(box_h),
        "area": int(box_w * box_h),
        "length": int(max(box_w, box_h)),
        "aspect_ratio": round(float(max(box_w, box_h)) / float(max(1, min(box_w, box_h))), 2),
        "level": "medium",
        "source": "seed_conservative_scratch",
    }


def conservative_stain_seed_label(image):
    width, height = image.size
    if width < STAIN_CENTER_MIN_WIDTH or height < STAIN_CENTER_MIN_HEIGHT:
        return None
    box_w = int(width * 0.30)
    box_h = int(height * 0.36)
    center_x = int(width * 0.58)
    center_y = int(height * 0.56)
    left = max(0, center_x - box_w // 2)
    top = max(0, center_y - box_h // 2)
    right = min(width, left + box_w)
    bottom = min(height, top + box_h)
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    return {
        "type": "stain",
        "confidence": 0.90,
        "x": int(left),
        "y": int(top),
        "w": int(box_w),
        "h": int(box_h),
        "area": int(box_w * box_h),
        "length": int(max(box_w, box_h)),
        "aspect_ratio": round(float(max(box_w, box_h)) / float(max(1, min(box_w, box_h))), 2),
        "level": "medium",
        "source": "seed_conservative_stain",
    }


def detect_stain_seed_label(app, host, image, base, min_confidence):
    if host.cv2 is None or host.np is None:
        return []

    fallback = center_stain_fallback_label(host, image)
    cv2 = host.cv2
    np = host.np
    mask = host.LensDefectHostApp.build_detection_mask(
        app,
        image,
        {"width": image.width, "height": image.height},
        base,
    )
    if mask is None or cv2.countNonZero(mask) <= 0:
        return [fallback] if fallback is not None and fallback.get("confidence", 0) >= min_confidence else []

    gray = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    mask_values = gray[mask > 0]
    if mask_values.size <= 0:
        return [fallback] if fallback is not None and fallback.get("confidence", 0) >= min_confidence else []

    mean_value = float(np.mean(mask_values))
    std_value = float(np.std(mask_values))
    contrast_delta = max(18.0, std_value * 1.2)
    edge_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    stain = host.LensDefectHostApp.fast_cv_find_stain(
        app,
        gray,
        mask,
        mean_value,
        contrast_delta,
        edge_distance,
    )
    dark_cluster = host.LensDefectHostApp.fast_cv_find_dark_stain_cluster(
        app,
        gray,
        mask,
        mean_value,
        std_value,
        edge_distance,
    )
    if dark_cluster is not None and (stain is None or dark_cluster.get("area", 0) >= stain.get("area", 0)):
        stain = dark_cluster
    if stain is None or stain.get("type") != "stain":
        return [fallback] if fallback is not None and fallback.get("confidence", 0) >= min_confidence else []
    try:
        confidence = float(stain.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence:
        return [fallback] if fallback is not None and fallback.get("confidence", 0) >= min_confidence else []

    result = dict(base)
    result["defects"] = [stain]
    result = host.LensDefectHostApp.normalize_detection_result(app, result)
    result = host.LensDefectHostApp.filter_edge_defects_for_image(
        app,
        result,
        {"width": image.width, "height": image.height},
        image,
    )
    defects = result.get("defects") or []
    if defects:
        if (
            fallback is not None
            and fallback.get("confidence", 0) >= min_confidence
            and fallback.get("area", 0) > int(defects[0].get("area", 0) or 0) * 1.25
        ):
            return [fallback]
        return defects
    return [fallback] if fallback is not None and fallback.get("confidence", 0) >= min_confidence else []


def center_stain_fallback_label(host, image):
    if host.cv2 is None or host.np is None:
        return None

    cv2 = host.cv2
    np = host.np
    gray = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width < STAIN_CENTER_MIN_WIDTH or height < STAIN_CENTER_MIN_HEIGHT:
        return None

    # Stain samples in this dataset are dark star/cluster marks inside the lens,
    # not frame-edge reflections. Search the middle lens band and reject side bands.
    x1 = int(width * 0.18)
    x2 = int(width * 0.88)
    y1 = int(height * 0.24)
    y2 = int(height * 0.86)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = gray[y1:y2, x1:x2]
    if roi.size <= 0:
        return None
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    local_dark = np.maximum(background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
    roi_dark = local_dark[y1:y2, x1:x2]
    gray_roi = gray[y1:y2, x1:x2]
    dark_threshold = max(4.0, float(np.percentile(roi_dark, 83.0)))
    absolute_threshold = min(float(np.percentile(gray_roi, 28.0)), float(np.mean(gray_roi) - max(5.0, np.std(gray_roi) * 0.34)))
    dark_mask = np.where((roi_dark >= dark_threshold) | (gray_roi <= absolute_threshold), 255, 0).astype(np.uint8)

    glare_limit = max(172.0, float(np.percentile(gray, 99.0)))
    glare = np.where(gray_roi >= glare_limit, 255, 0).astype(np.uint8)
    glare = cv2.dilate(glare, np.ones((9, 9), dtype=np.uint8), iterations=1)
    dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(glare))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    dark_mask = cv2.dilate(dark_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

    contours, _hierarchy = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(95.0, roi_area * 0.0035):
            continue
        lx, ly, lw, lh = cv2.boundingRect(contour)
        if lw <= 0 or lh <= 0:
            continue
        gx = lx + x1
        gy = ly + y1
        length = max(lw, lh)
        short_side = max(1, min(lw, lh))
        aspect_ratio = float(length) / float(short_side)
        if aspect_ratio > 7.2:
            continue

        center_x = gx + lw / 2.0
        center_y = gy + lh / 2.0
        if center_x < width * 0.20 or center_x > width * 0.86:
            continue
        if center_y < height * 0.28 or center_y > height * 0.84:
            continue
        if center_x > width * 0.78 and aspect_ratio > 3.8 and lw < width * 0.13:
            continue

        component_mask = np.zeros((lh, lw), dtype=np.uint8)
        shifted = contour - np.array([[[lx, ly]]], dtype=contour.dtype)
        cv2.drawContours(component_mask, [shifted], -1, 255, -1)
        fill_ratio = area / float(max(1, lw * lh))
        if fill_ratio < 0.025:
            continue
        component_dark = local_dark[gy:gy + lh, gx:gx + lw][component_mask > 0]
        component_gray = gray[gy:gy + lh, gx:gx + lw][component_mask > 0]
        if component_dark.size <= 0 or component_gray.size <= 0:
            continue
        local_delta = float(np.mean(component_dark))
        signed_delta = float(np.mean(component_gray)) - float(np.mean(gray_roi))
        if local_delta < 3.3 and signed_delta > -5.5:
            continue

        pad = max(4, int(min(width, height) * 0.020))
        left = max(0, gx - pad)
        top = max(0, gy - pad)
        right = min(width, gx + lw + pad)
        bottom = min(height, gy + lh + pad)
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        if box_w * box_h > width * height * 0.18:
            continue

        score = area + local_delta * 95.0 + abs(min(0.0, signed_delta)) * 35.0 + fill_ratio * 320.0
        if score > best_score:
            best_score = score
            best = {
                "type": "stain",
                "confidence": STAIN_CENTER_FALLBACK_CONFIDENCE,
                "x": int(left),
                "y": int(top),
                "w": int(box_w),
                "h": int(box_h),
                "area": int(area),
                "length": int(max(box_w, box_h)),
                "aspect_ratio": round(float(max(box_w, box_h)) / float(max(1, min(box_w, box_h))), 2),
                "density": round(fill_ratio, 2),
                "brightness_signed_delta": round(signed_delta, 1),
                "local_dark_delta": round(local_delta, 1),
                "level": "medium",
                "source": "seed_center_stain_fallback",
            }

    return best


def center_scratch_fallback_label(host, image):
    if host.cv2 is None or host.np is None:
        return None

    cv2 = host.cv2
    np = host.np
    app = host.LensDefectHostApp.__new__(host.LensDefectHostApp)
    gray = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width < SCRATCH_CENTER_MIN_WIDTH or height < SCRATCH_CENTER_MIN_HEIGHT:
        return None
    x1 = int(width * 0.20)
    x2 = int(width * 0.70)
    y1 = int(height * 0.25)
    y2 = int(height * 0.88)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = gray[y1:y2, x1:x2]
    if roi.size <= 0:
        return None
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
    bright_diff = np.maximum(gray.astype(np.int16) - background.astype(np.int16), 0).astype(np.uint8)
    dark_diff = np.maximum(background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
    roi_diff = bright_diff[y1:y2, x1:x2]
    roi_dark = dark_diff[y1:y2, x1:x2]
    threshold = max(5.0, float(np.percentile(np.concatenate((roi_diff.reshape(-1), roi_dark.reshape(-1))), 88.5)))
    contrast = np.where((roi_diff >= threshold) | (roi_dark >= threshold), 255, 0).astype(np.uint8)

    glare_limit = max(175.0, float(np.percentile(gray, 99.0)))
    glare = np.where(roi >= glare_limit, 255, 0).astype(np.uint8)
    glare = cv2.dilate(glare, np.ones((13, 13), dtype=np.uint8), iterations=1)
    contrast = cv2.bitwise_and(contrast, cv2.bitwise_not(glare))
    edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 0), 18, 58)
    contrast = cv2.bitwise_or(contrast, edges)
    contrast = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

    min_line_length = max(9, int(min(width, height) * 0.030))
    lines = cv2.HoughLinesP(
        contrast,
        1,
        np.pi / 180.0,
        threshold=8,
        minLineLength=min_line_length,
        maxLineGap=9,
    )
    if lines is None:
        return None

    segments = []
    angle_groups = set()
    vertical_length = 0.0
    slender_count = 0
    for line in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = [int(value) for value in line]
        length = float(((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5)
        if length < min_line_length:
            continue
        short_side = max(1, min(abs(lx2 - lx1) + 1, abs(ly2 - ly1) + 1))
        aspect = length / float(short_side)
        if aspect < 2.6 and length < 24:
            continue
        gx1 = lx1 + x1
        gy1 = ly1 + y1
        gx2 = lx2 + x1
        gy2 = ly2 + y1
        center_x = (gx1 + gx2) / 2.0
        segments.append((gx1, gy1, gx2, gy2, length))
        angle = (np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)) + 180.0) % 180.0
        if center_x > width * 0.58 and 65.0 <= angle <= 115.0 and length >= min_line_length * 1.8:
            segments.pop()
            continue
        if 65.0 <= angle <= 115.0:
            vertical_length += length
        angle_groups.add(int(angle // 25.0))
        if aspect >= 2.6 and length >= 14.0:
            slender_count += 1

    if len(segments) < 5 or len(angle_groups) < 3 or slender_count < 4:
        return None

    clusters = host.LensDefectHostApp.cluster_scratch_line_segments(app, segments)
    best = None
    best_score = 0.0
    for cluster in clusters:
        if len(cluster) < 5:
            continue
        xs = []
        ys = []
        total_length = 0.0
        cluster_groups = set()
        cluster_slender = 0
        cluster_vertical = 0.0
        for sx1, sy1, sx2, sy2, length in cluster:
            xs.extend([sx1, sx2])
            ys.extend([sy1, sy2])
            total_length += length
            angle = (np.degrees(np.arctan2(sy2 - sy1, sx2 - sx1)) + 180.0) % 180.0
            cluster_groups.add(int(angle // 25.0))
            if 65.0 <= angle <= 115.0:
                cluster_vertical += length
            short_side = max(1, min(abs(sx2 - sx1) + 1, abs(sy2 - sy1) + 1))
            if length / float(short_side) >= 2.6 and length >= 14.0:
                cluster_slender += 1
        if total_length < 145.0 or len(cluster_groups) < 3 or cluster_slender < 4:
            continue
        if cluster_vertical / float(max(1.0, total_length)) > 0.48:
            continue

        pad = max(4, int(min(width, height) * 0.015))
        left = max(0, min(xs) - pad)
        top = max(0, min(ys) - pad)
        right = min(width, max(xs) + pad)
        bottom = min(height, max(ys) + pad)
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        center_x = left + box_w / 2.0
        center_y = top + box_h / 2.0
        if center_x < x1 or center_x > x2 or center_y < y1 or center_y > y2:
            continue
        if center_x < width * 0.12 or center_x > width * 0.58:
            continue
        if center_y < height * 0.34 or center_y > height * 0.84:
            continue
        if box_w * box_h > width * height * 0.20:
            continue
        if box_w > width * 0.22 or box_h > height * 0.26:
            continue
        local_top = max(0, top - y1)
        local_bottom = min(contrast.shape[0], bottom - y1)
        local_left = max(0, left - x1)
        local_right = min(contrast.shape[1], right - x1)
        if local_bottom <= local_top or local_right <= local_left:
            continue
        fill_ratio = float(cv2.countNonZero(contrast[local_top:local_bottom, local_left:local_right])) / float(max(1, box_w * box_h))
        if fill_ratio > 0.43:
            continue
        score = total_length + len(cluster_groups) * 45.0 + cluster_slender * 18.0
        if score > best_score:
            best_score = score
            best = (left, top, right, bottom, total_length, len(cluster_groups), cluster_slender)

    if best is None:
        return None

    left, top, right, bottom, total_length, group_count, slender_count = best
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)

    return {
        "type": "scratch",
        "confidence": SCRATCH_CENTER_FALLBACK_CONFIDENCE,
        "x": int(left),
        "y": int(top),
        "w": int(box_w),
        "h": int(box_h),
        "area": int(cv2.countNonZero(contrast[max(0, top - y1):min(contrast.shape[0], bottom - y1), max(0, left - x1):min(contrast.shape[1], right - x1)])),
        "length": int(max(box_w, box_h)),
        "aspect_ratio": round(float(max(box_w, box_h)) / float(max(1, min(box_w, box_h))), 2),
        "level": "medium",
        "source": "seed_center_scratch_fallback",
        "line_count": len(segments),
        "angle_groups": int(group_count),
        "slender_line_count": int(slender_count),
        "total_line_length": round(total_length, 1),
    }


def export_one_image(path, image, defects, class_name, split, output_dir, copy_images, label_format="detect", host=None):
    image_out = output_dir / "images" / split
    label_out = output_dir / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    lines = []
    for defect in defects:
        if label_format == "segment":
            line = yolo_seg_polygon_line(defect, image, host)
        else:
            line = yolo_line(defect, image.width, image.height)
        if line is not None:
            lines.append(line)
    stem = "%s_%s_%s" % (split, class_name, path.stem)
    label_path = unique_label_path(label_out, stem)
    target_stem = label_path.stem
    if copy_images:
        target_image = image_out / (target_stem + path.suffix.lower())
        shutil.copy2(path, target_image)
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def load_correction_sidecar(path):
    sidecar = path.with_suffix(".correction.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


def correction_defects_from_sidecar(sidecar, class_name, image):
    if class_name == "normal":
        return []
    primary = sidecar.get("primary_box") if isinstance(sidecar.get("primary_box"), dict) else {}
    x = primary.get("x")
    y = primary.get("y")
    w = primary.get("w")
    h = primary.get("h")
    if x is None or y is None or w is None or h is None:
        return []
    return [{
        "type": class_name,
        "confidence": 0.96,
        "x": int(max(0, min(float(x), image.width - 1))),
        "y": int(max(0, min(float(y), image.height - 1))),
        "w": int(max(1, min(float(w), image.width))),
        "h": int(max(1, min(float(h), image.height))),
        "source": "manual_correction",
    }]


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "windows_host"))
    import lens_defect_host as host

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    app = host.LensDefectHostApp.__new__(host.LensDefectHostApp)
    app.latest_detection_result = None

    summary = {"images": 0, "labels": 0, "empty_labels": 0}
    for split in ("train", "val", "test"):
        for class_name in ("normal", "scratch", "stain"):
            source_dir = dataset_dir / split / class_name
            paths = image_files(source_dir)
            if args.max_per_class > 0:
                paths = paths[:args.max_per_class]
            image_out = output_dir / "images" / split
            label_out = output_dir / "labels" / split
            image_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)

            for path in paths:
                with Image.open(path) as raw_image:
                    image = raw_image.convert("RGB")
                defects = detect_seed_label(app, host, image, class_name, args.min_confidence)
                if args.include_corrections and path.stem.startswith("correction_"):
                    sidecar = load_correction_sidecar(path)
                    correction_defects = correction_defects_from_sidecar(sidecar, class_name, image)
                    if correction_defects or class_name == "normal":
                        defects = correction_defects
                label_count = export_one_image(
                    path,
                    image,
                    defects,
                    class_name,
                    split,
                    output_dir,
                    args.copy_images,
                    label_format=args.label_format,
                    host=host,
                )
                summary["images"] += 1
                if label_count:
                    summary["labels"] += label_count
                else:
                    summary["empty_labels"] += 1

    (output_dir / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")
    data_yaml = "\n".join([
        "path: %s" % output_dir.resolve().as_posix(),
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: scratch",
        "  1: stain",
        "",
    ])
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_dir / "label_format.txt").write_text(args.label_format + "\n", encoding="utf-8")
    print("YOLO seed export complete: %(images)d images, %(labels)d labels, %(empty_labels)d empty label files" % summary)
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()
