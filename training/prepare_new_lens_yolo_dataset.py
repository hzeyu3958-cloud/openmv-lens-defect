import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Append new-lens correction/capture samples to a YOLO seed dataset.")
    parser.add_argument("--base", default="dataset_yolo_seed_refined")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="outputs/yolo_seed_new_lens")
    parser.add_argument("--date-prefix", default="20260604")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--sample-base-train", type=int, default=0)
    parser.add_argument("--sample-base-val", type=int, default=0)
    parser.add_argument("--sample-base-test", type=int, default=0)
    parser.add_argument("--sample-base-class-train", type=int, default=0)
    parser.add_argument("--sample-base-class-val", type=int, default=0)
    parser.add_argument("--sample-base-class-test", type=int, default=0)
    parser.add_argument("--sample-base-background-train", type=int, default=0)
    parser.add_argument("--sample-base-background-val", type=int, default=0)
    parser.add_argument("--sample-base-background-test", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260604)
    return parser.parse_args()


def yolo_line(class_index, box, image_w, image_h):
    x, y, w, h = box
    x = max(0, min(int(x), image_w - 1))
    y = max(0, min(int(y), image_h - 1))
    w = max(1, min(int(w), image_w - x))
    h = max(1, min(int(h), image_h - y))
    cx = (x + w / 2.0) / float(max(1, image_w))
    cy = (y + h / 2.0) / float(max(1, image_h))
    bw = w / float(max(1, image_w))
    bh = h / float(max(1, image_h))
    return f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def bright_ring_scratch_box(image_path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape[:2]
    if image_w < 80 or image_h < 80:
        return None

    x1 = int(image_w * 0.06)
    x2 = int(image_w * 0.74)
    y1 = int(image_h * 0.18)
    y2 = int(image_h * 0.82)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_gray = gray[y1:y2, x1:x2]
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=13, sigmaY=13)
    bright = np.maximum(gray.astype(np.int16) - background.astype(np.int16), 0).astype(np.uint8)
    roi_bright = bright[y1:y2, x1:x2]
    if roi_gray.size <= 0 or roi_bright.size <= 0:
        return None

    bright_threshold = max(12.0, float(np.percentile(roi_bright, 90.0)))
    gray_threshold = max(118.0, float(np.percentile(roi_gray, 96.0)))
    mask = np.where((roi_bright >= bright_threshold) | (roi_gray >= gray_threshold), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 50.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        if not (15 <= w <= 150 and 15 <= h <= 150 and aspect <= 2.4):
            continue
        center_x = x1 + x + w / 2.0
        center_y = y1 + y + h / 2.0
        if center_x < image_w * 0.10 or center_x > image_w * 0.70:
            continue
        if center_y < image_h * 0.20 or center_y > image_h * 0.78:
            continue
        candidates.append((area, x1 + x, y1 + y, w, h))

    if not candidates:
        return None
    _area, x, y, w, h = max(candidates, key=lambda item: item[0])
    pad = max(4, int(max(w, h) * 0.12))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image_w, x + w + pad)
    bottom = min(image_h, y + h + pad)
    return left, top, max(1, right - left), max(1, bottom - top)


def bright_cross_scratch_box(image_path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape[:2]
    if image_w < 160 or image_h < 120:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=11, sigmaY=11)
    bright = np.maximum(enhanced.astype(np.int16) - background.astype(np.int16), 0).astype(np.uint8)
    top_hat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
    )
    response = np.maximum(bright, top_hat)

    allowed = np.zeros_like(gray, dtype=np.uint8)
    allowed[
        int(image_h * 0.01):int(image_h * 0.86),
        int(image_w * 0.02):int(image_w * 0.93),
    ] = 255
    values = response[allowed > 0]
    if values.size <= 0 or float(np.percentile(values, 98.0)) < 22.0:
        return None

    threshold = max(10.0, float(np.percentile(values, 88.0)))
    candidate = np.where((response >= threshold) & (allowed > 0), 255, 0).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0), 20, 64)
    edge_support = cv2.bitwise_and(
        edges,
        cv2.dilate(candidate, np.ones((3, 3), dtype=np.uint8), iterations=1),
    )
    candidate = cv2.bitwise_or(candidate, edge_support)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    if cv2.countNonZero(candidate) < 16:
        return None

    min_line_length = max(18, int(min(image_w, image_h) * 0.055))
    lines = cv2.HoughLinesP(
        candidate,
        1,
        np.pi / 180.0,
        threshold=7,
        minLineLength=min_line_length,
        maxLineGap=max(8, int(min(image_w, image_h) * 0.03)),
    )
    if lines is None:
        return None

    segments = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length < min_line_length:
            continue
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if center_x < image_w * 0.03 or center_x > image_w * 0.91:
            continue
        if center_y < image_h * 0.02 or center_y > image_h * 0.84:
            continue
        samples = max(5, int(length / 6.0))
        active_count = 0
        line_responses = []
        for sample_index in range(samples + 1):
            ratio = float(sample_index) / float(samples)
            px = max(0, min(image_w - 1, int(round(x1 + (x2 - x1) * ratio))))
            py = max(0, min(image_h - 1, int(round(y1 + (y2 - y1) * ratio))))
            if candidate[py, px] > 0:
                active_count += 1
            line_responses.append(int(response[py, px]))
        if float(active_count) / float(samples + 1) < 0.18:
            continue
        if line_responses and float(np.percentile(line_responses, 65.0)) < threshold * 0.35:
            continue
        angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
        if (
            (center_x < image_w * 0.10 or center_x > image_w * 0.86)
            and length > min(image_w, image_h) * 0.22
        ):
            continue
        segments.append((x1, y1, x2, y2, length, angle))

    if len(segments) < 6:
        return None

    intersections = []
    segment_intersection_counts = [0 for _item in segments]
    tolerance = max(8, int(min(image_w, image_h) * 0.035))
    for first_index, first in enumerate(segments):
        x1, y1, x2, y2, _first_length, first_angle = first
        for second_index in range(first_index + 1, len(segments)):
            x3, y3, x4, y4, _second_length, second_angle = segments[second_index]
            angle_gap = abs(first_angle - second_angle)
            angle_gap = min(angle_gap, 180.0 - angle_gap)
            if angle_gap < 20.0:
                continue
            denominator = float((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            if abs(denominator) < 1e-6:
                continue
            px = (
                (x1 * y2 - y1 * x2) * (x3 - x4)
                - (x1 - x2) * (x3 * y4 - y3 * x4)
            ) / denominator
            py = (
                (x1 * y2 - y1 * x2) * (y3 - y4)
                - (y1 - y2) * (x3 * y4 - y3 * x4)
            ) / denominator
            if (
                px < min(x1, x2) - tolerance
                or px > max(x1, x2) + tolerance
                or px < min(x3, x4) - tolerance
                or px > max(x3, x4) + tolerance
                or py < min(y1, y2) - tolerance
                or py > max(y1, y2) + tolerance
                or py < min(y3, y4) - tolerance
                or py > max(y3, y4) + tolerance
            ):
                continue
            if px < image_w * 0.04 or px > image_w * 0.90 or py < image_h * 0.02 or py > image_h * 0.82:
                continue
            intersections.append((px, py))
            segment_intersection_counts[first_index] += 1
            segment_intersection_counts[second_index] += 1

    if len(intersections) < 3:
        return None

    points = np.array(intersections, dtype=np.float32)
    x_low, x_high = np.percentile(points[:, 0], [5.0, 95.0])
    y_low, y_high = np.percentile(points[:, 1], [5.0, 95.0])
    cluster_pad = max(48, int(min(image_w, image_h) * 0.22))
    selected = []
    for index, segment in enumerate(segments):
        x1, y1, x2, y2, length, _angle = segment
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if (
            segment_intersection_counts[index] >= 3
            and x_low - cluster_pad <= center_x <= x_high + cluster_pad
            and y_low - cluster_pad <= center_y <= y_high + cluster_pad
            and length >= min_line_length * 0.85
        ):
            selected.append(segment)

    if len(selected) < 4:
        return None

    xs = []
    ys = []
    total_length = 0.0
    angle_groups = set()
    for x1, y1, x2, y2, length, angle in selected:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        total_length += length
        angle_groups.add(int(angle // 22.5))
    if total_length < min(image_w, image_h) * 0.95 or len(angle_groups) < 3:
        return None

    compact_x_low, compact_x_high = np.percentile(points[:, 0], [10.0, 90.0])
    compact_y_low, compact_y_high = np.percentile(points[:, 1], [10.0, 90.0])
    compact_pad = max(22, int(min(image_w, image_h) * 0.09))
    left = max(0, int(round(compact_x_low)) - compact_pad)
    top = max(0, int(round(compact_y_low)) - compact_pad)
    right = min(image_w, int(round(compact_x_high)) + compact_pad + 1)
    bottom = min(image_h, int(round(compact_y_high)) + compact_pad + 1)
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    box_area_ratio = float(box_w * box_h) / float(max(1, image_w * image_h))
    if box_area_ratio > 0.36:
        return None
    local_candidate = candidate[top:bottom, left:right]
    fill_ratio = float(cv2.countNonZero(local_candidate)) / float(max(1, local_candidate.size))
    if fill_ratio > 0.34:
        return None
    return int(left), int(top), int(box_w), int(box_h)


def thin_scratch_box(image_path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape[:2]
    if image_w < 80 or image_h < 80:
        return None

    x1 = int(image_w * 0.06)
    x2 = int(image_w * 0.86)
    y1 = int(image_h * 0.08)
    y2 = int(image_h * 0.90)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size <= 0:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_gray)
    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=5, sigmaY=5)
    dark = np.maximum(background.astype(np.int16) - enhanced.astype(np.int16), 0).astype(np.uint8)
    bright = np.maximum(enhanced.astype(np.int16) - background.astype(np.int16), 0).astype(np.uint8)
    response = cv2.addWeighted(dark, 0.72, bright, 0.28, 0.0)

    response_threshold = max(5.0, float(np.percentile(response, 88.0)))
    candidate = np.where(response >= response_threshold, 255, 0).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    if cv2.countNonZero(candidate) < 8:
        return None

    edges = cv2.Canny(enhanced, 20, 72)
    edges = cv2.bitwise_or(edges, candidate)
    min_line_length = max(18, int(min(image_w, image_h) * 0.055))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=7,
        minLineLength=min_line_length,
        maxLineGap=max(5, int(min(image_w, image_h) * 0.025)),
    )
    if lines is None:
        return None

    segments = []
    angle_totals = {}
    for line in lines[:, 0, :]:
        x_start, y_start, x_end, y_end = [int(value) for value in line]
        length = float(((x_end - x_start) ** 2 + (y_end - y_start) ** 2) ** 0.5)
        if length < min_line_length:
            continue
        line_mask = np.zeros_like(roi_gray, dtype=np.uint8)
        cv2.line(line_mask, (x_start, y_start), (x_end, y_end), 255, 2)
        line_response = response[line_mask > 0]
        if line_response.size and float(np.percentile(line_response, 65.0)) < response_threshold * 0.55:
            continue
        angle = (np.degrees(np.arctan2(y_end - y_start, x_end - x_start)) + 180.0) % 180.0
        angle_group = int(angle // 18.0)
        angle_totals[angle_group] = angle_totals.get(angle_group, 0.0) + length
        segments.append((x_start, y_start, x_end, y_end, length, angle_group))

    if not segments:
        return None
    dominant_group, dominant_total = max(angle_totals.items(), key=lambda item: item[1])
    dominant_segments = [item for item in segments if item[5] == dominant_group]
    if dominant_total < max(22.0, min_line_length):
        return None

    xs = []
    ys = []
    for x_start, y_start, x_end, y_end, _length, _group in dominant_segments:
        xs.extend([x_start, x_end])
        ys.extend([y_start, y_end])
    left = max(0, min(xs) - 4)
    top = max(0, min(ys) - 4)
    right = min(x2 - x1 - 1, max(xs) + 5)
    bottom = min(y2 - y1 - 1, max(ys) + 5)
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    aspect = float(max(box_w, box_h)) / float(max(1, min(box_w, box_h)))
    if aspect < 1.55:
        return None

    box_area_ratio = float(box_w * box_h) / float(max(1, image_w * image_h))
    if box_area_ratio > 0.09:
        return None
    local_candidate = candidate[top:bottom + 1, left:right + 1]
    fill_ratio = float(cv2.countNonZero(local_candidate)) / float(max(1, local_candidate.size))
    if fill_ratio > 0.50:
        return None

    pad = max(4, int(max(box_w, box_h) * 0.10))
    global_left = max(0, x1 + left - pad)
    global_top = max(0, y1 + top - pad)
    global_right = min(image_w, x1 + right + pad)
    global_bottom = min(image_h, y1 + bottom + pad)
    return global_left, global_top, max(1, global_right - global_left), max(1, global_bottom - global_top)


def dark_stain_box(image_path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape[:2]
    if image_w < 80 or image_h < 80:
        return None

    x1 = int(image_w * 0.00)
    x2 = int(image_w * 0.68)
    y1 = int(image_h * 0.20)
    y2 = int(image_h * 0.98)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size <= 0:
        return None
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    dark = np.maximum(background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
    roi_dark = dark[y1:y2, x1:x2]

    gray_threshold = min(
        float(np.percentile(roi_gray, 30.0)),
        float(np.mean(roi_gray) - max(9.0, float(np.std(roi_gray)) * 0.32)),
    )
    dark_threshold = max(13.0, float(np.percentile(roi_dark, 83.0)))
    mask = np.where(
        ((roi_gray <= gray_threshold) & (roi_dark >= 6)) | (roi_dark >= dark_threshold),
        255,
        0,
    ).astype(np.uint8)
    mask[roi_gray >= 205] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 350.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 25 or h < 28:
            continue
        global_x = x1 + x
        global_y = y1 + y
        center_x = (global_x + w / 2.0) / float(image_w)
        center_y = (global_y + h / 2.0) / float(image_h)
        if center_x > 0.62 or center_y < 0.27 or center_y > 0.92:
            continue
        if w * h > image_w * image_h * 0.26:
            continue
        crop_gray = gray[global_y:global_y + h, global_x:global_x + w]
        crop_dark = dark[global_y:global_y + h, global_x:global_x + w]
        darkness = max(0.0, float(np.mean(roi_gray)) - float(np.mean(crop_gray)))
        score = area * (1.0 + darkness / 30.0) * (1.0 + float(np.mean(crop_dark)) / 18.0)
        candidates.append((score, global_x, global_y, w, h))

    if not candidates:
        return None
    _score, x, y, w, h = max(candidates, key=lambda item: item[0])
    pad = max(5, int(max(w, h) * 0.08))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image_w, x + w + pad)
    bottom = min(image_h, y + h + pad)
    return left, top, max(1, right - left), max(1, bottom - top)


def date_prefixes(value):
    prefixes = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return prefixes or [""]


def image_paths(folder, prefixes):
    if not folder.exists():
        return []
    prefixes = date_prefixes(prefixes)
    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and any(prefix in path.name for prefix in prefixes)
    )


def copy_image_with_label(source_image, output_dir, split, class_name, label_text):
    stem = f"newlens_{split}_{class_name}_{source_image.stem}"
    suffix = source_image.suffix.lower()
    image_target = output_dir / "images" / split / f"{stem}{suffix}"
    label_target = output_dir / "labels" / split / f"{stem}.txt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, image_target)
    label_target.write_text(label_text, encoding="utf-8")


def write_cv_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    extension = ".jpg" if suffix in (".jpg", ".jpeg") else suffix
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"Failed to encode augmented image: {path}")
    encoded.tofile(str(path))


def augment_bright_cross_scratch_image(image):
    variants = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for clip in (1.4, 1.8, 2.4):
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
        variants.append(("clahe%02d" % int(clip * 10), cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))
    for name, alpha, beta in (
        ("bright08", 1.08, 8),
        ("bright14", 1.14, 12),
        ("dark92", 0.92, -6),
        ("dark86", 0.86, -10),
        ("contrast", 1.22, -8),
        ("flat", 0.78, 18),
    ):
        variants.append((name, cv2.convertScaleAbs(image, alpha=alpha, beta=beta)))
    for gamma in (0.78, 0.88, 1.14, 1.28):
        table = np.array([
            min(255, max(0, int(((value / 255.0) ** gamma) * 255.0 + 0.5)))
            for value in range(256)
        ], dtype=np.uint8)
        variants.append(("gamma%03d" % int(gamma * 100), cv2.LUT(image, table)))
    for sigma in (0.45, 0.75, 1.0):
        blur = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)
        variants.append(("soft%02d" % int(sigma * 100), cv2.addWeighted(image, 0.82, blur, 0.18, 0.0)))
    for strength in (0.25, 0.38, 0.52):
        sharp = cv2.addWeighted(
            image,
            1.0 + strength,
            cv2.GaussianBlur(image, (0, 0), sigmaX=1.0),
            -strength,
            0.0,
        )
        variants.append(("sharp%02d" % int(strength * 100), np.clip(sharp, 0, 255).astype(np.uint8)))
    for quality in (55, 62, 70):
        success, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success:
            decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            if decoded is not None:
                variants.append(("jpeg%d" % quality, decoded))
    return variants


def copy_bright_cross_scratch_augments(source_image, output_dir, split, class_name, label_text):
    if split != "train" or class_name != "scratch" or not label_text.strip():
        return 0
    if "screenshot_cross" not in source_image.stem and "bright_cross" not in source_image.stem:
        return 0
    try:
        encoded = np.fromfile(str(source_image), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    if image is None:
        return 0

    stem = f"newlens_{split}_{class_name}_{source_image.stem}"
    suffix = source_image.suffix.lower()
    count = 0
    for name, variant in augment_bright_cross_scratch_image(image):
        image_target = output_dir / "images" / split / f"{stem}_aug_{name}{suffix}"
        label_target = output_dir / "labels" / split / f"{stem}_aug_{name}.txt"
        write_cv_image(image_target, variant)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text(label_text, encoding="utf-8")
        count += 1
    return count


def base_image_paths(base_dir, split):
    images_dir = base_dir / "images" / split
    if not images_dir.exists():
        return []
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def matching_base_label(base_dir, split, image_path):
    return base_dir / "labels" / split / f"{image_path.stem}.txt"


def has_label_text(label_path):
    if not label_path.exists():
        return False
    try:
        return bool(label_path.read_text(encoding="utf-8").strip())
    except UnicodeDecodeError:
        return bool(label_path.read_text().strip())


def label_classes(label_path):
    if not label_path.exists():
        return set()
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = label_path.read_text()
    classes = set()
    for line in text.splitlines():
        parts = line.split()
        if parts:
            classes.add(parts[0])
    return classes


def choose_base_subset(base_dir, split, limit, rng):
    images = base_image_paths(base_dir, split)
    if limit <= 0 or limit >= len(images):
        return images

    labelled = []
    backgrounds = []
    for image_path in images:
        label_path = matching_base_label(base_dir, split, image_path)
        if has_label_text(label_path):
            labelled.append(image_path)
        else:
            backgrounds.append(image_path)

    background_count = min(len(backgrounds), max(1, int(limit * 0.20))) if backgrounds else 0
    labelled_count = min(len(labelled), max(0, limit - background_count))
    selected = []
    if labelled_count:
        selected.extend(rng.sample(labelled, labelled_count))
    if background_count:
        selected.extend(rng.sample(backgrounds, background_count))

    remaining = min(limit, len(images)) - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        leftovers = [path for path in images if path not in selected_set]
        selected.extend(rng.sample(leftovers, min(remaining, len(leftovers))))
    return sorted(selected)


def choose_base_balanced_subset(base_dir, split, total_limit, class_limit, background_limit, rng):
    if class_limit <= 0 and background_limit <= 0:
        return choose_base_subset(base_dir, split, total_limit, rng)

    images = base_image_paths(base_dir, split)
    if not images:
        return []

    by_class = {"0": [], "1": []}
    backgrounds = []
    labelled = []
    for image_path in images:
        classes = label_classes(matching_base_label(base_dir, split, image_path))
        if classes:
            labelled.append(image_path)
        else:
            backgrounds.append(image_path)
        for class_name in by_class:
            if class_name in classes:
                by_class[class_name].append(image_path)

    selected = []
    selected_set = set()
    for class_name in sorted(by_class):
        pool = [path for path in by_class[class_name] if path not in selected_set]
        if class_limit > 0 and pool:
            take = min(class_limit, len(pool))
            chosen = rng.sample(pool, take)
            selected.extend(chosen)
            selected_set.update(chosen)

    if background_limit > 0 and backgrounds:
        pool = [path for path in backgrounds if path not in selected_set]
        take = min(background_limit, len(pool))
        chosen = rng.sample(pool, take)
        selected.extend(chosen)
        selected_set.update(chosen)

    if total_limit > 0 and len(selected) < min(total_limit, len(images)):
        leftovers = [path for path in images if path not in selected_set]
        take = min(total_limit - len(selected), len(leftovers))
        if take > 0:
            selected.extend(rng.sample(leftovers, take))

    if not selected and total_limit <= 0:
        return images
    return sorted(selected)


def copy_base_subset(base_dir, output_dir, limits, seed):
    rng = random.Random(seed)
    summary = {}
    for split in ("train", "val", "test"):
        selected = choose_base_subset(base_dir, split, limits.get(split, 0), rng)
        for image_path in selected:
            image_target = output_dir / "images" / split / image_path.name
            label_target = output_dir / "labels" / split / f"{image_path.stem}.txt"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_target)
            label_path = matching_base_label(base_dir, split, image_path)
            if label_path.exists():
                shutil.copy2(label_path, label_target)
            else:
                label_target.write_text("", encoding="utf-8")
        summary[split] = len(selected)
    return summary


def copy_base_balanced_subset(base_dir, output_dir, total_limits, class_limits, background_limits, seed):
    rng = random.Random(seed)
    summary = {}
    for split in ("train", "val", "test"):
        selected = choose_base_balanced_subset(
            base_dir,
            split,
            total_limits.get(split, 0),
            class_limits.get(split, 0),
            background_limits.get(split, 0),
            rng,
        )
        for image_path in selected:
            image_target = output_dir / "images" / split / image_path.name
            label_target = output_dir / "labels" / split / f"{image_path.stem}.txt"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_target)
            label_path = matching_base_label(base_dir, split, image_path)
            if label_path.exists():
                shutil.copy2(label_path, label_target)
            else:
                label_target.write_text("", encoding="utf-8")
        summary[split] = len(selected)
    return summary


def rewrite_data_yaml(output_dir):
    data_yaml = "\n".join([
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: scratch",
        "  1: stain",
        "",
    ])
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_dir / "classes.txt").write_text("scratch\nstain\n", encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    base_dir = (project_root / args.base).resolve()
    dataset_dir = (project_root / args.dataset).resolve()
    output_dir = (project_root / args.output).resolve()

    if not base_dir.exists():
        raise SystemExit(f"Missing base YOLO dataset: {base_dir}")
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    if not output_dir.exists():
        sample_limits = {
            "train": int(args.sample_base_train),
            "val": int(args.sample_base_val),
            "test": int(args.sample_base_test),
        }
        class_limits = {
            "train": int(args.sample_base_class_train),
            "val": int(args.sample_base_class_val),
            "test": int(args.sample_base_class_test),
        }
        background_limits = {
            "train": int(args.sample_base_background_train),
            "val": int(args.sample_base_background_val),
            "test": int(args.sample_base_background_test),
        }
        if any(limit > 0 for limit in class_limits.values()) or any(limit > 0 for limit in background_limits.values()):
            base_summary = copy_base_balanced_subset(
                base_dir,
                output_dir,
                sample_limits,
                class_limits,
                background_limits,
                int(args.random_seed),
            )
            print("Copied balanced base YOLO dataset:", base_summary)
        elif any(limit > 0 for limit in sample_limits.values()):
            base_summary = copy_base_subset(base_dir, output_dir, sample_limits, int(args.random_seed))
            print("Copied sampled base YOLO dataset:", base_summary)
        else:
            shutil.copytree(base_dir, output_dir)
    rewrite_data_yaml(output_dir)

    summary = {
        "normal_empty": 0,
        "scratch_labeled": 0,
        "scratch_augmented": 0,
        "scratch_unlabeled": 0,
        "stain_labeled": 0,
        "stain_unlabeled": 0,
    }
    for split in ("train", "val", "test"):
        normal_dir = dataset_dir / split / "normal"
        for image_path in image_paths(normal_dir, args.date_prefix):
            copy_image_with_label(image_path, output_dir, split, "normal", "")
            summary["normal_empty"] += 1

        scratch_dir = dataset_dir / split / "scratch"
        for image_path in image_paths(scratch_dir, args.date_prefix):
            with Image.open(image_path) as image:
                image_w, image_h = image.size
            box = bright_cross_scratch_box(image_path)
            bright_cross = box is not None
            if box is None:
                box = thin_scratch_box(image_path)
            if box is None:
                box = bright_ring_scratch_box(image_path)
            if box is None:
                summary["scratch_unlabeled"] += 1
                copy_image_with_label(image_path, output_dir, split, "scratch", "")
                continue
            label_text = yolo_line(0, box, image_w, image_h)
            copy_image_with_label(image_path, output_dir, split, "scratch", label_text)
            if bright_cross:
                summary["scratch_augmented"] += copy_bright_cross_scratch_augments(
                    image_path,
                    output_dir,
                    split,
                    "scratch",
                    label_text,
                )
            summary["scratch_labeled"] += 1

        stain_dir = dataset_dir / split / "stain"
        for image_path in image_paths(stain_dir, args.date_prefix):
            with Image.open(image_path) as image:
                image_w, image_h = image.size
            box = dark_stain_box(image_path)
            if box is None:
                summary["stain_unlabeled"] += 1
                copy_image_with_label(image_path, output_dir, split, "stain", "")
                continue
            copy_image_with_label(image_path, output_dir, split, "stain", yolo_line(1, box, image_w, image_h))
            summary["stain_labeled"] += 1

    print("Prepared new-lens YOLO dataset:", output_dir)
    print(summary)


if __name__ == "__main__":
    main()
