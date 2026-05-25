import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_TO_ID = {"scratch": 0, "stain": 1}
SCRATCH_CENTER_FALLBACK_CONFIDENCE = 0.88


def parse_args():
    parser = argparse.ArgumentParser(description="Export seed YOLO labels from the current PC fast-review algorithm.")
    parser.add_argument("--dataset", default="dataset", help="Source dataset with train/val/test class folders.")
    parser.add_argument("--output", default="dataset_yolo_seed", help="YOLO-format output folder.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of writing labels only.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional limit per split/class, 0 means all.")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    return parser.parse_args()


def image_files(folder):
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


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
        return detect_stain_seed_label(app, host, image, base, min_confidence)

    detection = host.LensDefectHostApp.fast_cv_detect_defect(app, image, None)
    fallback = None
    if class_name == "scratch":
        fallback = center_scratch_fallback_label(host, image)
    if detection is None or detection.get("type") != class_name:
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
    return []


def detect_stain_seed_label(app, host, image, base, min_confidence):
    if host.cv2 is None or host.np is None:
        return []

    cv2 = host.cv2
    np = host.np
    mask = host.LensDefectHostApp.build_detection_mask(
        app,
        image,
        {"width": image.width, "height": image.height},
        base,
    )
    if mask is None or cv2.countNonZero(mask) <= 0:
        return []

    gray = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    mask_values = gray[mask > 0]
    if mask_values.size <= 0:
        return []

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
        return []
    try:
        confidence = float(stain.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence:
        return []

    result = dict(base)
    result["defects"] = [stain]
    result = host.LensDefectHostApp.normalize_detection_result(app, result)
    result = host.LensDefectHostApp.filter_edge_defects_for_image(
        app,
        result,
        {"width": image.width, "height": image.height},
        image,
    )
    return result.get("defects") or []


def center_scratch_fallback_label(host, image):
    if host.cv2 is None or host.np is None:
        return None

    cv2 = host.cv2
    np = host.np
    app = host.LensDefectHostApp.__new__(host.LensDefectHostApp)
    gray = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width < 560 or height < 360:
        return None
    x1 = int(width * 0.46)
    x2 = int(width * 0.66)
    y1 = int(height * 0.30)
    y2 = int(height * 0.82)
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
        if center_x < width * 0.50 or center_x > width * 0.69:
            continue
        if box_w * box_h > width * height * 0.20:
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


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "windows_host"))
    import lens_defect_host as host

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
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
                lines = [
                    line
                    for line in (yolo_line(defect, image.width, image.height) for defect in defects)
                    if line is not None
                ]
                stem = "%s_%s_%s" % (split, class_name, path.stem)
                if args.copy_images:
                    target_image = image_out / (stem + path.suffix.lower())
                    shutil.copy2(path, target_image)
                (label_out / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
                summary["images"] += 1
                if lines:
                    summary["labels"] += len(lines)
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
    print("YOLO seed export complete: %(images)d images, %(labels)d labels, %(empty_labels)d empty label files" % summary)
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()
