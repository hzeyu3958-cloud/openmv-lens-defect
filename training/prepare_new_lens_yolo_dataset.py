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
            box = thin_scratch_box(image_path)
            if box is None:
                box = bright_ring_scratch_box(image_path)
            if box is None:
                summary["scratch_unlabeled"] += 1
                copy_image_with_label(image_path, output_dir, split, "scratch", "")
                continue
            copy_image_with_label(image_path, output_dir, split, "scratch", yolo_line(0, box, image_w, image_h))
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
