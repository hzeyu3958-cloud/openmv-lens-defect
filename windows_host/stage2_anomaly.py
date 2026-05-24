from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = 192
DEFAULT_THRESHOLD_PERCENTILE = 99.7
DEFAULT_MIN_STD = 0.025
DEFAULT_MIN_AREA_RATIO = 0.00008
DEFAULT_RULE_OVERLAP_RATIO = 0.015
STAIN_CONFIRM_MIN_OVERLAP = 0.08
STAIN_CONFIRM_SCORE_FACTOR = 1.55
STAGE2_STAIN_SCORE_FACTOR = 1.65
STAGE2_STAIN_MIN_AREA = 180
ALLOW_STAGE2_ONLY_DEFECTS = False
STAGE2_ONLY_SCRATCH_SCORE_FACTOR = 1.0
STAGE2_ONLY_SCRATCH_MIN_LENGTH = 24
STAGE2_ONLY_SCRATCH_MIN_ASPECT_RATIO = 5.0
STAGE2_ONLY_STAIN_SCORE_FACTOR = 1.85
KEEP_UNCONFIRMED_RULE_DEFECTS = False

FEATURE_NAMES = ("gray", "local_residual", "gradient", "bright_spot", "dark_spot")


def mask_bounds(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    return cv2.boundingRect(points)


def extract_lens_patch(frame, analysis_mask, image_size):
    bounds = mask_bounds(analysis_mask)
    if bounds is None:
        return None, None, None

    x, y, w, h = bounds
    if w <= 0 or h <= 0:
        return None, None, None

    crop = frame[y:y + h, x:x + w]
    crop_mask = analysis_mask[y:y + h, x:x + w]
    patch = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    patch_mask = cv2.resize(crop_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return patch, patch_mask, bounds


def feature_stack(patch):
    gray_u8 = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
    local_residual = np.abs(gray - background)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(cv2.magnitude(grad_x, grad_y), 0.0, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bright_spot = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32) / 255.0
    dark_spot = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) / 255.0

    return np.dstack((gray, local_residual, gradient, bright_spot, dark_spot)).astype(np.float32)


def anomaly_score(features, mean, std, patch_mask):
    z = (features - mean) / std
    score = np.sqrt(np.mean(z * z, axis=2)).astype(np.float32)
    score[patch_mask == 0] = 0.0
    return score


def train_reference(samples, threshold_percentile=DEFAULT_THRESHOLD_PERCENTILE, min_std=DEFAULT_MIN_STD):
    if not samples:
        raise RuntimeError("No normal lens samples were provided for stage-2 training.")

    features = np.stack([sample[0] for sample in samples], axis=0)
    masks = [sample[1] for sample in samples]
    mean = np.mean(features, axis=0).astype(np.float32)
    std = np.std(features, axis=0).astype(np.float32)
    std = np.maximum(std, float(min_std)).astype(np.float32)

    score_values = []
    for feature, patch_mask in samples:
        score = anomaly_score(feature, mean, std, patch_mask)
        values = score[patch_mask > 0]
        if values.size:
            score_values.append(values.astype(np.float32))

    if not score_values:
        raise RuntimeError("The stage-2 lens masks are empty.")

    all_scores = np.concatenate(score_values)
    threshold = float(np.percentile(all_scores, threshold_percentile))
    threshold = max(threshold, 2.5)

    return {
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "image_size": int(features.shape[1]),
        "threshold_percentile": float(threshold_percentile),
        "feature_names": np.array(FEATURE_NAMES),
        "sample_count": int(len(samples)),
    }


def save_reference(path, reference):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reference)


class Stage2AnomalyModel:
    def __init__(self, path, mean, std, threshold, image_size, sample_count=0):
        self.path = str(path)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), DEFAULT_MIN_STD)
        self.threshold = float(threshold)
        self.image_size = int(image_size)
        self.sample_count = int(sample_count)

    @classmethod
    def load(cls, path):
        data = np.load(str(path), allow_pickle=False)
        return cls(
            path=path,
            mean=data["mean"],
            std=data["std"],
            threshold=float(data["threshold"]),
            image_size=int(data["image_size"]),
            sample_count=int(data["sample_count"]) if "sample_count" in data.files else 0,
        )

    def infer(self, frame, analysis_mask):
        patch, patch_mask, bounds = extract_lens_patch(frame, analysis_mask, self.image_size)
        if patch is None:
            return {
                "enabled": True,
                "available": False,
                "reason": "empty_lens_mask",
                "mask": np.zeros(frame.shape[:2], dtype=np.uint8),
                "score_map": np.zeros(frame.shape[:2], dtype=np.float32),
            }

        features = feature_stack(patch)
        score = anomaly_score(features, self.mean, self.std, patch_mask)
        score = cv2.GaussianBlur(score, (3, 3), 0)

        patch_candidate = np.where(score >= self.threshold, 255, 0).astype(np.uint8)
        patch_candidate = cv2.bitwise_and(patch_candidate, patch_mask)
        # Thin scratches can be only one or two pixels wide in the 192px patch.
        # A 3x3 open erases them, so preserve line candidates and let contour
        # shape/score filters reject small noise later.
        patch_candidate = cv2.morphologyEx(
            patch_candidate,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        patch_candidate = cv2.dilate(patch_candidate, np.ones((2, 2), dtype=np.uint8), iterations=1)

        x, y, w, h = bounds
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_score = np.zeros(frame.shape[:2], dtype=np.float32)
        resized_mask = cv2.resize(patch_candidate, (w, h), interpolation=cv2.INTER_NEAREST)
        resized_score = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
        full_mask[y:y + h, x:x + w] = cv2.bitwise_and(resized_mask, analysis_mask[y:y + h, x:x + w])
        full_score[y:y + h, x:x + w] = resized_score

        contours, _hierarchy = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lens_area = max(1, int(cv2.countNonZero(analysis_mask)))
        defects = []
        for contour in contours:
            defect = contour_to_stage2_defect(contour, lens_area, full_score, self.threshold)
            if defect is not None:
                defects.append(defect)

        defects.sort(key=lambda item: item["score"], reverse=True)
        lens_scores = full_score[analysis_mask > 0]
        max_score = float(np.max(lens_scores)) if lens_scores.size else 0.0
        mean_score = float(np.mean(lens_scores)) if lens_scores.size else 0.0
        return {
            "enabled": True,
            "available": True,
            "model_path": self.path,
            "threshold": self.threshold,
            "sample_count": self.sample_count,
            "candidate_count": len(defects),
            "score_max": max_score,
            "score_mean": mean_score,
            "defects": defects,
            "mask": full_mask,
            "score_map": full_score,
        }


def safe_round(value, digits=2):
    return round(float(value), digits)


def estimate_level(area, length):
    if length >= 90 or area >= 500:
        return "medium"
    return "light"


def contour_to_stage2_defect(contour, lens_area, score_map, threshold):
    area = int(cv2.contourArea(contour))
    min_area = max(12, int(lens_area * DEFAULT_MIN_AREA_RATIO))
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    length = max(w, h)
    short_side = max(1, min(w, h))
    aspect_ratio = float(length) / float(short_side)

    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
    component_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(component_mask, [shifted], -1, 255, -1)
    scores = score_map[y:y + h, x:x + w][component_mask > 0]
    if scores.size == 0:
        return None

    score_mean = float(np.mean(scores))
    score_peak = float(np.max(scores))
    confidence = 0.58 + min(0.36, max(0.0, score_peak - threshold) / max(1.0, threshold) * 0.36)

    if aspect_ratio >= 5.0 and length >= 24 and short_side <= max(8, int((lens_area ** 0.5) * 0.03)):
        defect_type = "scratch"
    else:
        defect_type = "stain"

    return {
        "type": defect_type,
        "confidence": safe_round(min(0.96, confidence)),
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": area,
        "length": int(length),
        "aspect_ratio": safe_round(aspect_ratio),
        "level": estimate_level(area, length),
        "score": safe_round(score_peak),
        "score_mean": safe_round(score_mean),
        "stage2_only": True,
    }


def rect_overlap_ratio(defect, mask):
    x, y, w, h = defect["x"], defect["y"], defect["w"], defect["h"]
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(cv2.countNonZero(roi)) / float(roi.size)


def rect_iou(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def is_stage2_only_defect(defect, stage2_result):
    threshold = float(stage2_result.get("threshold", 0.0))
    score = float(defect.get("score", 0.0) or 0.0)
    defect_type = defect.get("type", "")
    length = int(defect.get("length", 0) or 0)
    area = int(defect.get("area", 0) or 0)
    aspect_ratio = float(defect.get("aspect_ratio", 0.0) or 0.0)

    if defect_type == "scratch":
        return (
            score >= threshold * STAGE2_ONLY_SCRATCH_SCORE_FACTOR
            and length >= STAGE2_ONLY_SCRATCH_MIN_LENGTH
            and aspect_ratio >= STAGE2_ONLY_SCRATCH_MIN_ASPECT_RATIO
        )

    if defect_type == "stain":
        return (
            score >= threshold * STAGE2_ONLY_STAIN_SCORE_FACTOR
            and area >= STAGE2_STAIN_MIN_AREA
        )

    return False


def merge_with_rule_defects(rule_defects, stage2_result, max_defects=20, overlap_ratio=DEFAULT_RULE_OVERLAP_RATIO):
    if not stage2_result.get("available"):
        return list(rule_defects), stage2_result

    stage2_mask = stage2_result["mask"]
    score_map = stage2_result["score_map"]
    confirmed = []
    kept = []
    for defect in rule_defects:
        x, y, w, h = defect["x"], defect["y"], defect["w"], defect["h"]
        score_roi = score_map[y:y + h, x:x + w]
        peak = float(np.max(score_roi)) if score_roi.size else 0.0
        overlap = rect_overlap_ratio(defect, stage2_mask)
        defect_type = defect.get("type", "")
        if defect_type == "stain":
            is_confirmed = overlap >= STAIN_CONFIRM_MIN_OVERLAP and peak >= stage2_result["threshold"] * STAIN_CONFIRM_SCORE_FACTOR
        else:
            is_confirmed = overlap >= overlap_ratio or peak >= stage2_result["threshold"] * 0.95
        if is_confirmed:
            updated = dict(defect)
            updated["stage2_confirmed"] = True
            updated["stage2_score"] = safe_round(peak)
            updated["confidence"] = safe_round(min(0.99, float(updated["confidence"]) + 0.06))
            confirmed.append(updated)
            kept.append(updated)
        elif KEEP_UNCONFIRMED_RULE_DEFECTS:
            updated = dict(defect)
            updated["stage2_confirmed"] = False
            updated["stage2_score"] = safe_round(peak)
            kept.append(updated)

    added = []
    if ALLOW_STAGE2_ONLY_DEFECTS:
        for stage2_defect in stage2_result.get("defects", []):
            if any(rect_iou(stage2_defect, old) > 0.20 for old in kept):
                continue
            if not is_stage2_only_defect(stage2_defect, stage2_result):
                continue
            updated = dict(stage2_defect)
            updated["stage2_added"] = True
            updated["stage2_confirmed"] = True
            updated["stage2_score"] = safe_round(updated.get("score", 0.0))
            added.append(updated)

    merged = kept + added
    merged.sort(key=lambda item: (item.get("stage2_score", item.get("score", 0.0)), item["area"]), reverse=True)
    stage2_result["confirmed_rule_count"] = len(confirmed)
    stage2_result["unconfirmed_rule_count"] = max(0, len(kept) - len(confirmed))
    stage2_result["added_stage2_count"] = len(added)
    return merged[:max_defects], stage2_result
