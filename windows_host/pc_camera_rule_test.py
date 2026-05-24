import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from stage2_anomaly import Stage2AnomalyModel, merge_with_rule_defects


SUPPORTED_DEFECT_TYPES = ("scratch", "stain")
DEFECT_TYPE_NAME = {
    "scratch": "划痕",
    "stain": "污点/油污",
}
LEVEL_SCORE = {
    "normal": 0,
    "light": 1,
    "medium": 2,
    "serious": 3,
}

STD_FACTOR = 1.35
MIN_CONTRAST_DELTA = 50
MIN_AREA = 16
BLOB_MERGE_KERNEL = 5
MAX_DEFECTS = 20
ENABLE_EDGE_CANDIDATES = False
MAX_LENSES = 2
LENS_MIN_AREA_RATIO = 0.04
LENS_MAX_AREA_RATIO = 0.80
LENS_MAX_ASPECT_RATIO = 3.4
LENS_EDGE_IGNORE_RATIO = 0.085
LOCAL_BACKGROUND_SIGMA = 23
ROBUST_DELTA_FACTOR = 4.2
MIN_COMPONENT_MEAN_DELTA = 12
MIN_COMPONENT_PEAK_DELTA = 34

CANNY_LOW = 50
CANNY_HIGH = 90

SCRATCH_MIN_ASPECT_RATIO = 5.0
SCRATCH_MIN_LENGTH = 24
STAIN_MIN_AREA = 220
STAIN_MAX_ASPECT_RATIO = 4.5
MEDIUM_AREA = 220

WINDOW_NAME = "PC Camera OpenMV Rule Test - q quit, s save, r ROI, c clear, m mode, +/- sensitivity"


def safe_round(value, digits=2):
    return round(float(value), digits)


def defect_type_name(value):
    return DEFECT_TYPE_NAME.get(value, value)


def estimate_level(defect_type, area, length):
    if defect_type == "scratch":
        if length >= 70 or area >= MEDIUM_AREA:
            return "medium"
        return "light"
    if defect_type == "stain":
        if area >= MEDIUM_AREA:
            return "medium"
        return "light"
    return "normal"


def preprocess(gray):
    return cv2.GaussianBlur(gray, (3, 3), 0)


def build_roi(frame_shape, roi):
    height, width = frame_shape[:2]
    if roi is None:
        margin_x = int(width * 0.12)
        margin_y = int(height * 0.12)
        return margin_x, margin_y, width - margin_x * 2, height - margin_y * 2

    x, y, w, h = roi
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def anomaly_mask(gray, roi, min_delta):
    x, y, w, h = roi
    roi_img = gray[y:y + h, x:x + w]
    mean = float(np.mean(roi_img))
    stdev = float(np.std(roi_img))
    delta = max(float(min_delta), stdev * STD_FACTOR)
    dark_high = max(0, int(mean - delta))
    bright_low = min(255, int(mean + delta))

    mask = np.zeros_like(gray, dtype=np.uint8)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    if dark_high > 3:
        roi_mask = cv2.bitwise_or(roi_mask, cv2.inRange(roi_img, 0, dark_high))
    if bright_low < 252:
        roi_mask = cv2.bitwise_or(roi_mask, cv2.inRange(roi_img, bright_low, 255))

    edges = cv2.Canny(roi_img, CANNY_LOW, CANNY_HIGH)
    roi_mask = cv2.bitwise_or(roi_mask, edges)

    kernel = np.ones((BLOB_MERGE_KERNEL, BLOB_MERGE_KERNEL), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    mask[y:y + h, x:x + w] = roi_mask
    return mask


def default_edge_ignore(roi, edge_ignore):
    if edge_ignore > 0:
        return int(edge_ignore)
    return max(8, int(min(roi[2], roi[3]) * LENS_EDGE_IGNORE_RATIO))


def contour_touches_roi_edge(contour, roi_w, roi_h):
    x, y, w, h = cv2.boundingRect(contour)
    margin = 3
    return x <= margin or y <= margin or x + w >= roi_w - margin or y + h >= roi_h - margin


def score_lens_candidate(ellipse, contour_area, roi_w, roi_h):
    (cx, cy), (axis_a, axis_b), _angle = ellipse
    long_axis = max(axis_a, axis_b)
    short_axis = max(1.0, min(axis_a, axis_b))
    aspect = long_axis / short_axis
    if aspect > LENS_MAX_ASPECT_RATIO:
        return 0.0

    ellipse_area = np.pi * axis_a * axis_b / 4.0
    roi_area = float(roi_w * roi_h)
    area_ratio = ellipse_area / roi_area
    if area_ratio < LENS_MIN_AREA_RATIO or area_ratio > LENS_MAX_AREA_RATIO:
        return 0.0

    center_dx = abs(cx - roi_w / 2.0) / max(1.0, roi_w / 2.0)
    center_dy = abs(cy - roi_h / 2.0) / max(1.0, roi_h / 2.0)
    center_score = 1.0 - min(1.0, (center_dx + center_dy) / 2.0)
    fill_score = min(1.0, contour_area / max(1.0, ellipse_area))
    aspect_score = 1.0 - min(1.0, abs(aspect - 1.55) / 2.2)
    return area_ratio * 0.45 + center_score * 0.25 + fill_score * 0.18 + aspect_score * 0.12


def find_lens_ellipses(frame, roi):
    x, y, w, h = roi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    edges = cv2.Canny(roi_gray, 35, 95)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        if len(contour) < 5:
            continue
        contour_area = cv2.contourArea(contour)
        if contour_area < w * h * 0.01:
            continue
        ellipse = cv2.fitEllipse(contour)
        score = score_lens_candidate(ellipse, contour_area, w, h)
        if score <= 0:
            continue
        (cx, cy), axes, angle = ellipse
        full_ellipse = ((cx + x, cy + y), axes, angle)
        candidates.append((score, full_ellipse))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for score, ellipse in candidates:
        cx, cy = ellipse[0]
        too_close = False
        for _old_score, old_ellipse in selected:
            ox, oy = old_ellipse[0]
            distance = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            min_axis = min(ellipse[1][0], ellipse[1][1], old_ellipse[1][0], old_ellipse[1][1])
            if distance < min_axis * 0.35:
                too_close = True
                break
        if not too_close:
            selected.append((score, ellipse))
        if len(selected) >= MAX_LENSES:
            break

    return [ellipse for _score, ellipse in selected]


def center_fallback_ellipse(roi):
    x, y, w, h = roi
    center = (x + w / 2.0, y + h / 2.0)
    axes = (w * 0.78, h * 0.66)
    return (center, axes, 0.0)


def create_lens_segmentation(frame, roi, edge_ignore, lens_mode):
    ellipses = [] if lens_mode == "center" else find_lens_ellipses(frame, roi)
    source = "auto"
    found = bool(ellipses)
    if not ellipses:
        ellipses = [center_fallback_ellipse(roi)]
        source = "center_fallback"

    lens_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for ellipse in ellipses:
        cv2.ellipse(lens_mask, ellipse, 255, -1)

    ignore = default_edge_ignore(roi, edge_ignore)
    if ignore > 0:
        distance = cv2.distanceTransform(lens_mask, cv2.DIST_L2, 3)
        analysis_mask = np.where(distance > ignore, 255, 0).astype(np.uint8)
    else:
        analysis_mask = lens_mask.copy()

    return {
        "found": found,
        "source": source,
        "edge_ignore": ignore,
        "ellipses": ellipses,
        "lens_mask": lens_mask,
        "analysis_mask": analysis_mask,
    }


def anomaly_mask_in_lens(gray, roi, analysis_mask, min_delta, use_edges):
    x, y, w, h = roi
    roi_img = gray[y:y + h, x:x + w]
    roi_lens_mask = analysis_mask[y:y + h, x:x + w]
    lens_pixels = roi_img[roi_lens_mask > 0]
    if lens_pixels.size == 0:
        empty = np.zeros_like(gray, dtype=np.uint8)
        return empty, empty, int(min_delta)

    background = cv2.GaussianBlur(
        roi_img,
        (0, 0),
        sigmaX=LOCAL_BACKGROUND_SIGMA,
        sigmaY=LOCAL_BACKGROUND_SIGMA,
    )
    roi_diff = cv2.absdiff(roi_img, background)
    lens_diff = roi_diff[roi_lens_mask > 0]
    median = float(np.median(lens_diff))
    mad = float(np.median(np.abs(lens_diff.astype(np.float32) - median)))
    robust_sigma = 1.4826 * mad
    delta = int(max(float(min_delta), median + robust_sigma * ROBUST_DELTA_FACTOR))

    roi_mask = cv2.inRange(roi_diff, delta, 255)

    if use_edges:
        edges = cv2.Canny(roi_img, CANNY_LOW, CANNY_HIGH)
        roi_mask = cv2.bitwise_or(roi_mask, edges)
    roi_mask = cv2.bitwise_and(roi_mask, roi_lens_mask)

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((BLOB_MERGE_KERNEL, BLOB_MERGE_KERNEL), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, open_kernel)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_kernel)
    roi_mask = cv2.bitwise_and(roi_mask, roi_lens_mask)

    mask = np.zeros_like(gray, dtype=np.uint8)
    diff_map = np.zeros_like(gray, dtype=np.uint8)
    mask[y:y + h, x:x + w] = roi_mask
    diff_map[y:y + h, x:x + w] = roi_diff
    return mask, diff_map, delta


def contour_to_defect(contour, lens_area, diff_map):
    area = int(cv2.contourArea(contour))
    min_area = max(MIN_AREA, int(lens_area * 0.00001))
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
    component_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(component_mask, [shifted], -1, 255, -1)
    component_pixels = diff_map[y:y + h, x:x + w][component_mask > 0]
    if component_pixels.size == 0:
        return None

    mean_delta = float(np.mean(component_pixels))
    peak_delta = float(np.max(component_pixels))
    if mean_delta < MIN_COMPONENT_MEAN_DELTA or peak_delta < MIN_COMPONENT_PEAK_DELTA:
        return None

    length = max(w, h)
    short_side = max(1, min(w, h))
    aspect_ratio = float(length) / float(short_side)
    lens_scale = max(1.0, lens_area ** 0.5)
    stain_min_area = max(STAIN_MIN_AREA, int(lens_area * 0.00010))
    stain_max_area = max(stain_min_area * 2, int(lens_area * 0.00075))
    stain_max_side = max(24, int(lens_scale * 0.13))
    scratch_min_length = max(SCRATCH_MIN_LENGTH, int(lens_scale * 0.035))
    scratch_max_short_side = max(6, int(lens_scale * 0.025))

    defect_type = None
    confidence = 0.0
    if (
        aspect_ratio >= SCRATCH_MIN_ASPECT_RATIO
        and length >= scratch_min_length
        and short_side <= scratch_max_short_side
    ):
        defect_type = "scratch"
        confidence = 0.76 + min(0.16, (aspect_ratio - SCRATCH_MIN_ASPECT_RATIO) / 25.0)
    elif (
        stain_min_area <= area <= stain_max_area
        and aspect_ratio <= STAIN_MAX_ASPECT_RATIO
        and length <= stain_max_side
    ):
        defect_type = "stain"
        confidence = 0.66 + min(0.18, area / float(max(1, stain_max_area)))

    if defect_type is None:
        return None

    level = estimate_level(defect_type, area, length)
    return {
        "type": defect_type,
        "confidence": safe_round(min(0.98, max(0.50, confidence))),
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": area,
        "length": int(length),
        "aspect_ratio": safe_round(aspect_ratio),
        "level": level,
    }


def detect_defects(frame, roi, min_delta, edge_ignore, lens_mode, use_edges):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)
    segmentation = create_lens_segmentation(frame, roi, edge_ignore, lens_mode)
    mask, diff_map, candidate_delta = anomaly_mask_in_lens(
        processed,
        roi,
        segmentation["analysis_mask"],
        min_delta,
        use_edges,
    )
    segmentation["candidate_delta"] = int(candidate_delta)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lens_area = max(1, int(cv2.countNonZero(segmentation["analysis_mask"])))

    defects = []
    for contour in contours:
        defect = contour_to_defect(contour, lens_area, diff_map)
        if defect is not None:
            defects.append(defect)

    defects.sort(key=lambda item: item["area"], reverse=True)
    return defects[:MAX_DEFECTS], segmentation


def load_stage2_model(path):
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists():
        raise RuntimeError("二级异常检测模型不存在：%s" % model_path)
    model = Stage2AnomalyModel.load(model_path)
    print("已加载二级异常检测模型：%s，正常样本数=%d，阈值=%.2f" % (
        model.path,
        model.sample_count,
        model.threshold,
    ))
    return model


def apply_stage2(frame, defects, segmentation, stage2_model):
    if stage2_model is None:
        segmentation["stage2"] = {"enabled": False}
        return defects

    stage2_result = stage2_model.infer(frame, segmentation["analysis_mask"])
    merged, stage2_result = merge_with_rule_defects(defects, stage2_result, max_defects=MAX_DEFECTS)
    segmentation["stage2"] = stage2_result
    return merged


def build_result(defects, frame_shape, roi, segmentation):
    summary = {defect_type: 0 for defect_type in SUPPORTED_DEFECT_TYPES}
    overall_level = "normal"
    for defect in defects:
        summary[defect["type"]] += 1
        if LEVEL_SCORE[defect["level"]] > LEVEL_SCORE[overall_level]:
            overall_level = defect["level"]

    result = {
        "has_defect": len(defects) > 0,
        "defect_count": len(defects),
        "summary": summary,
        "overall_level": overall_level,
        "defects": defects,
        "timestamp": int(time.time() * 1000),
        "model": "pc_camera_rule_detector",
        "frame": {"w": int(frame_shape[1]), "h": int(frame_shape[0])},
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "lens": {
            "found": bool(segmentation["found"]),
            "source": segmentation["source"],
            "count": len(segmentation["ellipses"]),
            "edge_ignore": int(segmentation["edge_ignore"]),
            "candidate_delta": int(segmentation.get("candidate_delta", 0)),
        },
    }
    stage2 = segmentation.get("stage2")
    if stage2:
        result["stage2"] = {
            "enabled": bool(stage2.get("enabled", False)),
            "available": bool(stage2.get("available", False)),
            "model_path": stage2.get("model_path", ""),
            "sample_count": int(stage2.get("sample_count", 0)),
            "threshold": safe_round(stage2.get("threshold", 0.0)),
            "score_max": safe_round(stage2.get("score_max", 0.0)),
            "score_mean": safe_round(stage2.get("score_mean", 0.0)),
            "candidate_count": int(stage2.get("candidate_count", 0)),
            "confirmed_rule_count": int(stage2.get("confirmed_rule_count", 0)),
            "added_stage2_count": int(stage2.get("added_stage2_count", 0)),
        }
    return result


def draw_mask_contours(display, mask, color, thickness):
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, color, thickness)


def draw_overlay(frame, result, segmentation, min_delta):
    display = frame.copy()
    roi = result["roi"]
    cv2.rectangle(
        display,
        (roi["x"], roi["y"]),
        (roi["x"] + roi["w"], roi["y"] + roi["h"]),
        (0, 180, 255),
        2,
    )
    draw_mask_contours(display, segmentation["lens_mask"], (0, 255, 0), 2)
    draw_mask_contours(display, segmentation["analysis_mask"], (255, 220, 0), 1)
    stage2 = segmentation.get("stage2", {})
    if stage2.get("available") and "mask" in stage2:
        draw_mask_contours(display, stage2["mask"], (255, 0, 255), 1)

    for defect in result["defects"]:
        x, y, w, h = defect["x"], defect["y"], defect["w"], defect["h"]
        padding = max(8, int(min(frame.shape[:2]) * 0.02))
        center = (x + w // 2, y + h // 2)
        axes = (max(10, w // 2 + padding), max(10, h // 2 + padding))
        cv2.ellipse(display, center, axes, 0, 0, 360, (0, 0, 255), 3)
        label = "%s %.2f" % (defect["type"], defect["confidence"])
        cv2.putText(display, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    stage2_status = "off"
    if stage2.get("enabled"):
        stage2_status = "on" if stage2.get("available") else "no-mask"
    status = "defects:%d  sensitivity:%d  lens:%s/%d  delta:%d  stage2:%s" % (
        result["defect_count"],
        min_delta,
        result["lens"]["source"],
        result["lens"]["count"],
        result["lens"].get("candidate_delta", 0),
        stage2_status,
    )
    cv2.putText(display, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return display


def print_result(result, last_print_time):
    now = time.time()
    if now - last_print_time < 0.8:
        return last_print_time
    print(json.dumps(result, ensure_ascii=False))
    return now


def save_frame(frame):
    output_dir = Path("outputs") / "pc_camera_rule_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("pc_camera_%Y%m%d_%H%M%S.jpg")
    path = output_dir / filename
    write_image(path, frame)
    print("已保存截图：%s" % path)


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("图片编码失败：%s" % path)
    encoded.tofile(str(path))


def parse_roi(value):
    if not value:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI 格式应为 x,y,w,h")
    return tuple(parts)


def run_image(path, roi, min_delta, edge_ignore, lens_mode, use_edges, stage2_model=None, no_window=False, no_save=False):
    frame = read_image(path)
    if frame is None:
        raise RuntimeError("无法读取图片：%s" % path)
    active_roi = build_roi(frame.shape, roi)
    defects, segmentation = detect_defects(frame, active_roi, min_delta, edge_ignore, lens_mode, use_edges)
    defects = apply_stage2(frame, defects, segmentation, stage2_model)
    result = build_result(defects, frame.shape, active_roi, segmentation)
    display = draw_overlay(frame, result, segmentation, min_delta)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not no_save:
        save_frame(display)
    if no_window:
        return
    cv2.imshow(WINDOW_NAME, display)
    cv2.waitKey(0)


def run_camera(camera_index, roi, min_delta, edge_ignore, lens_mode, use_edges, stage2_model=None):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("无法打开电脑摄像头：%s" % camera_index)

    last_print_time = 0.0
    print("已打开电脑摄像头。q 退出，s 保存截图，r 框选镜片 ROI，c 清除 ROI，m 切换镜片模式，+/- 调整灵敏度。")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("读取摄像头画面失败")

        active_roi = build_roi(frame.shape, roi)
        defects, segmentation = detect_defects(frame, active_roi, min_delta, edge_ignore, lens_mode, use_edges)
        defects = apply_stage2(frame, defects, segmentation, stage2_model)
        result = build_result(defects, frame.shape, active_roi, segmentation)
        display = draw_overlay(frame, result, segmentation, min_delta)
        cv2.imshow(WINDOW_NAME, display)
        last_print_time = print_result(result, last_print_time)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_frame(display)
        if key == ord("r"):
            selected = cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=True)
            sx, sy, sw, sh = [int(value) for value in selected]
            if sw > 0 and sh > 0:
                roi = (sx, sy, sw, sh)
                print("已设置 ROI：x=%d, y=%d, w=%d, h=%d" % roi)
        if key == ord("c"):
            roi = None
            print("已清除 ROI，恢复默认检测区域。")
        if key == ord("m"):
            lens_mode = "center" if lens_mode == "auto" else "auto"
            print("镜片定位模式：%s" % lens_mode)
        if key in (ord("+"), ord("=")):
            min_delta = max(1, min_delta - 2)
            print("灵敏度提高，MIN_CONTRAST_DELTA=%d" % min_delta)
        if key in (ord("-"), ord("_")):
            min_delta = min(80, min_delta + 2)
            print("灵敏度降低，MIN_CONTRAST_DELTA=%d" % min_delta)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="用电脑摄像头模拟 OpenMV 规则检测。")
    parser.add_argument("--camera", type=int, default=0, help="电脑摄像头编号，默认 0。")
    parser.add_argument("--image", help="也可以先用一张本地图片测试算法。")
    parser.add_argument("--roi", type=parse_roi, help="检测区域，格式 x,y,w,h。默认自动取画面中间区域。")
    parser.add_argument("--min-delta", type=int, default=MIN_CONTRAST_DELTA, help="亮度差阈值，越小越敏感。")
    parser.add_argument("--edge-ignore", type=int, default=0, help="镜片边缘忽略像素。默认按镜片大小自动计算。")
    parser.add_argument("--lens-mode", choices=("auto", "center"), default="auto", help="auto 自动找椭圆镜片；center 强制使用中心椭圆。")
    parser.add_argument("--use-edges", action="store_true", help="启用 Canny 边缘候选。默认关闭，以减少背景纹理误检。")
    parser.add_argument("--stage2-model", help="二级异常检测模型 .npz。启用后会用 PaDiM-Lite 思路复核一级规则候选。")
    parser.add_argument("--no-window", action="store_true", help="图片测试时不弹出窗口，只输出 JSON 并保存结果图。")
    parser.add_argument("--no-save", action="store_true", help="图片测试时不保存叠加结果图，加快批量测试。")
    args = parser.parse_args()
    stage2_model = load_stage2_model(args.stage2_model)

    if args.image:
        run_image(
            args.image,
            args.roi,
            args.min_delta,
            args.edge_ignore,
            args.lens_mode,
            args.use_edges,
            stage2_model=stage2_model,
            no_window=args.no_window,
            no_save=args.no_save,
        )
    else:
        run_camera(
            args.camera,
            args.roi,
            args.min_delta,
            args.edge_ignore,
            args.lens_mode,
            args.use_edges,
            stage2_model=stage2_model,
        )


if __name__ == "__main__":
    main()
