import csv
import ctypes
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageTk
except ImportError:
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageStat = None
    ImageTk = None

STAGE2_IMPORT_ERROR = None
try:
    import cv2
except Exception as exc:
    cv2 = None
    STAGE2_IMPORT_ERROR = exc

try:
    import numpy as np
except Exception as exc:
    np = None
    if STAGE2_IMPORT_ERROR is None:
        STAGE2_IMPORT_ERROR = exc

Stage2AnomalyModel = None
merge_with_rule_defects = None
STAGE2_READY = False
SINGLE_INSTANCE_MUTEX_HANDLE = None
SINGLE_INSTANCE_MUTEX_NAME = "Local\\LensDefectHostOpenMVN6"


APP_TITLE = "OpenMV N6 眼镜片缺陷识别上位机"
DEFAULT_BAUDRATE = "115200"
READ_TIMEOUT_SECONDS = 0.02
CAPTURE_TIMEOUT_SECONDS = 12
FRAME_TIMEOUT_SECONDS = 2.0
SERIAL_READ_CHUNK_SIZE = 16384
SERIAL_IMAGE_BYTES_PER_SECOND = 60000
SERIAL_NO_DATA_WARNING_SECONDS = 5.0
SERIAL_NO_DATA_REPEAT_SECONDS = 5.0
AUTO_START_RECEIVE = False

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name in ("windows_host", "dist", "release") else APP_DIR
HISTORY_DIR = APP_DIR / "history"
HISTORY_JSONL = HISTORY_DIR / "detection_history.jsonl"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_STAGE2_MODEL = DEFAULT_MODELS_DIR / "lens_stage2_anomaly.npz"
DEFAULT_YOLO_MODEL = DEFAULT_MODELS_DIR / "lens_yolo.onnx"
DEFAULT_YOLO_LABELS = DEFAULT_MODELS_DIR / "lens_yolo_labels.txt"
DEFAULT_YOLO_META = DEFAULT_MODELS_DIR / "lens_yolo_meta.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
METADATA_FILENAME = "metadata.csv"
SPLIT_TARGET_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
MIN_RECOMMENDED_IMAGES_PER_CLASS = 100
HOST_DEFECT_CONFIRM_UPDATES = 1
HOST_NORMAL_CONFIRM_UPDATES = 2
STAGE2_MIN_INTERVAL_SECONDS = 1.2
FAST_REVIEW_ENABLED = True
FAST_REVIEW_PROMOTE_NORMAL = True
FAST_REVIEW_MIN_INTERVAL_SECONDS = 0.12
FAST_REVIEW_DEFECT_CONFIRM_FRAMES = 1
FAST_REVIEW_IOU_THRESHOLD = 0.12
FAST_REVIEW_CLASSIFIER_AREA_RATIO = 0.18
FAST_REVIEW_STRONG_SCRATCH_CONFIDENCE = 0.78
FAST_REVIEW_STRONG_STAIN_CONFIDENCE = 0.82
FAST_REVIEW_STAIN_MIN_AREA_RATIO = 0.018
FAST_REVIEW_STAIN_MAX_AREA_RATIO = 0.22
FAST_REVIEW_STAIN_MIN_DELTA = 24.0
FAST_REVIEW_STAIN_MIN_FILL_RATIO = 0.16
FAST_REVIEW_STAIN_EDGE_MARGIN_RATIO = 0.035
FAST_REVIEW_STAIN_MIN_EDGE_DISTANCE = 8
FAST_REVIEW_STAIN_MIN_MASK_OVERLAP = 0.58
FAST_REVIEW_DARK_CLUSTER_MIN_AREA_RATIO = 0.00055
FAST_REVIEW_DARK_CLUSTER_MIN_BOX_FILL = 0.010
FAST_REVIEW_DARK_CLUSTER_MIN_SHORT_SIDE_RATIO = 0.022
FAST_REVIEW_DARK_CLUSTER_MAX_ASPECT = 6.5
FAST_REVIEW_DARK_CLUSTER_MIN_ANGLE_GROUPS = 2
FAST_REVIEW_DARK_CLUSTER_MIN_LOCAL_DELTA = 7.0
FAST_REVIEW_DARK_CLUSTER_MIN_SIGNED_DELTA = 14.0
FAST_REVIEW_DARK_CLUSTER_WEAK_LOCAL_DELTA = 4.5
FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO = 0.075
FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE = 12
FAST_REVIEW_SCRATCH_MIN_AGGREGATE_ASPECT = 1.45
FAST_REVIEW_SCRATCH_MIN_MASK_OVERLAP = 0.40
FAST_REVIEW_SCRATCH_MAX_BOX_AREA_RATIO = 0.10
FAST_REVIEW_SCRATCH_MIN_LINE_ASPECT = 1.75
FAST_REVIEW_SCRATCH_MIN_BRIGHT_LINE_ASPECT = 1.65
FAST_REVIEW_SCRATCH_MAX_FILL_RATIO = 0.52
FAST_REVIEW_BRIGHT_SCRATCH_MIN_ANGLE_GROUPS = 2
FAST_REVIEW_BRIGHT_SCRATCH_MIN_LOCAL_DELTA = 6.0
FAST_REVIEW_BRIGHT_SCRATCH_MIN_LINE_COUNT = 3
FAST_REVIEW_BRIGHT_SCRATCH_MIN_TOTAL_LENGTH = 58.0
FAST_REVIEW_BRIGHT_SCRATCH_STRONG_MIN_DELTA = 18.0
FAST_REVIEW_BRIGHT_SCRATCH_STRONG_MIN_LENGTH = 82.0
FAST_REVIEW_BRIGHT_SCRATCH_MAX_BOX_AREA_RATIO = 0.075
FAST_REVIEW_BRIGHT_SCRATCH_MAX_FILL_RATIO = 0.36
FAST_REVIEW_EDGE_FILTER_STAIN_MIN_OVERLAP = 0.72
FAST_REVIEW_EDGE_FILTER_SCRATCH_MIN_OVERLAP = 0.52
FAST_REVIEW_EDGE_FILTER_STAIN_CENTER_DISTANCE_RATIO = 0.055
FAST_REVIEW_EDGE_FILTER_STAIN_BOX_DISTANCE_RATIO = 0.040
FAST_REVIEW_EDGE_FILTER_STAIN_ROI_MARGIN_RATIO = 0.12
FAST_REVIEW_EDGE_FILTER_STAIN_EDGE_ASPECT = 1.40
FAST_REVIEW_EDGE_FILTER_SCRATCH_CENTER_DISTANCE_RATIO = 0.040
FAST_REVIEW_EDGE_FILTER_SCRATCH_BOX_DISTANCE_RATIO = 0.038
FAST_REVIEW_EDGE_FILTER_SCRATCH_ROI_MARGIN_RATIO = 0.10
FAST_REVIEW_EDGE_FILTER_SCRATCH_EDGE_ASPECT = 1.75
FAST_REVIEW_INNER_MASK_MARGIN_RATIO = 0.22
FAST_REVIEW_CENTER_ROI_MARGIN_RATIO = 0.23
FAST_REVIEW_BOX_ROI_MARGIN_RATIO = 0.12
FAST_REVIEW_CENTER_DEFECT_X_MARGIN_RATIO = 0.22
FAST_REVIEW_CENTER_DEFECT_Y_MARGIN_RATIO = 0.18
FAST_REVIEW_FRAME_SIDE_REJECT_RATIO = 0.58
FAST_REVIEW_FRAME_TALL_LINE_MIN_ASPECT = 2.2
FAST_REVIEW_FRAME_TALL_LINE_MAX_WIDTH_RATIO = 0.085
FAST_REVIEW_GLARE_LINE_MIN_BRIGHTNESS = 145.0
FAST_REVIEW_GLARE_LINE_MIN_LOCAL_DELTA = 18.0
FAST_REVIEW_LENS_CONTOUR_MIN_WIDTH_RATIO = 0.28
FAST_REVIEW_LENS_CONTOUR_MIN_HEIGHT_RATIO = 0.20
FAST_REVIEW_LENS_CONTOUR_MIN_CENTER_Y_RATIO = 0.30
FAST_REVIEW_AUTO_ROI_MIN_AREA_RATIO = 0.018
FAST_REVIEW_AUTO_ROI_MIN_WIDTH_RATIO = 0.20
FAST_REVIEW_AUTO_ROI_MIN_HEIGHT_RATIO = 0.18
FAST_REVIEW_AUTO_ROI_PADDING_RATIO = 0.08
FAST_REVIEW_AUTO_ROI_EDGE_MARGIN_RATIO = 0.10
FAST_REVIEW_NARROW_SCRATCH_MIN_WIDTH = 3
FAST_REVIEW_NARROW_SCRATCH_MAX_ASPECT = 12.0
FAST_REVIEW_NARROW_SCRATCH_MAX_AREA = 180
FAST_REVIEW_DARK_STAIN_MAX_AREA_RATIO = 0.42
FAST_REVIEW_DARK_STAIN_SOLID_MIN_AREA_RATIO = 0.020
FAST_REVIEW_DARK_STAIN_SOLID_MIN_FILL = 0.24
FAST_REVIEW_DARK_STAIN_MAX_WEAK_AREA_RATIO = 0.72
FAST_REVIEW_DARK_STAIN_MAX_WEAK_BOX_RATIO = 1.05
FAST_REVIEW_DARK_STAIN_WEAK_LOCAL_DELTA = 5.2
FAST_REVIEW_WEAK_SCRATCH_MAX_AREA_RATIO = 0.006
FAST_REVIEW_WEAK_SCRATCH_MAX_BOX_RATIO = 0.025
FAST_REVIEW_ROI_CONTAINMENT_TOLERANCE_RATIO = 0.025
FAST_REVIEW_SCRATCH_CROSSHATCH_STAIN_MAX_RATIO = 0.72
FAST_REVIEW_SCRATCH_CROSSHATCH_MIN_LINE_LENGTH = 74.0
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_LINE_COUNT = 4
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_TOTAL_LENGTH = 82.0
FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_BOX_AREA_RATIO = 0.12
FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_STRONG_BOX_AREA_RATIO = 0.32
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_BRIGHT_DELTA = 13.0
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LINES = 3
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LENGTH = 28.0
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_ASPECT = 4.0
FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_BLOB_LINE_RATIO = 0.42
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_ASPECT = 1.28
FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_FILL_RATIO = 0.30
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_STRONG_ANGLE_GROUPS = 5
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_STRONG_TOTAL_LENGTH = 260.0
FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_STRONG_SLENDER_LINES = 3
FAST_REVIEW_CENTER_CROSS_MIN_X_RATIO = 0.46
FAST_REVIEW_CENTER_CROSS_MAX_X_RATIO = 0.66
FAST_REVIEW_CENTER_CROSS_MIN_Y_RATIO = 0.30
FAST_REVIEW_CENTER_CROSS_MAX_Y_RATIO = 0.82
FAST_REVIEW_CENTER_CROSS_BOX_MIN_X_RATIO = 0.50
FAST_REVIEW_CENTER_CROSS_BOX_MAX_X_RATIO = 0.69
FAST_REVIEW_CENTER_CROSS_MIN_LINE_COUNT = 5
FAST_REVIEW_CENTER_CROSS_MIN_ANGLE_GROUPS = 3
FAST_REVIEW_CENTER_CROSS_MIN_SLENDER_LINES = 4
FAST_REVIEW_CENTER_CROSS_MIN_TOTAL_LENGTH = 145.0
FAST_REVIEW_CENTER_CROSS_MIN_LOCAL_DELTA = 5.0
FAST_REVIEW_CENTER_CROSS_MAX_BOX_AREA_RATIO = 0.20
FAST_REVIEW_CENTER_CROSS_MAX_FILL_RATIO = 0.43
FAST_REVIEW_CENTER_CROSS_MAX_VERTICAL_RATIO = 0.48
FAST_REVIEW_GLARE_MASK_PERCENTILE = 97.5
FAST_REVIEW_GLARE_MASK_MIN_GRAY = 175.0
FAST_REVIEW_STAR_SCRATCH_MIN_LINES = 4
FAST_REVIEW_STAR_SCRATCH_MIN_LENGTH = 72.0
FAST_REVIEW_STAR_SCRATCH_MIN_FILL = 0.12
FAST_REVIEW_STAR_SCRATCH_MAX_FILL = 0.62
FAST_REVIEW_BRIGHT_SCRATCH_MIN_SIGNED_DELTA = 1.2
FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA = -0.8
FAST_REVIEW_KEEP_PC_RESULT_SECONDS = 2.20
YOLO_INPUT_SIZE = 640
YOLO_CONFIDENCE_THRESHOLD = 0.08
YOLO_NMS_THRESHOLD = 0.45
YOLO_MIN_INTERVAL_SECONDS = 0.10
YOLO_MISSING_MODEL_RECHECK_SECONDS = 2.0
YOLO_FALLBACK_INPUT_SIZES = (416, 512, 320, 640)

SUPPORTED_DEFECT_TYPES = (
    "scratch",
    "stain",
)

DEFECT_TYPE_ORDER = ["normal"] + list(SUPPORTED_DEFECT_TYPES)
SUMMARY_TYPES = list(SUPPORTED_DEFECT_TYPES)

DEFECT_TYPE_NAME = {
    "normal": "正常",
    "scratch": "划痕",
    "stain": "污渍",
}

CLASS_DISPLAY_ORDER = [DEFECT_TYPE_NAME[class_name] for class_name in DEFECT_TYPE_ORDER]

LEVEL_NAME = {
    "normal": "正常",
    "error": "错误",
    "light": "轻微",
    "medium": "中等",
    "serious": "严重",
}

LEVEL_SCORE = {
    "normal": 0,
    "light": 1,
    "medium": 2,
    "serious": 3,
}


@dataclass
class UiMessage:
    kind: str
    payload: object


def defect_type_name(value):
    return DEFECT_TYPE_NAME.get(value, value)


def defect_type_key(value):
    if value in DEFECT_TYPE_ORDER:
        return value
    for key, name in DEFECT_TYPE_NAME.items():
        if value == name:
            return key
    return "normal"


def level_name(value):
    return LEVEL_NAME.get(value, value)


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_filename_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def list_image_files(folder):
    if not folder.exists():
        return []
    return [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def count_image_files(folder):
    return len(list_image_files(folder))


def metadata_path(dataset_dir):
    return dataset_dir / METADATA_FILENAME


def acquire_single_instance_lock():
    global SINGLE_INSTANCE_MUTEX_HANDLE
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, SINGLE_INSTANCE_MUTEX_NAME)
    if not handle:
        return True
    already_exists = kernel32.GetLastError() == 183
    if already_exists:
        kernel32.CloseHandle(handle)
        return False
    SINGLE_INSTANCE_MUTEX_HANDLE = handle
    return True


def normalize_model_class_name(value, class_index=None):
    text = str(value or "").strip().lower()
    if text in SUPPORTED_DEFECT_TYPES:
        return text
    if "scratch" in text or "scrape" in text or "mark" in text or "划痕" in text or "刮" in text:
        return "scratch"
    if "stain" in text or "dirt" in text or "dirty" in text or "spot" in text or "污" in text:
        return "stain"
    if text == "normal" or "正常" in text:
        return ""
    if class_index == 0:
        return "scratch"
    if class_index == 1:
        return "stain"
    return ""


def find_training_script():
    candidates = [
        PROJECT_ROOT / "training" / "train_lens_classifier.py",
        APP_DIR / "training" / "train_lens_classifier.py",
        APP_DIR.parent / "training" / "train_lens_classifier.py",
        Path.cwd() / "training" / "train_lens_classifier.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_yolo_seed_export_script():
    candidates = [
        PROJECT_ROOT / "training" / "export_yolo_seed_labels.py",
        APP_DIR / "training" / "export_yolo_seed_labels.py",
        APP_DIR.parent / "training" / "export_yolo_seed_labels.py",
        Path.cwd() / "training" / "export_yolo_seed_labels.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_yolo_training_script():
    candidates = [
        PROJECT_ROOT / "training" / "train_yolo_from_seed.py",
        APP_DIR / "training" / "train_yolo_from_seed.py",
        APP_DIR.parent / "training" / "train_yolo_from_seed.py",
        Path.cwd() / "training" / "train_yolo_from_seed.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_yolo_preview_script():
    candidates = [
        PROJECT_ROOT / "training" / "preview_yolo_seed_labels.py",
        APP_DIR / "training" / "preview_yolo_seed_labels.py",
        APP_DIR.parent / "training" / "preview_yolo_seed_labels.py",
        Path.cwd() / "training" / "preview_yolo_seed_labels.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_training_python():
    candidates = [
        PROJECT_ROOT / ".venv_training" / "Scripts" / "python.exe",
        APP_DIR / ".venv_training" / "Scripts" / "python.exe",
        APP_DIR.parent / ".venv_training" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "python"


def ensure_stage2_runtime_loaded():
    global STAGE2_READY, STAGE2_IMPORT_ERROR, Stage2AnomalyModel, merge_with_rule_defects
    if STAGE2_READY:
        return True
    if cv2 is None or np is None:
        if STAGE2_IMPORT_ERROR is None:
            STAGE2_IMPORT_ERROR = "缺少 OpenCV 或 NumPy"
        return False
    try:
        from stage2_anomaly import Stage2AnomalyModel as LoadedStage2AnomalyModel
        from stage2_anomaly import merge_with_rule_defects as loaded_merge_with_rule_defects
    except Exception as exc:
        STAGE2_IMPORT_ERROR = exc
        return False

    Stage2AnomalyModel = LoadedStage2AnomalyModel
    merge_with_rule_defects = loaded_merge_with_rule_defects
    STAGE2_READY = True
    STAGE2_IMPORT_ERROR = None
    return True


class YoloOnnxDetector:
    def __init__(self, model_path, labels_path=None, input_size=YOLO_INPUT_SIZE):
        if cv2 is None or np is None:
            raise RuntimeError("缺少 OpenCV 或 NumPy，不能加载 YOLO ONNX 模型。")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))
        self.labels = self._load_labels(labels_path)
        self.input_size = self._resolve_input_size(input_size)
        self._working_input_size = None
        self.net = self._read_onnx_model(self.model_path)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

    def _read_onnx_model(self, model_path):
        try:
            return cv2.dnn.readNetFromONNX(str(model_path))
        except Exception as first_error:
            try:
                model_bytes = np.frombuffer(Path(model_path).read_bytes(), dtype=np.uint8)
                return cv2.dnn.readNetFromONNX(model_bytes)
            except Exception:
                raise first_error

    def _load_labels(self, labels_path):
        candidates = []
        if labels_path:
            candidates.append(Path(labels_path))
        candidates.append(self.model_path.with_suffix(".txt"))
        candidates.append(DEFAULT_YOLO_LABELS)
        for candidate in candidates:
            if candidate and candidate.exists():
                labels = [
                    line.strip()
                    for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if line.strip()
                ]
                if labels:
                    return labels
        return list(SUPPORTED_DEFECT_TYPES)

    def _resolve_input_size(self, input_size):
        candidates = [
            self.model_path.with_name(self.model_path.stem + "_meta.json"),
            DEFAULT_YOLO_META,
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in ("input_size", "image_size", "imgsz"):
                try:
                    value = int(metadata.get(key, 0) or 0)
                except (TypeError, ValueError):
                    value = 0
                if value >= 128:
                    return value
        return int(input_size)

    def detect(self, image):
        rgb = np.array(image.convert("RGB"))
        frame_h, frame_w = rgb.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return []
        output_size, outputs = self._forward_with_compatible_size(rgb)
        return self._parse_outputs(outputs, frame_w, frame_h, output_size)

    def _forward_with_compatible_size(self, rgb):
        sizes = []
        if self._working_input_size:
            sizes.append(int(self._working_input_size))
        sizes.append(int(self.input_size))
        for size in YOLO_FALLBACK_INPUT_SIZES:
            sizes.append(int(size))
        last_error = None
        for size in dict.fromkeys(sizes):
            try:
                outputs = self._forward_at_size(rgb, size)
            except Exception as exc:
                last_error = exc
                continue
            self._working_input_size = size
            return size, outputs
        if last_error is not None:
            raise last_error
        return int(self.input_size), self._forward_at_size(rgb, int(self.input_size))

    def _forward_at_size(self, rgb, input_size):
        blob = cv2.dnn.blobFromImage(
            rgb,
            1.0 / 255.0,
            (int(input_size), int(input_size)),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        return self.net.forward()

    def _parse_outputs(self, outputs, frame_w, frame_h, input_size=None):
        input_size = int(input_size or self.input_size)
        data = np.asarray(outputs)
        data = np.squeeze(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            data = data.reshape(-1, data.shape[-1])
        if data.shape[0] < data.shape[1] and data.shape[0] in (5, 6, 7, 84, 85):
            data = data.T

        boxes = []
        scores = []
        class_ids = []
        for row in data:
            if row.size < 5:
                continue
            values = row.astype(float)
            class_index = 0
            confidence = 0.0
            label_count = len(self.labels)
            # YOLOv8 exports usually return [x, y, w, h, class...].
            # YOLOv5-style exports return [x, y, w, h, objectness, class...].
            if label_count > 0 and values.size == 4 + label_count:
                class_scores = values[4:]
                class_index = int(np.argmax(class_scores))
                confidence = float(class_scores[class_index])
            elif values.size >= 6:
                objectness = float(values[4])
                class_scores = values[5:]
                if class_scores.size > 0:
                    class_index = int(np.argmax(class_scores))
                    confidence = objectness * float(class_scores[class_index])
                else:
                    confidence = objectness
            else:
                confidence = float(values[4])

            if confidence < YOLO_CONFIDENCE_THRESHOLD:
                continue

            label_text = self.labels[class_index] if class_index < len(self.labels) else str(class_index)
            defect_type = normalize_model_class_name(label_text, class_index)
            if defect_type not in SUPPORTED_DEFECT_TYPES:
                continue

            cx, cy, bw, bh = values[:4]
            if max(cx, cy, bw, bh) <= 2.0:
                cx *= frame_w
                bw *= frame_w
                cy *= frame_h
                bh *= frame_h
            else:
                scale_x = float(frame_w) / float(input_size)
                scale_y = float(frame_h) / float(input_size)
                cx *= scale_x
                bw *= scale_x
                cy *= scale_y
                bh *= scale_y

            left = max(0, int(cx - bw / 2.0))
            top = max(0, int(cy - bh / 2.0))
            width = min(frame_w - left, max(1, int(bw)))
            height = min(frame_h - top, max(1, int(bh)))
            boxes.append([left, top, width, height])
            scores.append(float(confidence))
            class_ids.append((class_index, defect_type))

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_THRESHOLD)
        if len(keep) == 0:
            return []
        indexes = np.array(keep).reshape(-1)
        detections = []
        for index in indexes:
            left, top, width, height = boxes[int(index)]
            _class_index, defect_type = class_ids[int(index)]
            area = int(width * height)
            length = int(max(width, height))
            detections.append({
                "type": defect_type,
                "confidence": round(float(scores[int(index)]), 2),
                "x": int(left),
                "y": int(top),
                "w": int(width),
                "h": int(height),
                "area": area,
                "length": length,
                "aspect_ratio": round(float(length) / float(max(1, min(width, height))), 2),
                "level": "medium" if area >= 500 or length >= 70 else "light",
                "source": "yolo_onnx",
            })
        return detections


class SerialLineReader:
    def __init__(self, ui_queue):
        self.ui_queue = ui_queue
        self.serial_port = None
        self.thread = None
        self.running = False

    def start(self, port, baudrate):
        if serial is None:
            raise RuntimeError("缺少 pyserial，请先运行 run_windows_host.bat 安装依赖。")
        if self.running:
            return

        self.serial_port = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            timeout=READ_TIMEOUT_SECONDS,
        )
        try:
            self.serial_port.reset_input_buffer()
        except Exception:
            pass
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None

    def _read_loop(self):
        self.ui_queue.put(UiMessage("status", "串口已打开，正在接收检测 JSON 和 OpenMV 画面"))
        last_data_time = time.time()
        last_no_data_warning_time = 0.0

        while self.running:
            try:
                if self.serial_port is None:
                    break
                raw = self.serial_port.readline()
                if not raw:
                    now = time.time()
                    if (
                        now - last_data_time >= SERIAL_NO_DATA_WARNING_SECONDS
                        and now - last_no_data_warning_time >= SERIAL_NO_DATA_REPEAT_SECONDS
                    ):
                        self.ui_queue.put(UiMessage(
                            "no_data",
                            "串口已打开，但 N6 没有输出 JSON/画面。请按 N6 RESET 或重新插拔 USB。",
                        ))
                        last_no_data_warning_time = now
                    continue
                last_data_time = time.time()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("IMG_BEGIN "):
                    self._read_image_frame(line)
                elif line.startswith("{"):
                    self.ui_queue.put(UiMessage("line", line))
                elif line.startswith("ERR ") or line.startswith("BOOT "):
                    self.ui_queue.put(UiMessage("status", line))
            except Exception as exc:
                self.ui_queue.put(UiMessage("error", "串口读取失败：%s" % exc))
                break

        self.running = False
        self.ui_queue.put(UiMessage("status", "检测接收已停止"))

    def _read_image_frame(self, header_line):
        parts = header_line.split()
        if len(parts) < 2:
            return

        size = int(parts[1])
        width = int(parts[2]) if len(parts) >= 4 else 0
        height = int(parts[3]) if len(parts) >= 4 else 0
        transfer_timeout = max(FRAME_TIMEOUT_SECONDS, float(size) / float(SERIAL_IMAGE_BYTES_PER_SECOND) + 1.0)
        deadline = time.time() + transfer_timeout
        data = bytearray()

        while self.running and len(data) < size and time.time() < deadline:
            if self.serial_port is None:
                break
            remaining = size - len(data)
            chunk = self.serial_port.read(min(SERIAL_READ_CHUNK_SIZE, remaining))
            if chunk:
                data.extend(chunk)

        if len(data) != size:
            self.ui_queue.put(UiMessage("status", "OpenMV 画面数据不完整：%d / %d 字节" % (len(data), size)))
            return

        if self.serial_port is not None:
            self.serial_port.readline()

        self.ui_queue.put(UiMessage("live_image", {
            "image_bytes": bytes(data),
            "width": width,
            "height": height,
            "byte_count": len(data),
            "receive_time": now_text(),
        }))


class UsbImageCaptureClient:
    @staticmethod
    def capture(port, baudrate):
        if serial is None:
            raise RuntimeError("缺少 pyserial，请先运行 run_windows_host.bat 安装依赖。")

        with serial.Serial(port=port, baudrate=int(baudrate), timeout=0.25) as ser:
            time.sleep(0.25)
            ser.reset_input_buffer()
            ser.write(b"CAPTURE\n")
            ser.flush()

            size = None
            width = 0
            height = 0
            deadline = time.time() + CAPTURE_TIMEOUT_SECONDS

            while time.time() < deadline:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("ERR "):
                    raise RuntimeError(line[4:])
                if line.startswith("IMG_BEGIN "):
                    parts = line.split()
                    if len(parts) < 2:
                        raise RuntimeError("收到的图片头格式错误：%s" % line)
                    size = int(parts[1])
                    if len(parts) >= 4:
                        width = int(parts[2])
                        height = int(parts[3])
                    break

            if size is None:
                raise TimeoutError("等待 OpenMV 图片数据超时，请确认 N6 正在运行 n6_usb_image_capture.py")

            data = bytearray()
            while len(data) < size and time.time() < deadline:
                chunk = ser.read(size - len(data))
                if chunk:
                    data.extend(chunk)

            if len(data) != size:
                raise TimeoutError("图片数据不完整：收到 %d / %d 字节" % (len(data), size))

            end_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if end_line and end_line != "IMG_END":
                # 旧固件或串口缓冲偶尔会吞掉结束行，图片本身完整时不强制失败。
                pass

            return bytes(data), width, height


class LensDefectHostApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x720")
        self.root.minsize(940, 620)

        self.ui_queue = queue.Queue()
        self.reader = SerialLineReader(self.ui_queue)
        self.history_records = []
        self.auto_capture_running = False
        self.latest_detection_result = None
        self.latest_live_image_payload = None
        self.stage2_model = None
        self.stage2_model_path = None
        self.stage2_model_mtime = None
        self.yolo_detector = None
        self.yolo_model_path = None
        self.yolo_missing_checked_at = 0.0
        self.yolo_missing_signature = None
        self.last_yolo_infer_time = 0.0
        self.last_yolo_payload_key = None
        self.last_stage2_infer_time = 0.0
        self.last_fast_review_time = 0.0
        self.last_fast_review_key = None
        self.fast_review_candidate = None
        self.fast_review_candidate_count = 0
        self.last_pc_defect_result = None
        self.last_pc_defect_time = 0.0
        self.stable_detection_result = None
        self.pending_detection_class = None
        self.pending_detection_count = 0

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value=DEFAULT_BAUDRATE)
        self.status_var = tk.StringVar(value="状态：未连接")
        self.stage2_enabled_var = tk.BooleanVar(value=False)
        self.stage2_model_var = tk.StringVar(value=str(DEFAULT_STAGE2_MODEL))
        self.stage2_status_var = tk.StringVar(value="高速复核：已启用；二级模型未勾选")
        self.yolo_enabled_var = tk.BooleanVar(value=True)
        self.yolo_model_var = tk.StringVar(value=str(DEFAULT_YOLO_MODEL))
        self.yolo_status_var = tk.StringVar(value="YOLO：未加载；无模型时用电脑快速复核兜底")
        self._last_effective_roi = None

        self.has_defect_var = tk.StringVar(value="是否检测到缺陷：暂无数据")
        self.class_result_var = tk.StringVar(value="识别结果：暂无数据")
        self.confidence_result_var = tk.StringVar(value="置信度：--")
        self.count_var = tk.StringVar(value="缺陷总数：0")
        self.level_var = tk.StringVar(value="整体严重程度：暂无数据")
        self.lens_track_var = tk.StringVar(value="镜片跟踪：暂无数据")
        self.live_image_var = tk.StringVar(value="OpenMV 画面：请选择 OpenMV 的 COM 口，点击“开始接收识别结果和画面”")

        self.dataset_dir_var = tk.StringVar(value=str(DEFAULT_DATASET_DIR))
        self.dataset_class_var = tk.StringVar(value=defect_type_name("normal"))
        self.dataset_split_var = tk.StringVar(value="train")
        self.auto_split_var = tk.BooleanVar(value=True)
        self.capture_interval_var = tk.StringVar(value="1.0")
        self.capture_count_var = tk.StringVar(value="当前类别图片数：0")
        self.dataset_health_var = tk.StringVar(value="数据集体检：暂无统计")
        self.capture_preview_var = tk.StringVar(value="采集预览：暂无图片")
        self.capture_quality_var = tk.StringVar(value="质量提示：暂无图片")

        self.model_file_var = tk.StringVar(value=str(DEFAULT_MODELS_DIR / "lens_defect_classifier_int8.tflite"))
        self.labels_file_var = tk.StringVar(value=str(DEFAULT_MODELS_DIR / "lens_defect_labels.txt"))
        self.openmv_folder_var = tk.StringVar(value="")
        self.live_image_photo = None
        self.capture_preview_photo = None
        self.summary_tree = None
        self.defect_tree = None
        self.raw_text = None
        self.history_tree = None

        self._create_widgets()
        self.refresh_ports()
        self.load_history()
        self.refresh_dataset_counts()
        self.root.after(80, self._poll_ui_queue)
        self.root.after(350, self.preload_yolo_model)
        self.root.after(700, self.preload_stage2_model)
        if AUTO_START_RECEIVE:
            self.root.after(500, self.start_receive)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_widgets(self):
        self._configure_style()

        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root_frame, text=APP_TITLE, style="Title.TLabel")
        title.pack(anchor="w")

        serial_frame = ttk.LabelFrame(root_frame, text="USB/串口连接", padding=8)
        serial_frame.pack(fill=tk.X, pady=(6, 6))

        ttk.Label(serial_frame, text="COM 口").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=32, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(8, 12))

        ttk.Label(serial_frame, text="波特率").pack(side=tk.LEFT)
        self.baud_combo = ttk.Combobox(
            serial_frame,
            textvariable=self.baud_var,
            width=10,
            values=("9600", "19200", "38400", "57600", "115200", "921600"),
        )
        self.baud_combo.pack(side=tk.LEFT, padx=(8, 12))

        ttk.Button(serial_frame, text="刷新串口", command=self.refresh_ports).pack(side=tk.LEFT, padx=4)
        ttk.Label(serial_frame, text="提示：OpenMV N6 USB 连接电脑后会出现一个 COM 口").pack(side=tk.LEFT, padx=12)

        ttk.Label(root_frame, textvariable=self.status_var, style="Status.TLabel").pack(fill=tk.X, pady=(0, 6))

        self.notebook = ttk.Notebook(root_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.detect_tab = ttk.Frame(self.notebook, padding=6)
        self.dataset_tab = ttk.Frame(self.notebook, padding=8)
        self.train_tab = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.detect_tab, text="1 检测显示")
        self.notebook.add(self.dataset_tab, text="2 采集训练数据")
        self.notebook.add(self.train_tab, text="3 训练和部署")

        self._create_detect_tab()
        self._create_dataset_tab()
        self._create_train_tab()

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10), foreground="#1E6B5C")
        style.configure("Result.TLabel", font=("Microsoft YaHei UI", 11))
        style.configure("Treeview", rowheight=24, font=("Microsoft YaHei UI", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))

    def _create_detect_tab(self):
        button_frame = ttk.Frame(self.detect_tab)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="开始接收识别结果和画面", command=self.start_receive).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="停止接收", command=self.stop_receive).pack(side=tk.LEFT, padx=4)

        result_frame = ttk.Frame(self.detect_tab)
        result_frame.pack(fill=tk.X, pady=(6, 0))
        self.class_result_label = tk.Label(
            result_frame,
            textvariable=self.class_result_var,
            font=("Microsoft YaHei UI", 18, "bold"),
            bg="#F8F8F8",
            fg="#333333",
            relief="groove",
            anchor="w",
            padx=12,
            pady=7,
        )
        self.class_result_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            result_frame,
            textvariable=self.confidence_result_var,
            style="Result.TLabel",
            width=18,
            anchor="center",
        ).pack(side=tk.LEFT, padx=(10, 0), fill=tk.Y)

        stage2_frame = ttk.Frame(self.detect_tab)
        stage2_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(stage2_frame, text="启用电脑二级复核", variable=self.stage2_enabled_var).pack(side=tk.LEFT, padx=4)
        ttk.Entry(stage2_frame, textvariable=self.stage2_model_var, width=46).pack(side=tk.LEFT, padx=4)
        ttk.Button(stage2_frame, text="选择模型", command=self.choose_stage2_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(stage2_frame, text="加载模型", command=self.load_stage2_model_from_ui).pack(side=tk.LEFT, padx=4)
        ttk.Label(stage2_frame, textvariable=self.stage2_status_var).pack(side=tk.LEFT, padx=8)

        yolo_frame = ttk.Frame(self.detect_tab)
        yolo_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(yolo_frame, text="启用 YOLO 标框", variable=self.yolo_enabled_var).pack(side=tk.LEFT, padx=4)
        ttk.Entry(yolo_frame, textvariable=self.yolo_model_var, width=46).pack(side=tk.LEFT, padx=4)
        ttk.Button(yolo_frame, text="选择 YOLO", command=self.choose_yolo_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(yolo_frame, text="加载 YOLO", command=self.load_yolo_model_from_ui).pack(side=tk.LEFT, padx=4)
        ttk.Label(yolo_frame, textvariable=self.yolo_status_var).pack(side=tk.LEFT, padx=8)

        live_frame = ttk.LabelFrame(self.detect_tab, text="OpenMV 实时画面", padding=6)
        live_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.live_image_label = tk.Label(
            live_frame,
            textvariable=self.live_image_var,
            compound="top",
            bg="#20242A",
            relief="groove",
            width=84,
            height=26,
            anchor="center",
            justify="center",
        )
        self.live_image_label.pack(fill=tk.BOTH, expand=True)
        self.live_image_label.bind("<Configure>", self.on_live_image_resize)
        return
        ttk.Button(button_frame, text="开始接收 JSON 和画面", command=self.start_receive).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="停止接收", command=self.stop_receive).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="清空记录", command=self.clear_history).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="导出 CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="模拟一条", command=self.mock_one_result).pack(side=tk.LEFT, padx=4)

        stage2_frame = ttk.Frame(self.detect_tab)
        stage2_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(stage2_frame, text="启用电脑二级复核", variable=self.stage2_enabled_var).pack(side=tk.LEFT, padx=4)
        ttk.Entry(stage2_frame, textvariable=self.stage2_model_var, width=54).pack(side=tk.LEFT, padx=4)
        ttk.Button(stage2_frame, text="选择模型", command=self.choose_stage2_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(stage2_frame, text="加载模型", command=self.load_stage2_model_from_ui).pack(side=tk.LEFT, padx=4)
        ttk.Label(stage2_frame, textvariable=self.stage2_status_var).pack(side=tk.LEFT, padx=8)

        main_pane = ttk.PanedWindow(self.detect_tab, orient="horizontal")
        main_pane.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=2)

        result_frame = ttk.LabelFrame(left_frame, text="当前检测结果", padding=10)
        result_frame.pack(fill=tk.X)
        ttk.Label(result_frame, textvariable=self.has_defect_var, style="Result.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.count_var, style="Result.TLabel").pack(anchor="w", pady=3)
        ttk.Label(result_frame, textvariable=self.level_var, style="Result.TLabel").pack(anchor="w")
        ttk.Label(result_frame, textvariable=self.lens_track_var, style="Result.TLabel").pack(anchor="w", pady=(3, 0))

        summary_frame = ttk.LabelFrame(left_frame, text="各类缺陷数量", padding=8)
        summary_frame.pack(fill=tk.X, pady=(8, 8))
        self.summary_tree = ttk.Treeview(summary_frame, columns=("type", "count"), show="headings", height=3)
        self.summary_tree.heading("type", text="缺陷类型")
        self.summary_tree.heading("count", text="数量")
        self.summary_tree.column("type", width=160, anchor="center")
        self.summary_tree.column("count", width=80, anchor="center")
        self.summary_tree.pack(fill=tk.X)

        defect_frame = ttk.LabelFrame(left_frame, text="缺陷详情列表", padding=8)
        defect_frame.pack(fill=tk.BOTH, expand=True)
        self.defect_tree = ttk.Treeview(
            defect_frame,
            columns=("type", "confidence", "level", "x", "y", "w", "h", "area", "length", "ratio"),
            show="headings",
            height=12,
        )
        headings = {
            "type": "类型",
            "confidence": "置信度",
            "level": "等级",
            "x": "X",
            "y": "Y",
            "w": "W",
            "h": "H",
            "area": "面积",
            "length": "长度",
            "ratio": "长宽比",
        }
        widths = {
            "type": 95,
            "confidence": 70,
            "level": 60,
            "x": 45,
            "y": 45,
            "w": 45,
            "h": 45,
            "area": 70,
            "length": 70,
            "ratio": 70,
        }
        for column, text in headings.items():
            self.defect_tree.heading(column, text=text)
            self.defect_tree.column(column, width=widths[column], anchor="center")
        self.defect_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        defect_scroll = ttk.Scrollbar(defect_frame, orient="vertical", command=self.defect_tree.yview)
        defect_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.defect_tree.configure(yscrollcommand=defect_scroll.set)

        live_frame = ttk.LabelFrame(right_frame, text="OpenMV 实时画面", padding=8)
        live_frame.pack(fill=tk.X)
        self.live_image_label = tk.Label(
            live_frame,
            textvariable=self.live_image_var,
            compound="top",
            bg="#F8F8F8",
            relief="groove",
            width=80,
            height=24,
            anchor="center",
            justify="center",
        )
        self.live_image_label.pack(fill=tk.X)

        raw_frame = ttk.LabelFrame(right_frame, text="最近一次 JSON 原始数据", padding=8)
        raw_frame.pack(fill=tk.BOTH, expand=True)
        self.raw_text = self._create_text(raw_frame, height=8)

        history_frame = ttk.LabelFrame(right_frame, text="历史检测记录", padding=8)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=("time", "status", "count", "level", "timestamp"),
            show="headings",
            height=10,
        )
        for column, text, width in (
            ("time", "接收时间", 150),
            ("status", "结果", 80),
            ("count", "数量", 50),
            ("level", "等级", 60),
            ("timestamp", "OpenMV 时间戳", 110),
        ):
            self.history_tree.heading(column, text=text)
            self.history_tree.column(column, width=width, anchor="center")
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=history_scroll.set)

        self._reset_summary()

    def _create_dataset_tab(self):
        config_frame = ttk.LabelFrame(self.dataset_tab, text="数据集保存设置", padding=10)
        config_frame.pack(fill=tk.X)

        ttk.Label(config_frame, text="数据集目录").grid(row=0, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.dataset_dir_var, width=72).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(config_frame, text="选择目录", command=self.choose_dataset_dir).grid(row=0, column=2, padx=4)
        ttk.Button(config_frame, text="初始化分类文件夹", command=self.init_dataset_folders).grid(row=0, column=3, padx=4)

        ttk.Label(config_frame, text="类别").grid(row=1, column=0, sticky="w", pady=(8, 0))
        class_combo = ttk.Combobox(
            config_frame,
            textvariable=self.dataset_class_var,
            values=CLASS_DISPLAY_ORDER,
            state="readonly",
            width=20,
        )
        class_combo.grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
        class_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_dataset_counts())

        ttk.Label(config_frame, text="集合").grid(row=1, column=1, sticky="w", padx=(210, 0), pady=(8, 0))
        ttk.Combobox(
            config_frame,
            textvariable=self.dataset_split_var,
            values=("train", "val", "test"),
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(250, 0), pady=(8, 0))

        ttk.Label(config_frame, text="自动采集间隔秒").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(config_frame, textvariable=self.capture_interval_var, width=8).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            config_frame,
            text="按 70/20/10 自动分配 train/val/test",
            variable=self.auto_split_var,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))
        config_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(self.dataset_tab)
        action_frame.pack(fill=tk.X, pady=8)
        ttk.Button(action_frame, text="测试采图连接", command=self.test_capture_connection).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="拍一张保存到电脑", command=self.capture_one_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="开始自动采集", command=self.start_auto_capture).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="停止自动采集", command=self.stop_auto_capture).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="打开数据集目录", command=self.open_dataset_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="刷新统计", command=self.refresh_dataset_counts).pack(side=tk.LEFT, padx=4)
        ttk.Label(action_frame, textvariable=self.capture_count_var, style="Status.TLabel").pack(side=tk.LEFT, padx=12)

        pane = ttk.PanedWindow(self.dataset_tab, orient="horizontal")
        pane.pack(fill=tk.BOTH, expand=True)

        count_frame = ttk.LabelFrame(pane, text="数据集数量统计", padding=8)
        right_panel = ttk.Frame(pane)
        pane.add(count_frame, weight=1)
        pane.add(right_panel, weight=2)

        self.dataset_tree = ttk.Treeview(count_frame, columns=("class", "train", "val", "test", "total"), show="headings")
        for column, text, width in (
            ("class", "类别", 120),
            ("train", "train", 70),
            ("val", "val", 70),
            ("test", "test", 70),
            ("total", "总数", 70),
        ):
            self.dataset_tree.heading(column, text=text)
            self.dataset_tree.column(column, width=width, anchor="center")
        self.dataset_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Label(count_frame, textvariable=self.dataset_health_var, wraplength=360, style="Status.TLabel").pack(fill=tk.X, pady=(8, 0))

        preview_frame = ttk.LabelFrame(right_panel, text="最近采集预览和质量提示", padding=8)
        preview_frame.pack(fill=tk.X)
        self.capture_preview_label = tk.Label(
            preview_frame,
            textvariable=self.capture_preview_var,
            compound="top",
            bg="#F8F8F8",
            relief="groove",
            width=52,
            height=13,
            anchor="center",
            justify="center",
        )
        self.capture_preview_label.pack(fill=tk.X)
        ttk.Label(preview_frame, textvariable=self.capture_quality_var, wraplength=620, style="Status.TLabel").pack(fill=tk.X, pady=(6, 0))

        log_frame = ttk.LabelFrame(right_panel, text="采集日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.capture_log_text = self._create_text(log_frame, height=14)
        self._set_text(self.capture_log_text, "操作步骤：\n1. 把 openmv/n6_usb_image_capture.py 保存到 OpenMV N6 并运行。\n2. 关闭 OpenMV IDE 串口占用，或确保上位机能打开 N6 的 COM 口。\n3. 选择类别，建议保持“自动分配集合”开启后开始采集。")

    def _create_train_tab(self):
        workflow = (
            "推荐流程：OpenMV N6 拍图 -> USB 连接电脑 -> 电脑保存训练图片 -> 电脑训练模型 "
            "-> 选择 OpenMV 盘符/文件夹 -> 拷贝 .tflite 和 labels.txt -> N6 运行模型脚本。"
        )
        ttk.Label(self.train_tab, text=workflow, wraplength=1000, style="Result.TLabel").pack(anchor="w", pady=(0, 8))

        train_frame = ttk.LabelFrame(self.train_tab, text="电脑训练模型", padding=10)
        train_frame.pack(fill=tk.X)
        ttk.Button(train_frame, text="打开数据集目录", command=self.open_dataset_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="打开模型输出目录", command=self.open_models_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="启动本地训练脚本", command=self.run_training_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="生成 YOLO 初始标注", command=self.run_yolo_seed_export_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="预览 YOLO 标注", command=self.run_yolo_preview_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="训练 YOLO ONNX", command=self.run_yolo_training_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="打开训练脚本位置", command=self.open_training_script_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="刷新训练结果", command=self.load_training_summary).pack(side=tk.LEFT, padx=4)

        deploy_frame = ttk.LabelFrame(self.train_tab, text="把模型文件传回 OpenMV N6", padding=10)
        deploy_frame.pack(fill=tk.X, pady=(10, 8))
        deploy_frame.columnconfigure(1, weight=1)

        ttk.Label(deploy_frame, text="模型 .tflite").grid(row=0, column=0, sticky="w")
        ttk.Entry(deploy_frame, textvariable=self.model_file_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(deploy_frame, text="选择模型", command=self.choose_model_file).grid(row=0, column=2, padx=4)

        ttk.Label(deploy_frame, text="标签 .txt").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(deploy_frame, textvariable=self.labels_file_var).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(deploy_frame, text="选择标签", command=self.choose_labels_file).grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(deploy_frame, text="OpenMV 盘符/目录").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(deploy_frame, textvariable=self.openmv_folder_var).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(deploy_frame, text="选择目录", command=self.choose_openmv_folder).grid(row=2, column=2, padx=4, pady=(8, 0))

        ttk.Button(deploy_frame, text="复制模型到 OpenMV N6", command=self.copy_model_to_openmv).grid(row=3, column=1, sticky="w", padx=8, pady=(12, 0))

        result_frame = ttk.LabelFrame(self.train_tab, text="最近训练结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.training_result_text = self._create_text(result_frame, height=9)

        script_frame = ttk.LabelFrame(self.train_tab, text="N6 端脚本提示", padding=10)
        script_frame.pack(fill=tk.X)
        text = (
            "采集数据时：运行 openmv/n6_usb_image_capture.py。\n"
            "部署模型后：运行 openmv/n6_classifier_main.py，它会加载 /lens_defect_classifier_int8.tflite 和 /lens_defect_labels.txt。\n"
            "如果模型较大，建议之后加一张 32GB microSD，把模型放 /sd/ 目录。"
        )
        ttk.Label(script_frame, text=text, wraplength=1000).pack(anchor="w")
        self.load_training_summary()

    def _create_text(self, parent, height):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        widget = tk.Text(frame, height=height, wrap="word", font=("Consolas", 10))
        widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        widget.configure(yscrollcommand=scroll.set)
        widget.insert("1.0", "暂无数据")
        widget.configure(state="disabled")
        return widget

    def refresh_ports(self):
        if list_ports is None:
            self.status_var.set("状态：缺少 pyserial，运行 run_windows_host.bat 会自动安装依赖")
            return

        ports = []
        for port in list_ports.comports():
            ports.append("%s - %s" % (port.device, port.description))

        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            preferred = [
                item for item in ports
                if "OpenMV" in item or "USB" in item or "串行设备" in item
            ]
            self.port_var.set(preferred[0] if preferred else ports[0])
            self.status_var.set("状态：找到 %d 个串口，请选择 OpenMV N6 对应 COM 口" % len(ports))
        elif not ports:
            self.port_var.set("")
            self.status_var.set("状态：未找到串口。请用 USB 连接 OpenMV N6")
            self.live_image_var.set("OpenMV 画面：未找到串口，请连接 N6 后点击“刷新串口”")

    def selected_port(self):
        selected = self.port_var.get().strip()
        if not selected:
            raise RuntimeError("请先选择 COM 口。")
        return selected.split(" - ", 1)[0].strip()

    def selected_baudrate(self):
        baudrate = self.baud_var.get().strip()
        int(baudrate)
        return baudrate

    def start_receive(self):
        if self.reader.running:
            messagebox.showinfo("提示", "当前已经在接收检测 JSON 和 OpenMV 画面。")
            return
        try:
            self.reader.start(self.selected_port(), self.selected_baudrate())
            self.status_var.set("状态：正在接收 OpenMV 检测 JSON 和画面")
            self.live_image_var.set("OpenMV 画面：等待 N6 发来图片数据（IMG_BEGIN）")
        except Exception as exc:
            messagebox.showerror("连接失败", str(exc))
            self.status_var.set("状态：连接失败")

    def stop_receive(self):
        self.reader.stop()
        self.status_var.set("状态：已停止接收")

    def choose_dataset_dir(self):
        path = filedialog.askdirectory(title="选择数据集目录", initialdir=self.dataset_dir_var.get())
        if path:
            self.dataset_dir_var.set(path)
            self.refresh_dataset_counts()

    def init_dataset_folders(self):
        dataset_dir = Path(self.dataset_dir_var.get())
        for split in ("train", "val", "test"):
            for class_name in DEFECT_TYPE_ORDER:
                (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        self.refresh_dataset_counts()
        self.log_capture("已初始化数据集目录：%s" % dataset_dir)

    def test_capture_connection(self):
        try:
            self.capture_one_image(test_only=True)
            messagebox.showinfo("成功", "已收到 OpenMV N6 发来的图片数据，连接正常。")
        except Exception as exc:
            messagebox.showerror("测试失败", str(exc))

    def capture_one_image(self, test_only=False):
        try:
            settings = self.capture_settings()
            payload = self.capture_and_store_image(settings, test_only=test_only)
            self.apply_capture_result(payload)
        except Exception as exc:
            if test_only:
                raise
            self.status_var.set("状态：采集失败")
            self.log_capture("采集失败：%s" % exc)
            messagebox.showerror("采集失败", str(exc))

    def start_auto_capture(self):
        if self.auto_capture_running:
            messagebox.showinfo("提示", "自动采集已经在运行。")
            return
        try:
            interval = float(self.capture_interval_var.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "自动采集间隔必须是大于 0 的数字。")
            return

        try:
            settings = self.capture_settings()
        except Exception as exc:
            messagebox.showerror("采集失败", str(exc))
            return

        self.auto_capture_running = True
        self.status_var.set("状态：开始自动采集训练图片")
        threading.Thread(target=self._auto_capture_loop, args=(interval, settings), daemon=True).start()

    def stop_auto_capture(self):
        self.auto_capture_running = False
        self.status_var.set("状态：已请求停止自动采集")

    def _auto_capture_loop(self, interval, settings):
        while self.auto_capture_running:
            try:
                payload = self.capture_and_store_image(settings, test_only=False)
                self.ui_queue.put(UiMessage("capture_saved", payload))
            except Exception as exc:
                self.ui_queue.put(UiMessage("capture_error", str(exc)))
                self.auto_capture_running = False
                break
            time.sleep(interval)
        self.ui_queue.put(UiMessage("status", "自动采集已停止"))

    def capture_settings(self):
        if self.reader.running:
            raise RuntimeError("检测接收正在占用串口，请先停止接收。")
        return {
            "dataset_dir": Path(self.dataset_dir_var.get()),
            "class_name": defect_type_key(self.dataset_class_var.get()),
            "split": self.dataset_split_var.get(),
            "auto_split": bool(self.auto_split_var.get()),
            "port": self.selected_port(),
            "baudrate": self.selected_baudrate(),
        }

    def capture_and_store_image(self, settings, test_only=False):
        dataset_dir = settings["dataset_dir"]
        class_name = settings["class_name"]
        split = settings["split"]
        port = settings["port"]
        baudrate = settings["baudrate"]

        image_bytes, width, height = UsbImageCaptureClient.capture(port, baudrate)
        quality = self.inspect_capture_image(image_bytes, width, height)

        save_path = None
        if not test_only:
            if settings["auto_split"]:
                split = self.choose_balanced_split(dataset_dir, class_name)
            save_dir = dataset_dir / split / class_name
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = "%s_%s_%s.jpg" % (class_name, split, safe_filename_time())
            save_path = save_dir / filename
            with open(save_path, "wb") as file:
                file.write(image_bytes)
            self.append_capture_metadata(
                dataset_dir=dataset_dir,
                save_path=save_path,
                split=split,
                class_name=class_name,
                port=port,
                baudrate=baudrate,
                width=quality["width"],
                height=quality["height"],
                byte_count=len(image_bytes),
                quality=quality,
            )

        return {
            "test_only": test_only,
            "save_path": str(save_path) if save_path else "",
            "split": split,
            "class_name": class_name,
            "byte_count": len(image_bytes),
            "width": quality["width"],
            "height": quality["height"],
            "image_bytes": image_bytes,
            "quality": quality,
        }

    def choose_balanced_split(self, dataset_dir, class_name):
        counts = {}
        for split in ("train", "val", "test"):
            counts[split] = count_image_files(dataset_dir / split / class_name)
        total_after = sum(counts.values()) + 1

        best_split = "train"
        best_score = None
        for candidate in ("train", "val", "test"):
            candidate_counts = dict(counts)
            candidate_counts[candidate] += 1
            score = 0.0
            for split, ratio in SPLIT_TARGET_RATIOS.items():
                actual = candidate_counts[split] / total_after
                score += abs(actual - ratio)
            if best_score is None or score < best_score:
                best_split = candidate
                best_score = score
        return best_split

    def inspect_capture_image(self, image_bytes, width, height):
        quality = {
            "width": width,
            "height": height,
            "brightness": "",
            "contrast": "",
            "status": "未检查",
            "warnings": [],
        }

        if Image is None or ImageStat is None:
            quality["warnings"].append("未安装 Pillow，只保存图片，不显示 JPEG 预览和亮度检查")
            return quality

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                quality["width"], quality["height"] = image.size
                gray = image.convert("L")
                stat = ImageStat.Stat(gray)
                brightness = float(stat.mean[0])
                contrast = float(stat.stddev[0])
                quality["brightness"] = "%.1f" % brightness
                quality["contrast"] = "%.1f" % contrast

                if brightness < 35:
                    quality["warnings"].append("图片偏暗，建议增加补光或曝光")
                elif brightness > 220:
                    quality["warnings"].append("图片偏亮，缺陷细节可能被过曝淹没")
                if contrast < 12:
                    quality["warnings"].append("对比度偏低，透明镜片缺陷可能不明显")
                if min(image.size) < 96:
                    quality["warnings"].append("图片尺寸偏小，建议保持至少 128x128 以上")
        except Exception as exc:
            quality["warnings"].append("图片质量检查失败：%s" % exc)

        quality["status"] = "通过" if not quality["warnings"] else "需复核"
        return quality

    def append_capture_metadata(self, dataset_dir, save_path, split, class_name, port, baudrate, width, height, byte_count, quality):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_path(dataset_dir)
        is_new_file = not path.exists()
        try:
            relative_path = str(save_path.relative_to(dataset_dir))
        except ValueError:
            relative_path = str(save_path)

        with open(path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow([
                    "capture_time",
                    "relative_path",
                    "split",
                    "label",
                    "width",
                    "height",
                    "bytes",
                    "brightness",
                    "contrast",
                    "quality_status",
                    "quality_warnings",
                    "port",
                    "baudrate",
                    "source",
                ])
            writer.writerow([
                now_text(),
                relative_path,
                split,
                class_name,
                width,
                height,
                byte_count,
                quality.get("brightness", ""),
                quality.get("contrast", ""),
                quality.get("status", ""),
                "；".join(quality.get("warnings", [])),
                port,
                baudrate,
                "OpenMV N6 USB",
            ])

    def apply_capture_result(self, payload):
        quality = payload["quality"]
        quality_text = self.format_quality_text(quality)
        if payload["test_only"]:
            self.log_capture(
                "测试采图成功：%d 字节，尺寸 %dx%d，%s"
                % (payload["byte_count"], payload["width"], payload["height"], quality_text)
            )
        else:
            self.log_capture(
                "保存图片：%s，类别 %s，集合 %s，尺寸 %dx%d，%s"
                % (
                    payload["save_path"],
                    defect_type_name(payload["class_name"]),
                    payload["split"],
                    payload["width"],
                    payload["height"],
                    quality_text,
                )
            )
            self.refresh_dataset_counts()
        self.update_capture_preview(payload)

    def format_quality_text(self, quality):
        warnings = quality.get("warnings") or []
        base = "质量：%s" % quality.get("status", "未检查")
        if quality.get("brightness"):
            base += "，亮度 %s" % quality["brightness"]
        if quality.get("contrast"):
            base += "，对比度 %s" % quality["contrast"]
        if warnings:
            base += "；" + "；".join(warnings)
        return base

    def update_capture_preview(self, payload):
        quality_text = self.format_quality_text(payload["quality"])
        location = "测试采图，未保存"
        if payload["save_path"]:
            location = "%s / %s" % (payload["split"], defect_type_name(payload["class_name"]))
        self.capture_preview_var.set(
            "最近图片：%s\n尺寸：%dx%d，大小：%d 字节"
            % (location, payload["width"], payload["height"], payload["byte_count"])
        )
        self.capture_quality_var.set(quality_text)

        if Image is None or ImageTk is None:
            self.capture_preview_label.configure(image="")
            return

        try:
            with Image.open(BytesIO(payload["image_bytes"])) as image:
                image.thumbnail((520, 220))
                self.capture_preview_photo = ImageTk.PhotoImage(image.copy())
            self.capture_preview_label.configure(image=self.capture_preview_photo, compound="top")
        except Exception as exc:
            self.capture_preview_photo = None
            self.capture_preview_label.configure(image="")
            self.capture_quality_var.set(quality_text + "；预览失败：%s" % exc)

    def refresh_dataset_counts(self):
        if not hasattr(self, "dataset_tree"):
            return
        dataset_dir = Path(self.dataset_dir_var.get())
        self.dataset_tree.delete(*self.dataset_tree.get_children())

        selected_class = defect_type_key(self.dataset_class_var.get())
        selected_total = 0
        totals = {}
        split_missing_classes = []
        for class_name in DEFECT_TYPE_ORDER:
            counts = []
            total = 0
            for split in ("train", "val", "test"):
                folder = dataset_dir / split / class_name
                count = count_image_files(folder)
                counts.append(count)
                total += count
            if class_name == selected_class:
                selected_total = total
            totals[class_name] = total
            if total > 0 and (counts[1] == 0 or counts[2] == 0):
                split_missing_classes.append(defect_type_name(class_name))
            self.dataset_tree.insert("", tk.END, values=(defect_type_name(class_name), counts[0], counts[1], counts[2], total))

        self.capture_count_var.set("当前类别图片数：%d" % selected_total)
        self.dataset_health_var.set(self.build_dataset_health_text(totals, split_missing_classes))

    def build_dataset_health_text(self, totals, split_missing_classes):
        image_total = sum(totals.values())
        active_classes = [class_name for class_name, total in totals.items() if total > 0]
        low_classes = [
            defect_type_name(class_name)
            for class_name, total in totals.items()
            if 0 < total < MIN_RECOMMENDED_IMAGES_PER_CLASS
        ]

        messages = ["数据集体检：共 %d 张，有数据类别 %d/%d" % (image_total, len(active_classes), len(DEFECT_TYPE_ORDER))]
        if low_classes:
            messages.append("不足 %d 张：%s" % (MIN_RECOMMENDED_IMAGES_PER_CLASS, "、".join(low_classes[:5])))
        if split_missing_classes:
            messages.append("缺少 val/test：%s" % "、".join(split_missing_classes[:5]))

        non_zero_totals = [total for total in totals.values() if total > 0]
        if len(non_zero_totals) >= 2 and min(non_zero_totals) > 0 and max(non_zero_totals) / min(non_zero_totals) > 3:
            messages.append("类别数量差异较大，训练可能偏向样本多的类别")
        if image_total == 0:
            messages.append("建议先每类至少采 100 张，稳定后再扩到 300 张以上")

        metadata = metadata_path(Path(self.dataset_dir_var.get()))
        if metadata.exists():
            messages.append("已记录 %s" % metadata.name)
        else:
            messages.append("下一次采集会自动生成 metadata.csv")

        return "；".join(messages)

    def open_dataset_dir(self):
        path = Path(self.dataset_dir_var.get())
        path.mkdir(parents=True, exist_ok=True)
        self.open_path(path)

    def open_models_dir(self):
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.open_path(DEFAULT_MODELS_DIR)

    def load_training_summary(self):
        if not hasattr(self, "training_result_text"):
            return

        summary_path = DEFAULT_MODELS_DIR / "training_summary.json"
        if not summary_path.exists():
            self._set_text(
                self.training_result_text,
                "暂无训练结果。训练结束后这里会读取 models/training_summary.json、training_history.csv 和混淆矩阵文件。",
            )
            return

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_text(self.training_result_text, "训练结果读取失败：%s" % exc)
            return

        lines = []
        lines.append("数据集：%s" % summary.get("dataset", ""))
        lines.append("输入尺寸：%s x %s" % (summary.get("image_size", ""), summary.get("image_size", "")))
        lines.append("类别顺序：%s" % "、".join(summary.get("labels", [])))
        metrics = summary.get("metrics", {})
        for split in ("train", "val", "test"):
            values = metrics.get(split)
            if not values:
                continue
            accuracy = values.get("accuracy")
            loss = values.get("loss")
            lines.append("%s：accuracy=%s，loss=%s" % (split, self.format_metric(accuracy), self.format_metric(loss)))

        artifacts = summary.get("artifacts", {})
        if artifacts.get("int8_tflite"):
            lines.append("INT8 模型：%s" % artifacts["int8_tflite"])
        if artifacts.get("confusion_matrix"):
            lines.append("混淆矩阵：%s" % artifacts["confusion_matrix"])
        lines.append("更新时间：%s" % now_text())
        self._set_text(self.training_result_text, "\n".join(lines))

    def format_metric(self, value):
        if value is None:
            return "暂无"
        try:
            return "%.4f" % float(value)
        except (TypeError, ValueError):
            return str(value)

    def run_training_script(self):
        script = find_training_script()
        if script is None:
            messagebox.showerror("找不到训练脚本", "未找到 training/train_lens_classifier.py")
            return

        dataset_dir = Path(self.dataset_dir_var.get())
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        python_exe = find_training_python()
        command = (
            '"%s" "%s" --dataset "%s" --output "%s" --epochs 20 --image-size 128'
            % (python_exe, script, dataset_dir, DEFAULT_MODELS_DIR)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开训练命令窗口")

    def run_yolo_seed_export_script(self):
        script = find_yolo_seed_export_script()
        if script is None:
            messagebox.showerror("找不到脚本", "未找到 training/export_yolo_seed_labels.py")
            return

        dataset_dir = Path(self.dataset_dir_var.get())
        output_dir = PROJECT_ROOT / "dataset_yolo_seed"
        python_exe = find_training_python()
        command = (
            '"%s" "%s" --dataset "%s" --output "%s" --copy-images --min-confidence 0.70'
            % (python_exe, script, dataset_dir, output_dir)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开 YOLO 初始标注生成窗口")

    def run_yolo_training_script(self):
        script = find_yolo_training_script()
        if script is None:
            messagebox.showerror("找不到脚本", "未找到 training/train_yolo_from_seed.py")
            return

        python_exe = find_training_python()
        data_yaml = PROJECT_ROOT / "dataset_yolo_seed" / "data.yaml"
        output_model = DEFAULT_YOLO_MODEL
        command = (
            '"%s" "%s" --data "%s" --output "%s" --epochs 60 --image-size 640 --batch 8'
            % (python_exe, script, data_yaml, output_model)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开 YOLO ONNX 训练窗口")

    def run_yolo_preview_script(self):
        script = find_yolo_preview_script()
        if script is None:
            messagebox.showerror("找不到脚本", "未找到 training/preview_yolo_seed_labels.py")
            return

        python_exe = find_training_python()
        dataset_dir = PROJECT_ROOT / "dataset_yolo_seed"
        output_dir = PROJECT_ROOT / "outputs" / "yolo_seed_preview"
        command = (
            '"%s" "%s" --dataset "%s" --output "%s" --split all --max-images 300'
            % (python_exe, script, dataset_dir, output_dir)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.open_path(output_dir)
        self.status_var.set("状态：已打开 YOLO 标注预览生成窗口")

    def open_training_script_folder(self):
        script = find_training_script()
        if script is None:
            messagebox.showerror("找不到训练脚本", "未找到 training/train_lens_classifier.py")
            return
        self.open_path(script.parent)

    def choose_model_file(self):
        path = filedialog.askopenfilename(
            title="选择 TFLite 模型",
            filetypes=(("TFLite 模型", "*.tflite"), ("所有文件", "*.*")),
        )
        if path:
            self.model_file_var.set(path)

    def choose_labels_file(self):
        path = filedialog.askopenfilename(
            title="选择标签文件",
            filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*")),
        )
        if path:
            self.labels_file_var.set(path)

    def choose_openmv_folder(self):
        path = filedialog.askdirectory(title="选择 OpenMV N6 USB 盘符或文件夹")
        if path:
            self.openmv_folder_var.set(path)

    def copy_model_to_openmv(self):
        model_path = Path(self.model_file_var.get())
        labels_path = Path(self.labels_file_var.get())
        target_dir = Path(self.openmv_folder_var.get())

        if not model_path.exists():
            messagebox.showerror("错误", "模型文件不存在：%s" % model_path)
            return
        if not labels_path.exists():
            messagebox.showerror("错误", "标签文件不存在：%s" % labels_path)
            return
        if not target_dir.exists():
            messagebox.showerror("错误", "OpenMV 目标目录不存在：%s" % target_dir)
            return

        target_model = target_dir / "lens_defect_classifier_int8.tflite"
        target_labels = target_dir / "lens_defect_labels.txt"
        shutil.copy2(model_path, target_model)
        shutil.copy2(labels_path, target_labels)
        messagebox.showinfo("完成", "已复制：\n%s\n%s" % (target_model, target_labels))
        self.status_var.set("状态：模型文件已复制到 OpenMV N6")

    def open_path(self, path):
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def log_capture(self, message):
        old = self.capture_log_text.get("1.0", tk.END).strip()
        new_text = "%s  %s" % (now_text(), message)
        if old and old != "暂无数据":
            new_text = old + "\n" + new_text
        self._set_text(self.capture_log_text, new_text)

    def choose_stage2_model(self):
        path = filedialog.askopenfilename(
            title="选择二级异常检测模型",
            initialdir=str(DEFAULT_MODELS_DIR),
            filetypes=[("Stage2 model", "*.npz"), ("All files", "*.*")],
        )
        if path:
            self.stage2_model_var.set(path)

    def choose_yolo_model(self):
        path = filedialog.askopenfilename(
            title="选择 YOLO ONNX 模型",
            initialdir=str(DEFAULT_MODELS_DIR),
            filetypes=[("YOLO ONNX", "*.onnx"), ("All files", "*.*")],
        )
        if path:
            self.yolo_model_var.set(path)

    def load_yolo_model_from_ui(self):
        try:
            detector = self.get_yolo_detector(force_reload=True)
        except Exception as exc:
            self.yolo_status_var.set("YOLO：加载失败")
            messagebox.showerror("YOLO 模型加载失败", str(exc))
            return
        if detector is None:
            messagebox.showwarning("YOLO 模型未加载", "没有找到 YOLO ONNX 模型。先使用电脑快速复核兜底。")
            return
        messagebox.showinfo("YOLO 模型已加载", "模型：%s\n类别：%s" % (detector.model_path, "、".join(detector.labels)))

    def get_yolo_detector(self, force_reload=False):
        if not self.yolo_enabled_var.get():
            self.yolo_status_var.set("YOLO：已关闭")
            return None
        raw_text = self.yolo_model_var.get().strip()
        signature = raw_text or str(DEFAULT_YOLO_MODEL)
        now = time.monotonic()
        if (
            not force_reload
            and self.yolo_detector is None
            and self.yolo_missing_signature == signature
            and now - self.yolo_missing_checked_at < YOLO_MISSING_MODEL_RECHECK_SECONDS
        ):
            return None
        candidates = []
        if raw_text:
            candidates.append(Path(raw_text))
        candidates.append(DEFAULT_YOLO_MODEL)
        path = None
        for candidate in candidates:
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            if candidate.exists():
                path = candidate
                break
        if path is None:
            self.yolo_detector = None
            self.yolo_model_path = None
            self.yolo_missing_signature = signature
            self.yolo_missing_checked_at = now
            self.yolo_status_var.set("YOLO：未找到模型，快速复核兜底")
            return None
        if force_reload or self.yolo_detector is None or self.yolo_model_path != path:
            self.yolo_detector = YoloOnnxDetector(path, DEFAULT_YOLO_LABELS)
            self.yolo_model_path = path
            self.yolo_missing_signature = None
            self.yolo_missing_checked_at = 0.0
        self.yolo_status_var.set("YOLO：已加载 %s" % path.name)
        return self.yolo_detector

    def load_stage2_model_from_ui(self):
        try:
            model = self.get_stage2_model(force_reload=True)
        except Exception as exc:
            self.stage2_status_var.set("二级复核：加载失败")
            messagebox.showerror("二级模型加载失败", str(exc))
            return
        if model is None:
            messagebox.showwarning("二级模型未加载", "没有找到二级模型，请先采集 normal 图片并训练。")
            return
        messagebox.showinfo("二级模型已加载", "正常样本数：%d\n阈值：%.2f" % (model.sample_count, model.threshold))

    def get_stage2_model(self, force_reload=False):
        if not self.stage2_enabled_var.get():
            self.stage2_status_var.set("高速复核：已启用；二级模型已关闭")
            return None
        if not ensure_stage2_runtime_loaded():
            detail = str(STAGE2_IMPORT_ERROR) if STAGE2_IMPORT_ERROR is not None else "缺少 OpenCV 或 NumPy"
            self.stage2_status_var.set("二级复核：依赖加载失败：%s" % detail)
            return None

        path = self.resolve_stage2_model_path()
        if path is None:
            self.stage2_model = None
            self.stage2_status_var.set("二级复核：未训练")
            return None
        self.stage2_model_var.set(str(path))

        mtime = path.stat().st_mtime
        if force_reload or self.stage2_model is None or self.stage2_model_path != path or self.stage2_model_mtime != mtime:
            self.stage2_model = Stage2AnomalyModel.load(path)
            self.stage2_model_path = path
            self.stage2_model_mtime = mtime
        self.stage2_status_var.set(
            "二级复核：已加载，正常样本 %d，阈值 %.2f"
            % (self.stage2_model.sample_count, self.stage2_model.threshold)
        )
        return self.stage2_model

    def resolve_stage2_model_path(self):
        raw_text = self.stage2_model_var.get().strip()
        candidates = []
        if raw_text:
            candidates.append(Path(raw_text))
        candidates.extend([
            DEFAULT_STAGE2_MODEL,
            PROJECT_ROOT / "models" / "lens_stage2_anomaly.npz",
            Path.cwd() / "models" / "lens_stage2_anomaly.npz",
            APP_DIR / "models" / "lens_stage2_anomaly.npz",
            APP_DIR.parent / "models" / "lens_stage2_anomaly.npz",
        ])

        seen = set()
        for candidate in candidates:
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate
        return None

    def preload_stage2_model(self):
        if not self.stage2_enabled_var.get():
            return
        try:
            self.get_stage2_model(force_reload=True)
        except Exception as exc:
            self.stage2_status_var.set("二级复核：加载失败：%s" % exc)

    def preload_yolo_model(self):
        if not self.yolo_enabled_var.get():
            return
        try:
            self.get_yolo_detector(force_reload=True)
        except Exception as exc:
            self.yolo_status_var.set("YOLO：加载失败：%s" % exc)

    def _poll_ui_queue(self):
        latest_live_payload = None
        while True:
            try:
                message = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if message.kind == "line":
                self.handle_json_line(message.payload)
            elif message.kind == "status":
                self.status_var.set("状态：" + str(message.payload))
            elif message.kind == "error":
                self.status_var.set("状态：" + str(message.payload))
                messagebox.showerror("串口错误", str(message.payload))
            elif message.kind == "no_data":
                self.status_var.set("状态：" + str(message.payload))
                if self.latest_live_image_payload is None:
                    self.live_image_var.set("OpenMV 画面：N6 暂无输出，请按 RESET 或重新插拔 USB")
            elif message.kind == "capture_error":
                self.status_var.set("状态：采集失败")
                self.log_capture("采集失败：%s" % message.payload)
                messagebox.showerror("采集失败", str(message.payload))
            elif message.kind == "capture_saved":
                self.apply_capture_result(message.payload)
            elif message.kind == "live_image":
                latest_live_payload = message.payload

        if latest_live_payload is not None:
            self.update_live_image(latest_live_payload)

        self.root.after(20, self._poll_ui_queue)

    def handle_json_line(self, raw_line):
        self._set_raw_text(raw_line)
        try:
            result = json.loads(raw_line)
            if not isinstance(result, dict):
                raise ValueError("JSON 顶层必须是对象")
        except Exception as exc:
            self.status_var.set("状态：JSON 解析失败：%s" % exc)
            return

        result = self.normalize_detection_result(result)
        result = self.filter_edge_defects_for_live_image(result, self.latest_live_image_payload)
        result = self.merge_recent_pc_defect_result(result)
        result = self.stabilize_detection_result(result)
        self.update_result_ui(result)
        self.add_history(result, raw_line)
        if result.get("error"):
            self.status_var.set("状态：OpenMV 错误：%s" % result.get("error"))
        else:
            self.status_var.set("状态：%s 收到一条有效 JSON" % now_text())

        if self.latest_live_image_payload is not None:
            try:
                self.render_live_image(self.latest_live_image_payload)
            except Exception:
                pass

    def normalize_detection_result(self, result):
        clean_result = dict(result)
        if result.get("error"):
            clean_result["defects"] = []
            clean_result["summary"] = {defect_type: 0 for defect_type in SUMMARY_TYPES}
            clean_result["defect_count"] = 0
            clean_result["has_defect"] = False
            clean_result["overall_level"] = "error"
            return clean_result

        source_defects = result.get("defects") or []
        has_defect_list = "defects" in result
        defects = [
            defect for defect in source_defects
            if isinstance(defect, dict) and defect.get("type") in SUPPORTED_DEFECT_TYPES
        ]
        summary = {defect_type: 0 for defect_type in SUMMARY_TYPES}

        if has_defect_list:
            for defect in defects:
                summary[defect["type"]] += 1
            defect_count = len(defects)
            overall_level = "normal"
            for defect in defects:
                level = defect.get("level", "normal")
                if LEVEL_SCORE.get(level, 0) > LEVEL_SCORE.get(overall_level, 0):
                    overall_level = level
        else:
            source_summary = result.get("summary") or {}
            defect_count = 0
            for defect_type in SUMMARY_TYPES:
                try:
                    summary[defect_type] = int(source_summary.get(defect_type, 0) or 0)
                except (TypeError, ValueError):
                    summary[defect_type] = 0
                defect_count += summary[defect_type]
            overall_level = result.get("overall_level", "normal") if defect_count else "normal"

        clean_result["defects"] = defects
        clean_result["summary"] = summary
        clean_result["defect_count"] = defect_count
        clean_result["has_defect"] = defect_count > 0
        clean_result["overall_level"] = overall_level
        return clean_result

    def merge_recent_pc_defect_result(self, result):
        if not isinstance(result, dict):
            return result
        if result.get("error") or result.get("has_defect"):
            return result

        recent = self.last_pc_defect_result
        if not isinstance(recent, dict) or not recent.get("has_defect"):
            return result
        if time.monotonic() - float(self.last_pc_defect_time or 0.0) > FAST_REVIEW_KEEP_PC_RESULT_SECONDS:
            return result

        recent_defects = [
            dict(defect)
            for defect in (recent.get("defects") or [])
            if isinstance(defect, dict) and defect.get("type") in SUPPORTED_DEFECT_TYPES
        ]
        if not recent_defects:
            return result

        merged = dict(result)
        merged["defects"] = recent_defects
        merged["pc_fast_review"] = True
        merged["pc_fast_review_hold"] = True
        for key in ("frame", "roi", "lens"):
            if isinstance(recent.get(key), dict):
                merged[key] = dict(recent[key])
        return self.normalize_detection_result(merged)

    def filter_edge_defects_for_live_image(self, result, payload):
        if cv2 is None or np is None or Image is None:
            return result
        if not isinstance(result, dict) or not isinstance(payload, dict):
            return result
        defects = result.get("defects") or []
        if not defects:
            return result

        try:
            with Image.open(BytesIO(payload["image_bytes"])) as image:
                display_image = image.convert("RGB")
        except Exception:
            return result

        return self.filter_edge_defects_for_image(result, payload, display_image)

    def filter_edge_defects_for_image(self, result, payload, display_image):
        if cv2 is None or np is None:
            return result
        if not isinstance(result, dict) or not isinstance(payload, dict):
            return result
        defects = result.get("defects") or []
        if not defects:
            return result
        mask = self.build_detection_mask(display_image, payload, result)
        if mask is None or cv2.countNonZero(mask) <= 0:
            return result
        edge_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        display_gray = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2GRAY)

        frame = result.get("frame") or {}
        source_w = self._positive_int(frame.get("w")) or self._positive_int(payload.get("width")) or display_image.width
        source_h = self._positive_int(frame.get("h")) or self._positive_int(payload.get("height")) or display_image.height
        roi = self._last_effective_roi or self.infer_effective_detection_roi(result, payload, display_image)
        filtered = []
        removed = 0
        for defect in defects:
            if self.should_keep_defect_inside_lens(
                defect,
                mask,
                source_w,
                source_h,
                display_image.width,
                display_image.height,
                roi,
                edge_distance,
                display_gray,
            ):
                filtered.append(defect)
            else:
                removed += 1

        if removed <= 0:
            return result

        refined = dict(result)
        refined["defects"] = filtered
        refined["edge_filter"] = {"removed": removed}
        return self.normalize_detection_result(refined)

    def infer_effective_detection_roi(self, result, payload, display_image):
        if not isinstance(result, dict):
            return None
        image_w, image_h = display_image.size
        frame = result.get("frame") or {}
        source_w = self._positive_int(frame.get("w")) or self._positive_int(payload.get("width")) or image_w
        source_h = self._positive_int(frame.get("h")) or self._positive_int(payload.get("height")) or image_h
        lens = result.get("lens") if isinstance(result.get("lens"), dict) else None
        roi = result.get("roi") if isinstance(result.get("roi"), dict) else None
        candidate = lens if lens is not None and lens.get("found", True) else roi
        candidate = self.valid_roi_or_none(candidate, source_w, source_h)
        if candidate is not None and self.roi_is_low_confidence(candidate, source_w, source_h):
            candidate = None
        if candidate is None or self.roi_is_full_frame(candidate, source_w, source_h):
            auto_roi = self.detect_lens_roi_from_image(display_image)
            if auto_roi is not None:
                x_scale = float(source_w) / float(max(1, image_w))
                y_scale = float(source_h) / float(max(1, image_h))
                return {
                    "x": int(auto_roi["x"] * x_scale),
                    "y": int(auto_roi["y"] * y_scale),
                    "w": int(auto_roi["w"] * x_scale),
                    "h": int(auto_roi["h"] * y_scale),
                    "source": "pc_auto_roi",
                }
            return self.center_fallback_roi(source_w, source_h)
        return candidate

    def should_keep_defect_inside_lens(
        self,
        defect,
        mask,
        source_w,
        source_h,
        image_w,
        image_h,
        roi=None,
        distance=None,
        display_gray=None,
    ):
        if defect.get("type") not in SUPPORTED_DEFECT_TYPES:
            return False

        if not self.defect_center_is_inside_middle_zone(defect, roi, source_w, source_h):
            return False

        defect_type = defect.get("type")
        x_scale = float(image_w) / float(max(1, source_w))
        y_scale = float(image_h) / float(max(1, source_h))
        x = int(self._positive_int(defect.get("x")) * x_scale)
        y = int(self._positive_int(defect.get("y")) * y_scale)
        w = int(self._positive_int(defect.get("w")) * x_scale)
        h = int(self._positive_int(defect.get("h")) * y_scale)
        if w <= 0 or h <= 0:
            return False

        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))
        if self.defect_looks_like_side_glare_line(
            defect,
            display_gray=display_gray,
            roi=roi,
            source_w=source_w,
            source_h=source_h,
            image_w=image_w,
            image_h=image_h,
            scaled_box=(x, y, w, h),
        ):
            return False
        if not self.defect_center_is_inside_inner_roi(defect, roi, source_w, source_h):
            return False
        if self.defect_box_near_roi_edge(defect, roi, source_w, source_h):
            return False

        roi_mask = mask[y:y + h, x:x + w]
        if roi_mask.size <= 0:
            return False

        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        touches_roi_edge = False
        if defect_type == "stain":
            touches_roi_edge = self.defect_touches_roi_edge(
                defect,
                roi,
                source_w,
                source_h,
                FAST_REVIEW_EDGE_FILTER_STAIN_ROI_MARGIN_RATIO,
            )
            if touches_roi_edge and aspect_ratio >= FAST_REVIEW_EDGE_FILTER_STAIN_EDGE_ASPECT:
                return False
        elif defect_type == "scratch":
            touches_roi_edge = self.defect_touches_roi_edge(
                defect,
                roi,
                source_w,
                source_h,
                FAST_REVIEW_EDGE_FILTER_SCRATCH_ROI_MARGIN_RATIO,
            )
            if touches_roi_edge and aspect_ratio >= FAST_REVIEW_EDGE_FILTER_SCRATCH_EDGE_ASPECT:
                return False

        if not self.defect_box_is_inside_roi(defect, roi, source_w, source_h):
            return False

        overlap = float(cv2.countNonZero(roi_mask)) / float(max(1, w * h))
        if defect_type == "scratch":
            min_overlap = FAST_REVIEW_EDGE_FILTER_SCRATCH_MIN_OVERLAP
        else:
            min_overlap = FAST_REVIEW_EDGE_FILTER_STAIN_MIN_OVERLAP
        if overlap < min_overlap:
            return False

        if distance is None:
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        center_x = max(0, min(image_w - 1, int(x + w / 2)))
        center_y = max(0, min(image_h - 1, int(y + h / 2)))
        if mask[center_y, center_x] == 0:
            return False

        min_center_distance = max(6, int(min(image_w, image_h) * 0.018))
        if defect_type == "stain":
            min_center_distance = max(
                min_center_distance,
                int(min(image_w, image_h) * FAST_REVIEW_EDGE_FILTER_STAIN_CENTER_DISTANCE_RATIO),
            )
        elif defect_type == "scratch":
            min_center_distance = max(
                min_center_distance,
                int(min(image_w, image_h) * FAST_REVIEW_EDGE_FILTER_SCRATCH_CENTER_DISTANCE_RATIO),
            )
        if distance[center_y, center_x] < min_center_distance:
            return False

        if defect_type == "stain":
            inner = distance[y:y + h, x:x + w]
            if inner.size <= 0:
                return False
            min_box_distance = max(4, int(min(image_w, image_h) * FAST_REVIEW_EDGE_FILTER_STAIN_BOX_DISTANCE_RATIO))
            if float(np.percentile(inner, 60)) < min_box_distance:
                return False
        elif defect_type == "scratch":
            inner = distance[y:y + h, x:x + w]
            if inner.size <= 0:
                return False
            min_box_distance = max(5, int(min(image_w, image_h) * FAST_REVIEW_EDGE_FILTER_SCRATCH_BOX_DISTANCE_RATIO))
            source = str(defect.get("source", ""))
            is_line_source = source in (
                "pc_fast_review_line",
                "pc_fast_bright_scratch",
                "pc_fast_bright_crosshatch",
                "pc_fast_center_cross_scratch",
                "yolo_onnx",
            )
            percentile = 58 if is_line_source else 50
            if float(np.percentile(inner, percentile)) < min_box_distance:
                return False
            if touches_roi_edge and is_line_source:
                return False

        return True

    def defect_center_is_inside_middle_zone(self, defect, roi, source_w, source_h):
        if not isinstance(roi, dict):
            return True
        rx = self._positive_int(roi.get("x"))
        ry = self._positive_int(roi.get("y"))
        rw = self._positive_int(roi.get("w"))
        rh = self._positive_int(roi.get("h"))
        if rw <= 0 or rh <= 0:
            return True

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        x_margin = max(10, int(rw * FAST_REVIEW_CENTER_DEFECT_X_MARGIN_RATIO))
        y_margin = max(8, int(rh * FAST_REVIEW_CENTER_DEFECT_Y_MARGIN_RATIO))
        return (
            center_x >= rx + x_margin
            and center_x <= rx + rw - x_margin
            and center_y >= ry + y_margin
            and center_y <= ry + rh - y_margin
        )

    def defect_looks_like_side_glare_line(
        self,
        defect,
        display_gray,
        roi,
        source_w,
        source_h,
        image_w,
        image_h,
        scaled_box=None,
    ):
        if defect.get("type") != "scratch":
            return False
        source = str(defect.get("source", ""))
        if source not in (
            "yolo_onnx",
            "pc_fast_review_line",
            "pc_fast_bright_scratch",
            "pc_fast_bright_crosshatch",
            "pc_fast_center_cross_scratch",
        ):
            return False

        if scaled_box is None:
            x_scale = float(image_w) / float(max(1, source_w))
            y_scale = float(image_h) / float(max(1, source_h))
            x = int(self._positive_int(defect.get("x")) * x_scale)
            y = int(self._positive_int(defect.get("y")) * y_scale)
            w = int(self._positive_int(defect.get("w")) * x_scale)
            h = int(self._positive_int(defect.get("h")) * y_scale)
        else:
            x, y, w, h = scaled_box
        if w <= 0 or h <= 0:
            return False

        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        width_ratio = float(min(w, h)) / float(max(1, min(image_w, image_h)))
        if aspect_ratio < FAST_REVIEW_FRAME_TALL_LINE_MIN_ASPECT:
            return False
        if width_ratio > FAST_REVIEW_FRAME_TALL_LINE_MAX_WIDTH_RATIO:
            return False

        center_x = x + w / 2.0
        side_boundary = image_w * FAST_REVIEW_FRAME_SIDE_REJECT_RATIO
        if isinstance(roi, dict):
            rx = self._positive_int(roi.get("x"))
            rw = self._positive_int(roi.get("w"))
            if rw > 0:
                rx *= float(image_w) / float(max(1, source_w))
                rw *= float(image_w) / float(max(1, source_w))
                side_boundary = rx + rw * FAST_REVIEW_FRAME_SIDE_REJECT_RATIO
        if center_x < side_boundary:
            return False

        if source == "yolo_onnx":
            return True

        if display_gray is None:
            return False

        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))
        roi_gray = display_gray[y:y + h, x:x + w]
        if roi_gray.size <= 0:
            return False
        local_pad = max(8, int(min(image_w, image_h) * 0.025))
        lx1 = max(0, x - local_pad)
        ly1 = max(0, y - local_pad)
        lx2 = min(image_w, x + w + local_pad)
        ly2 = min(image_h, y + h + local_pad)
        local_gray = display_gray[ly1:ly2, lx1:lx2]
        local_mean = float(np.mean(local_gray)) if local_gray.size else float(np.mean(display_gray))
        roi_p90 = float(np.percentile(roi_gray, 90.0))
        return (
            roi_p90 >= FAST_REVIEW_GLARE_LINE_MIN_BRIGHTNESS
            and roi_p90 - local_mean >= FAST_REVIEW_GLARE_LINE_MIN_LOCAL_DELTA
        )

    def defect_box_is_inside_roi(self, defect, roi, source_w, source_h):
        if not isinstance(roi, dict):
            return True
        rx = self._positive_int(roi.get("x"))
        ry = self._positive_int(roi.get("y"))
        rw = self._positive_int(roi.get("w"))
        rh = self._positive_int(roi.get("h"))
        if rw <= 0 or rh <= 0:
            return True

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        tolerance = max(
            4,
            int(min(source_w, source_h, rw, rh) * FAST_REVIEW_ROI_CONTAINMENT_TOLERANCE_RATIO),
        )
        return (
            x >= rx - tolerance
            and y >= ry - tolerance
            and x + w <= rx + rw + tolerance
            and y + h <= ry + rh + tolerance
        )

    def defect_center_is_inside_inner_roi(self, defect, roi, source_w, source_h):
        if not isinstance(roi, dict):
            return True
        rx = self._positive_int(roi.get("x"))
        ry = self._positive_int(roi.get("y"))
        rw = self._positive_int(roi.get("w"))
        rh = self._positive_int(roi.get("h"))
        if rw <= 0 or rh <= 0:
            return True

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        margin = max(12, int(min(source_w, source_h, rw, rh) * FAST_REVIEW_CENTER_ROI_MARGIN_RATIO))
        return (
            center_x >= rx + margin
            and center_x <= rx + rw - margin
            and center_y >= ry + margin
            and center_y <= ry + rh - margin
        )

    def defect_box_near_roi_edge(self, defect, roi, source_w, source_h):
        if not isinstance(roi, dict):
            return False
        rx = self._positive_int(roi.get("x"))
        ry = self._positive_int(roi.get("y"))
        rw = self._positive_int(roi.get("w"))
        rh = self._positive_int(roi.get("h"))
        if rw <= 0 or rh <= 0:
            return False

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        margin = max(6, int(min(source_w, source_h, rw, rh) * FAST_REVIEW_BOX_ROI_MARGIN_RATIO))
        return (
            x <= rx + margin
            or y <= ry + margin
            or x + w >= rx + rw - margin
            or y + h >= ry + rh - margin
        )

    def defect_touches_roi_edge(self, defect, roi, source_w, source_h, margin_ratio):
        if not isinstance(roi, dict):
            return False

        rx = self._positive_int(roi.get("x"))
        ry = self._positive_int(roi.get("y"))
        rw = self._positive_int(roi.get("w"))
        rh = self._positive_int(roi.get("h"))
        if rw <= 0 or rh <= 0:
            return False

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        margin = max(10, int(min(source_w, source_h, rw, rh) * margin_ratio))
        return (
            x <= rx + margin
            or y <= ry + margin
            or x + w >= rx + rw - margin
            or y + h >= ry + rh - margin
        )

    def display_class_result(self, result):
        if result.get("error"):
            return "error", "模型错误", None

        if not result.get("has_defect", False):
            return "normal", "正常镜片", None

        defects = result.get("defects") or []
        if defects:
            primary = max(defects, key=lambda item: float(item.get("confidence", 0) or 0))
            defect_type = primary.get("type", "")
            confidence = primary.get("confidence", None)
            return defect_type, defect_type_name(defect_type), confidence

        summary = result.get("summary") or {}
        active = [(key, int(summary.get(key, 0) or 0)) for key in SUMMARY_TYPES]
        active = [item for item in active if item[1] > 0]
        if active:
            defect_type = max(active, key=lambda item: item[1])[0]
            return defect_type, defect_type_name(defect_type), None
        return "normal", "正常镜片", None

    def stabilize_detection_result(self, result):
        class_key, _class_text, _confidence = self.display_class_result(result)
        if self.stable_detection_result is None:
            self.stable_detection_result = result
            self.pending_detection_class = None
            self.pending_detection_count = 0
            return result

        stable_key, _stable_text, _stable_confidence = self.display_class_result(self.stable_detection_result)
        if class_key == stable_key:
            self.stable_detection_result = result
            self.pending_detection_class = None
            self.pending_detection_count = 0
            return result

        if class_key != self.pending_detection_class:
            self.pending_detection_class = class_key
            self.pending_detection_count = 1
        else:
            self.pending_detection_count += 1

        required_count = HOST_NORMAL_CONFIRM_UPDATES if class_key == "normal" else HOST_DEFECT_CONFIRM_UPDATES
        if self.pending_detection_count >= required_count:
            self.stable_detection_result = result
            self.pending_detection_class = None
            self.pending_detection_count = 0
            return result

        held = dict(self.stable_detection_result)
        held["stabilizer"] = {
            "holding": stable_key,
            "candidate": class_key,
            "candidate_count": self.pending_detection_count,
            "required_count": required_count,
        }
        return held

    def update_result_ui(self, result, render_live_image=True):
        self.latest_detection_result = result
        has_defect = bool(result.get("has_defect", False))
        defect_count = int(result.get("defect_count", 0))
        overall_level = result.get("overall_level", "normal")
        class_key, class_text, confidence = self.display_class_result(result)

        self.class_result_var.set("识别结果：%s" % class_text)
        if confidence is None:
            self.confidence_result_var.set("置信度：--")
        else:
            self.confidence_result_var.set("置信度：%.2f" % float(confidence))
        if hasattr(self, "class_result_label"):
            color = {
                "error": "#B3261E",
                "normal": "#1E6B5C",
                "scratch": "#B3261E",
                "stain": "#9A5B00",
            }.get(class_key, "#333333")
            self.class_result_label.configure(fg=color)
        self.has_defect_var.set("是否检测到缺陷：%s" % ("是" if has_defect else "否"))
        self.count_var.set("缺陷总数：%d" % defect_count)
        self.level_var.set("整体严重程度：%s" % level_name(overall_level))
        self.lens_track_var.set(self.format_lens_track_text(result.get("lens")))
        if result.get("error"):
            self.status_var.set("状态：OpenMV 错误：%s" % result.get("error"))

        if self.summary_tree is not None:
            self.summary_tree.delete(*self.summary_tree.get_children())
            summary = result.get("summary") or {}
            for defect_type in SUMMARY_TYPES:
                self.summary_tree.insert("", tk.END, values=(defect_type_name(defect_type), summary.get(defect_type, 0)))

        if self.defect_tree is not None:
            self.defect_tree.delete(*self.defect_tree.get_children())
            defects = result.get("defects") or []
            for defect in defects:
                self.defect_tree.insert(
                    "",
                    tk.END,
                    values=(
                        defect_type_name(defect.get("type", "")),
                        "%.2f" % float(defect.get("confidence", 0)),
                        level_name(defect.get("level", "normal")),
                        defect.get("x", 0),
                        defect.get("y", 0),
                        defect.get("w", 0),
                        defect.get("h", 0),
                        defect.get("area", 0),
                        defect.get("length", 0),
                        "%.2f" % float(defect.get("aspect_ratio", 0)),
                    ),
                )

        if render_live_image and self.latest_live_image_payload is not None:
            self.render_live_image(self.latest_live_image_payload)

    def update_live_image(self, payload):
        self.latest_live_image_payload = payload
        self.live_image_var.set("")
        if Image is None or ImageTk is None:
            self.live_image_label.configure(image="")
            return

        self.render_live_image(payload)

    def render_live_image(self, payload):
        if Image is None or ImageTk is None:
            return

        try:
            with Image.open(BytesIO(payload["image_bytes"])) as image:
                display_image = image.convert("RGB")
                self.apply_yolo_to_live_image(display_image, payload)
                self.apply_fast_cv_review_to_live_image(display_image, payload)
                self.apply_stage2_to_live_image(display_image, payload)
                filtered = self.filter_edge_defects_for_live_image(self.latest_detection_result, payload)
                if filtered is not self.latest_detection_result:
                    self.update_result_ui(filtered, render_live_image=False)
                self.draw_detection_boxes(display_image, payload)
                display_image = self.fill_live_image_area(display_image)
                self.live_image_photo = ImageTk.PhotoImage(display_image.copy())
            self.live_image_label.configure(image=self.live_image_photo, compound="top")
        except Exception as exc:
            self.live_image_photo = None
            self.live_image_label.configure(image="")
            self.live_image_var.set("OpenMV 画面：预览失败：%s" % exc)

    def fill_live_image_area(self, image):
        width = max(1, self.live_image_label.winfo_width() - 4)
        height = max(1, self.live_image_label.winfo_height() - 4)
        if width <= 4 or height <= 4:
            width, height = 640, 480
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        source_w, source_h = image.size
        scale = min(float(width) / float(source_w), float(height) / float(source_h), 1.0)
        if scale < 1.0:
            resized_w = max(1, int(source_w * scale))
            resized_h = max(1, int(source_h * scale))
            resized = image.resize((resized_w, resized_h), resampling)
        else:
            resized_w, resized_h = source_w, source_h
            resized = image
        filled = Image.new("RGB", (width, height), "#20242A")
        left = max(0, (width - resized_w) // 2)
        top = max(0, (height - resized_h) // 2)
        filled.paste(resized, (left, top))
        return filled

    def on_live_image_resize(self, _event):
        if self.latest_live_image_payload is not None:
            self.render_live_image(self.latest_live_image_payload)

    def fast_review_iou(self, first, second):
        ax = int(first.get("x", 0))
        ay = int(first.get("y", 0))
        aw = int(first.get("w", 0))
        ah = int(first.get("h", 0))
        bx = int(second.get("x", 0))
        by = int(second.get("y", 0))
        bw = int(second.get("w", 0))
        bh = int(second.get("h", 0))
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0

        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        if union <= 0:
            return 0.0
        return float(intersection) / float(union)

    def confirm_fast_review_detection(self, detection):
        if detection is None:
            self.fast_review_candidate = None
            self.fast_review_candidate_count = 0
            return False

        same_candidate = (
            isinstance(self.fast_review_candidate, dict)
            and self.fast_review_candidate.get("type") == detection.get("type")
            and self.fast_review_iou(self.fast_review_candidate, detection) >= FAST_REVIEW_IOU_THRESHOLD
        )
        if same_candidate:
            self.fast_review_candidate_count += 1
            self.fast_review_candidate = detection
        else:
            self.fast_review_candidate = detection
            self.fast_review_candidate_count = 1

        if self.fast_review_candidate_count < FAST_REVIEW_DEFECT_CONFIRM_FRAMES:
            return False

        detection["fast_review_confirmed_frames"] = self.fast_review_candidate_count
        return True

    def fast_review_current_is_classifier_roi(self, result):
        model_name = str(result.get("model", "")).lower()
        if "classifier" in model_name and not result.get("pc_fast_review"):
            return True

        defects = result.get("defects") or []
        if len(defects) != 1:
            return False

        defect = defects[0]
        frame = result.get("frame") or {}
        frame_w = self._positive_int(frame.get("w"))
        frame_h = self._positive_int(frame.get("h"))
        if frame_w <= 0 or frame_h <= 0:
            return False

        defect_w = self._positive_int(defect.get("w"))
        defect_h = self._positive_int(defect.get("h"))
        defect_area_ratio = float(defect_w * defect_h) / float(max(1, frame_w * frame_h))
        return defect_area_ratio >= FAST_REVIEW_CLASSIFIER_AREA_RATIO

    def should_apply_fast_review_detection(self, detection):
        detection_type = detection.get("type")
        if detection_type not in SUPPORTED_DEFECT_TYPES:
            return False

        try:
            confidence = float(detection.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0

        current_key, _current_text, _current_confidence = self.display_class_result(self.latest_detection_result)
        if current_key == "normal":
            if not FAST_REVIEW_PROMOTE_NORMAL:
                return False
            if detection_type == "scratch":
                if not detection.get("strong_internal_scratch"):
                    return False
                return confidence >= FAST_REVIEW_STRONG_SCRATCH_CONFIDENCE
            if detection_type == "stain":
                return confidence >= FAST_REVIEW_STRONG_STAIN_CONFIDENCE
            return False

        if current_key == detection_type:
            return True

        # The OpenMV classifier reports one broad ROI box. A confirmed PC-side
        # shape check is more reliable for scratch-vs-stain correction.
        if self.fast_review_current_is_classifier_roi(self.latest_detection_result):
            return True

        if current_key == "stain" and detection_type == "scratch":
            if not detection.get("strong_internal_scratch"):
                return False
            return confidence >= FAST_REVIEW_STRONG_SCRATCH_CONFIDENCE

        if current_key in SUPPORTED_DEFECT_TYPES and current_key != detection_type:
            return False

        if detection_type == "scratch":
            if not detection.get("strong_internal_scratch"):
                return False
            return confidence >= FAST_REVIEW_STRONG_SCRATCH_CONFIDENCE
        if detection_type == "stain":
            return confidence >= FAST_REVIEW_STRONG_STAIN_CONFIDENCE
        return False

    def apply_fast_cv_review_to_live_image(self, image, payload):
        if not FAST_REVIEW_ENABLED or cv2 is None or np is None:
            return
        if not isinstance(self.latest_detection_result, dict):
            return

        payload_key = "%s:%s" % (payload.get("receive_time", ""), payload.get("byte_count", ""))
        if payload_key == self.last_fast_review_key:
            return
        self.last_fast_review_key = payload_key
        now = time.monotonic()
        if now - self.last_fast_review_time < FAST_REVIEW_MIN_INTERVAL_SECONDS:
            return
        self.last_fast_review_time = now

        mask = self.build_detection_mask(image, payload, self.latest_detection_result)
        detection = self.fast_cv_detect_defect(image, mask)
        if detection is None:
            self.confirm_fast_review_detection(None)
            return

        if not self.confirm_fast_review_detection(detection):
            return

        if not self.should_apply_fast_review_detection(detection):
            return

        refined = dict(self.latest_detection_result)
        refined["frame"] = {"w": image.width, "h": image.height}
        refined["roi"] = self.best_live_detection_roi(
            self.latest_detection_result,
            image.width,
            image.height,
        )
        refined["defects"] = [detection]
        refined["pc_fast_review"] = True
        refined = self.normalize_detection_result(refined)
        refined = self.filter_edge_defects_for_image(refined, payload, image)
        if refined.get("has_defect"):
            self.last_pc_defect_result = dict(refined)
            self.last_pc_defect_time = time.monotonic()
        refined = self.stabilize_detection_result(refined)
        self.update_result_ui(refined, render_live_image=False)

    def apply_yolo_to_live_image(self, image, payload):
        if cv2 is None or np is None:
            return
        payload_key = "%s:%s" % (payload.get("receive_time", ""), payload.get("byte_count", ""))
        if payload_key == self.last_yolo_payload_key:
            return
        now = time.monotonic()
        if now - self.last_yolo_infer_time < YOLO_MIN_INTERVAL_SECONDS:
            return
        detector = self.get_yolo_detector()
        if detector is None:
            return

        self.last_yolo_payload_key = payload_key
        self.last_yolo_infer_time = now
        detections = detector.detect(image)
        if not detections:
            return

        base = self.latest_detection_result if isinstance(self.latest_detection_result, dict) else {}
        refined = dict(base)
        refined["frame"] = {"w": image.width, "h": image.height}
        refined["roi"] = self.best_live_detection_roi(base, image.width, image.height)
        refined["defects"] = detections
        refined["model"] = "yolo_onnx"
        refined["yolo"] = {"enabled": True, "model_path": str(detector.model_path)}
        refined = self.normalize_detection_result(refined)
        refined = self.filter_edge_defects_for_image(refined, payload, image)
        if not refined.get("has_defect"):
            mask = self.build_detection_mask(image, payload, refined)
            fallback_detection = self.fast_cv_detect_defect(image, mask)
            if fallback_detection is not None and fallback_detection.get("type") == "scratch":
                fallback_result = dict(refined)
                fallback_result["defects"] = [fallback_detection]
                fallback_result["pc_fast_review"] = True
                fallback_result["yolo_edge_fallback"] = True
                fallback_result = self.normalize_detection_result(fallback_result)
                fallback_result = self.filter_edge_defects_for_image(fallback_result, payload, image)
                if fallback_result.get("has_defect"):
                    refined = fallback_result
        if refined.get("has_defect"):
            self.last_pc_defect_result = dict(refined)
            self.last_pc_defect_time = time.monotonic()
        refined = self.stabilize_detection_result(refined)
        self.update_result_ui(refined, render_live_image=False)

    def best_live_detection_roi(self, result, image_w, image_h):
        if isinstance(result, dict):
            lens = result.get("lens")
            if isinstance(lens, dict) and lens.get("found", True):
                roi = self.valid_roi_or_none(lens, image_w, image_h)
                if roi is not None and not self.roi_is_low_confidence(roi, image_w, image_h):
                    return roi
            roi = self.valid_roi_or_none(result.get("roi"), image_w, image_h)
            if roi is not None and not self.roi_is_low_confidence(roi, image_w, image_h):
                return roi
        return self.center_fallback_roi(image_w, image_h)

    def valid_roi_or_none(self, roi, image_w, image_h):
        if not isinstance(roi, dict):
            return None
        x = self._positive_int(roi.get("x"))
        y = self._positive_int(roi.get("y"))
        w = self._positive_int(roi.get("w"))
        h = self._positive_int(roi.get("h"))
        if w <= 8 or h <= 8:
            return None
        x = max(0, min(x, max(0, image_w - 1)))
        y = max(0, min(y, max(0, image_h - 1)))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))
        clean = {"x": x, "y": y, "w": w, "h": h}
        if "source" in roi:
            clean["source"] = str(roi.get("source", ""))
        if "confidence" in roi:
            try:
                clean["confidence"] = float(roi.get("confidence") or 0.0)
            except (TypeError, ValueError):
                clean["confidence"] = 0.0
        if "found" in roi:
            clean["found"] = bool(roi.get("found"))
        return clean

    def roi_is_full_frame(self, roi, image_w, image_h):
        if not isinstance(roi, dict):
            return False
        x = self._positive_int(roi.get("x"))
        y = self._positive_int(roi.get("y"))
        w = self._positive_int(roi.get("w"))
        h = self._positive_int(roi.get("h"))
        return (
            x <= int(image_w * 0.03)
            and y <= int(image_h * 0.03)
            and x + w >= int(image_w * 0.97)
            and y + h >= int(image_h * 0.97)
        )

    def roi_is_low_confidence(self, roi, image_w, image_h):
        if not isinstance(roi, dict):
            return True
        source = str(roi.get("source", "")).lower()
        if roi.get("found") is False or source in ("fallback", "pc_center_fallback"):
            return True
        try:
            confidence = float(roi.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence <= 0.0 and source in ("fallback", ""):
            return True
        x = self._positive_int(roi.get("x"))
        y = self._positive_int(roi.get("y"))
        w = self._positive_int(roi.get("w"))
        h = self._positive_int(roi.get("h"))
        if w <= 0 or h <= 0:
            return True
        frame_area = float(max(1, image_w * image_h))
        area_ratio = float(w * h) / frame_area
        if self.roi_is_full_frame(roi, image_w, image_h) or area_ratio >= 0.76:
            return True
        center_x = (x + w / 2.0) / float(max(1, image_w))
        center_y = (y + h / 2.0) / float(max(1, image_h))
        return abs(center_x - 0.50) > 0.30 or abs(center_y - 0.53) > 0.30

    def center_fallback_roi(self, image_w, image_h):
        margin_x = int(image_w * 0.27)
        margin_y = int(image_h * 0.22)
        return {
            "x": margin_x,
            "y": margin_y,
            "w": max(1, image_w - margin_x * 2),
            "h": max(1, image_h - margin_y * 2),
            "source": "pc_center_fallback",
        }

    def fast_cv_detect_defect(self, image, mask=None):
        rgb = np.array(image)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width < 32 or height < 32:
            return None

        if mask is None:
            mask = self.build_detection_mask(image, {}, self.latest_detection_result)
        if mask is None:
            return None
        if cv2.countNonZero(mask) <= 0:
            return None

        mean_value = float(np.mean(gray[mask > 0]))
        std_value = float(np.std(gray[mask > 0]))
        contrast_delta = max(18.0, std_value * 1.2)
        edge_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        stain = self.fast_cv_find_stain(gray, mask, mean_value, contrast_delta, edge_distance)
        dark_cluster = self.fast_cv_find_dark_stain_cluster(gray, mask, mean_value, std_value, edge_distance)
        if dark_cluster is not None and (stain is None or dark_cluster.get("area", 0) >= stain.get("area", 0)):
            stain = dark_cluster
        scratch = self.fast_cv_find_scratch(gray, mask, edge_distance, None)
        return self.choose_fast_cv_defect(scratch, stain, mask)

    def choose_fast_cv_defect(self, scratch, stain, mask):
        if scratch is None:
            if self.stain_candidate_looks_like_scratch(stain, mask):
                return self.promote_stain_candidate_to_scratch(stain)
            return stain
        if stain is None:
            if self.scratch_candidate_looks_like_dark_stain(scratch):
                return self.promote_scratch_candidate_to_stain(scratch)
            return scratch

        scratch_area = int(scratch.get("area", 0) or 0)
        scratch_length = int(scratch.get("length", 0) or 0)
        scratch_aspect = float(scratch.get("aspect_ratio", 0) or 0)
        scratch_strong = bool(scratch.get("strong_internal_scratch"))
        stain_area = int(stain.get("area", 0) or 0)
        stain_aspect = float(stain.get("aspect_ratio", 0) or 0)
        stain_density = float(stain.get("density", 0) or 0)
        mask_area = max(1, int(cv2.countNonZero(mask))) if mask is not None else 1
        scratch_area_ratio = float(scratch_area) / float(mask_area)
        scratch_box_ratio = float(self._positive_int(scratch.get("w")) * self._positive_int(scratch.get("h"))) / float(mask_area)
        stain_area_ratio = float(stain_area) / float(mask_area)

        scratch_source = str(scratch.get("source", ""))
        scratch_line_count = int(scratch.get("line_count", 0) or 0)
        scratch_total_line_length = float(scratch.get("total_line_length", 0) or 0)
        scratch_signed_delta = float(scratch.get("local_signed_delta", 0) or 0)
        scratch_bright_delta = float(scratch.get("local_bright_delta", 0) or 0)
        scratch_dark_delta = float(scratch.get("local_dark_delta", 0) or 0)
        scratch_slender_lines = int(scratch.get("slender_line_count", 0) or 0)
        scratch_is_long_slender = (
            scratch_aspect >= 3.5
            and scratch_length >= 80
            and scratch_area_ratio <= 0.018
        )
        if scratch_is_long_slender:
            return scratch
        scratch_is_crosshatch = (
            scratch_source in ("pc_fast_review_line", "pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch")
            and scratch_strong
            and scratch_line_count >= 4
            and (
                scratch_source not in ("pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch")
                or scratch_slender_lines >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LINES
            )
            and scratch_total_line_length >= FAST_REVIEW_SCRATCH_CROSSHATCH_MIN_LINE_LENGTH
            and scratch_area <= max(2200, int(stain_area * 0.28))
            and scratch_bright_delta >= max(5.0, scratch_dark_delta * 0.70)
        )
        if scratch_is_crosshatch and stain_area_ratio < FAST_REVIEW_SCRATCH_CROSSHATCH_STAIN_MAX_RATIO:
            return scratch
        scratch_is_star = (
            scratch_source in ("pc_fast_review_line", "pc_fast_bright_scratch", "pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch")
            and scratch_strong
            and scratch_line_count >= FAST_REVIEW_STAR_SCRATCH_MIN_LINES
            and (
                scratch_source not in ("pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch")
                or scratch_slender_lines >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LINES
            )
            and scratch_total_line_length >= FAST_REVIEW_STAR_SCRATCH_MIN_LENGTH
            and scratch_area_ratio <= 0.055
            and stain_area_ratio < 0.26
            and scratch_bright_delta >= max(5.0, scratch_dark_delta * 0.70)
        )
        dark_star_stain = (
            scratch_source == "pc_fast_review_line"
            and scratch_line_count >= FAST_REVIEW_STAR_SCRATCH_MIN_LINES
            and scratch_dark_delta > scratch_bright_delta
            and scratch_signed_delta <= FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA
            and stain is not None
        )
        if dark_star_stain:
            return stain
        if scratch_is_star:
            return scratch
        bright_line_scratch = (
            scratch_source in ("pc_fast_review_line", "pc_fast_bright_scratch", "pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch")
            and scratch_signed_delta >= FAST_REVIEW_BRIGHT_SCRATCH_MIN_SIGNED_DELTA
            and scratch_bright_delta >= max(6.0, scratch_dark_delta * 0.9)
            and scratch_total_line_length >= 82.0
            and scratch_line_count >= 3
            and (
                scratch_source not in ("pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch")
                or scratch_slender_lines >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LINES
            )
            and scratch_area_ratio <= 0.020
        )
        if bright_line_scratch:
            return scratch
        if stain_density >= 0.32 and stain_area_ratio >= 0.22 and scratch_area_ratio < 0.04:
            return stain
        if stain_density >= 0.45 and stain_area_ratio >= 0.12 and scratch_area <= max(1500, stain_area * 0.08):
            return stain
        if stain_density >= 0.24 and stain_area_ratio >= 0.20 and scratch_area_ratio < 0.030:
            return stain
        if stain_density >= 0.30 and stain_area_ratio >= 0.08 and scratch_aspect < 2.6:
            return stain
        if stain_area_ratio >= 0.12 and stain_density >= 0.20 and scratch_aspect < 2.2:
            return stain
        if scratch_strong and scratch_aspect >= 2.0 and scratch_length >= 45 and stain_area_ratio < 0.18:
            return scratch
        if scratch_aspect >= 3.0 and scratch_area <= max(2200, stain_area * 0.35):
            return scratch
        if stain_density >= 0.18 and stain_aspect <= 3.4 and stain_area >= max(420, scratch_area * 2):
            return stain
        return scratch if scratch_strong else stain

    def stain_candidate_looks_like_scratch(self, stain, mask):
        if not isinstance(stain, dict):
            return False
        if str(stain.get("source", "")) != "pc_fast_dark_stain":
            return False
        mask_area = max(1, int(cv2.countNonZero(mask))) if mask is not None else 1
        area = float(stain.get("area", 0) or 0)
        density = float(stain.get("density", 0) or 0)
        angle_groups = int(stain.get("angle_groups", 0) or 0)
        length = float(stain.get("length", 0) or 0)
        aspect_ratio = float(stain.get("aspect_ratio", 0) or 0)
        signed_delta = float(stain.get("brightness_signed_delta", 0) or 0)
        local_delta = float(stain.get("local_dark_delta", 0) or 0)
        area_ratio = area / float(mask_area)
        return (
            angle_groups >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_LINE_COUNT
            and length >= 42.0
            and aspect_ratio >= 1.15
            and area_ratio <= 0.82
            and density >= 0.45
            and signed_delta >= -34.0
            and local_delta <= 10.0
        )

    def promote_stain_candidate_to_scratch(self, stain):
        promoted = dict(stain)
        promoted["type"] = "scratch"
        promoted["source"] = "pc_fast_dark_scratch"
        promoted["confidence"] = max(0.84, min(0.93, float(stain.get("confidence", 0.84) or 0.84)))
        promoted["strong_internal_scratch"] = True
        promoted["level"] = "medium" if self._positive_int(stain.get("length")) >= 65 else "light"
        return promoted

    def scratch_candidate_looks_like_dark_stain(self, scratch):
        if not isinstance(scratch, dict):
            return False
        if str(scratch.get("source", "")) != "pc_fast_review_line":
            return False
        signed_delta = float(scratch.get("local_signed_delta", 0) or 0)
        dark_delta = float(scratch.get("local_dark_delta", 0) or 0)
        bright_delta = float(scratch.get("local_bright_delta", 0) or 0)
        line_count = int(scratch.get("line_count", 0) or 0)
        fill_ratio = float(scratch.get("fill_ratio", 0) or 0)
        return (
            line_count >= FAST_REVIEW_STAR_SCRATCH_MIN_LINES
            and signed_delta <= FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA
            and dark_delta > bright_delta * 1.35
            and fill_ratio >= FAST_REVIEW_STAR_SCRATCH_MIN_FILL
        )

    def promote_scratch_candidate_to_stain(self, scratch):
        promoted = dict(scratch)
        promoted["type"] = "stain"
        promoted["source"] = "pc_fast_dark_star_stain"
        promoted["confidence"] = max(0.84, min(0.94, float(scratch.get("confidence", 0.84) or 0.84)))
        promoted["density"] = float(scratch.get("fill_ratio", 0) or 0)
        promoted["level"] = "medium" if self._positive_int(scratch.get("length")) >= 55 else "light"
        promoted.pop("strong_internal_scratch", None)
        return promoted

    def fast_cv_find_stain(self, gray, mask, mean_value, contrast_delta, edge_distance=None):
        dark_limit = max(0, int(mean_value - contrast_delta))
        dark_mask = np.where(gray <= dark_limit, 255, 0).astype(np.uint8)
        dark_mask = cv2.bitwise_and(dark_mask, mask)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

        contours, _hierarchy = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_area = max(1, int(cv2.countNonZero(mask)))
        min_area = max(1600.0, float(mask_area) * 0.010)
        max_area = float(mask_area) * FAST_REVIEW_STAIN_MAX_AREA_RATIO
        best = None
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            if edge_distance is not None:
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                min_edge_distance = max(
                    FAST_REVIEW_STAIN_MIN_EDGE_DISTANCE,
                    int(min(gray.shape[:2]) * FAST_REVIEW_STAIN_EDGE_MARGIN_RATIO),
                )
                if edge_distance[center_y, center_x] < min_edge_distance:
                    continue
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            if aspect_ratio > 5.2:
                continue
            mask_roi = mask[y:y + h, x:x + w]
            if mask_roi.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, w * h))
            if mask_overlap < FAST_REVIEW_STAIN_MIN_MASK_OVERLAP:
                continue
            rect_mean = float(np.mean(gray[y:y + h, x:x + w]))
            signed_delta = rect_mean - mean_value
            fill_ratio = area / float(max(1, w * h))
            min_delta = max(18.0, FAST_REVIEW_STAIN_MIN_DELTA * 0.75)
            if signed_delta > -min_delta or fill_ratio < FAST_REVIEW_STAIN_MIN_FILL_RATIO:
                continue
            if area > best_area:
                best_area = area
                best = {
                    "type": "stain",
                    "confidence": 0.86,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "length": int(max(w, h)),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if area >= min_area * 2 else "light",
                    "source": "pc_fast_review",
                }
        return best

    def fast_cv_find_dark_stain_cluster(self, gray, mask, mean_value, std_value, edge_distance=None):
        mask_values = gray[mask > 0]
        if mask_values.size <= 0:
            return None

        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
        local_dark = np.maximum(local_background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
        local_values = local_dark[mask > 0]
        local_threshold = max(
            FAST_REVIEW_DARK_CLUSTER_MIN_LOCAL_DELTA,
            float(np.percentile(local_values, 82.0)) if local_values.size else FAST_REVIEW_DARK_CLUSTER_MIN_LOCAL_DELTA,
        )
        dark_limit = min(
            int(mean_value - max(10.0, std_value * 0.45)),
            int(np.percentile(mask_values, 28.0)),
        )
        dark_mask = np.where(gray <= max(0, dark_limit), 255, 0).astype(np.uint8)
        local_mask = np.where(local_dark >= local_threshold, 255, 0).astype(np.uint8)
        dark_mask = cv2.bitwise_or(dark_mask, local_mask)
        dark_mask = cv2.bitwise_and(dark_mask, mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((1, 1), dtype=np.uint8))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))
        dark_mask = cv2.dilate(dark_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)

        contours, _hierarchy = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_area = max(1, int(cv2.countNonZero(mask)))
        min_area = max(160.0, float(mask_area) * FAST_REVIEW_DARK_CLUSTER_MIN_AREA_RATIO)
        best = None
        best_score = 0.0
        min_short_side = max(10, int((mask_area ** 0.5) * FAST_REVIEW_DARK_CLUSTER_MIN_SHORT_SIDE_RATIO))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            length = max(w, h)
            short_side = max(1, min(w, h))
            if short_side < min_short_side:
                continue
            aspect_ratio = float(length) / float(short_side)
            if aspect_ratio > FAST_REVIEW_DARK_CLUSTER_MAX_ASPECT:
                continue

            if edge_distance is not None:
                center_x = max(0, min(gray.shape[1] - 1, int(x + w / 2)))
                center_y = max(0, min(gray.shape[0] - 1, int(y + h / 2)))
                if edge_distance[center_y, center_x] < max(8, int(min(gray.shape[:2]) * 0.035)):
                    continue

            mask_roi = mask[y:y + h, x:x + w]
            if mask_roi.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, w * h))
            if mask_overlap < FAST_REVIEW_STAIN_MIN_MASK_OVERLAP:
                continue

            component_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(component_mask, [shifted], -1, 255, -1)
            fill_ratio = area / float(max(1, w * h))
            if fill_ratio < FAST_REVIEW_DARK_CLUSTER_MIN_BOX_FILL:
                continue
            area_ratio = area / float(mask_area)
            if (
                area_ratio > FAST_REVIEW_DARK_STAIN_MAX_AREA_RATIO
                and fill_ratio < FAST_REVIEW_DARK_STAIN_SOLID_MIN_FILL
            ):
                continue

            signed_delta = float(np.mean(gray[y:y + h, x:x + w][component_mask > 0])) - mean_value
            local_mean = float(np.mean(local_dark[y:y + h, x:x + w][component_mask > 0]))
            box_area_ratio = float(w * h) / float(mask_area)
            if (
                area_ratio > FAST_REVIEW_DARK_STAIN_MAX_WEAK_AREA_RATIO
                and box_area_ratio > FAST_REVIEW_DARK_STAIN_MAX_WEAK_BOX_RATIO
                and local_mean < FAST_REVIEW_DARK_STAIN_WEAK_LOCAL_DELTA
            ):
                continue
            if (
                abs(signed_delta) < FAST_REVIEW_DARK_CLUSTER_MIN_SIGNED_DELTA
                and local_mean < FAST_REVIEW_DARK_CLUSTER_WEAK_LOCAL_DELTA
            ):
                continue
            solid_dark_stain = (
                area_ratio >= FAST_REVIEW_DARK_STAIN_SOLID_MIN_AREA_RATIO
                and fill_ratio >= FAST_REVIEW_DARK_STAIN_SOLID_MIN_FILL
                and signed_delta <= -max(6.0, std_value * 0.18)
            )
            if not solid_dark_stain and signed_delta > -max(8.0, std_value * 0.25) and local_mean < local_threshold * 1.05:
                continue

            angle_groups = self.count_dark_cluster_angle_groups(gray, x, y, w, h, component_mask)
            if not solid_dark_stain and angle_groups < FAST_REVIEW_DARK_CLUSTER_MIN_ANGLE_GROUPS and fill_ratio < 0.035:
                continue

            score = area + angle_groups * 80.0 + abs(signed_delta) * 4.0 + local_mean * 5.0
            if score > best_score:
                best_score = score
                confidence = min(0.95, 0.84 + min(0.07, abs(signed_delta) / 160.0) + min(0.04, local_mean / 180.0))
                best = {
                    "type": "stain",
                    "confidence": round(confidence, 2),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "density": round(fill_ratio, 2),
                    "angle_groups": int(angle_groups),
                    "brightness_signed_delta": round(signed_delta, 1),
                    "local_dark_delta": round(local_mean, 1),
                    "level": "medium" if area >= min_area * 2.2 or length >= 70 else "light",
                    "source": "pc_fast_dark_stain",
                }

        return best

    def count_dark_cluster_angle_groups(self, gray, x, y, w, h, component_mask):
        roi = gray[y:y + h, x:x + w]
        if roi.size <= 0:
            return 0
        edges = cv2.Canny(roi, 25, 70)
        edges = cv2.bitwise_and(edges, component_mask)
        min_line_length = max(8, int(max(w, h) * 0.22))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=8,
            minLineLength=min_line_length,
            maxLineGap=7,
        )
        if lines is None:
            return 0

        groups = set()
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
            groups.add(int(angle // 30.0))
        return len(groups)

    def fast_cv_find_scratch(self, gray, mask, edge_distance=None, stain=None):
        if stain is not None:
            stain_area = int(stain.get("area", 0) or 0)
            stain_density = float(stain.get("density", 0) or 0)
            if stain_area >= 1600 and stain_density >= 0.22:
                return None

        edges = cv2.Canny(gray, 30, 60)
        edges = cv2.bitwise_and(edges, mask)
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(edges, 8)
        candidates = []
        height, width = gray.shape[:2]
        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        for index in range(1, component_count):
            x, y, w, h, area = stats[index]
            length = max(int(w), int(h))
            short_side = max(1, min(int(w), int(h)))
            aspect_ratio = float(length) / float(short_side)
            if int(area) < 28:
                continue
            if (
                short_side < FAST_REVIEW_NARROW_SCRATCH_MIN_WIDTH
                and (
                    aspect_ratio > FAST_REVIEW_NARROW_SCRATCH_MAX_ASPECT
                    or int(area) <= FAST_REVIEW_NARROW_SCRATCH_MAX_AREA
                )
            ):
                continue
            if length < 26 or short_side > 26 or aspect_ratio < 1.8:
                continue
            center_x = int(x) + int(w) / 2.0
            center_y = int(y) + int(h) / 2.0
            if center_x < width * 0.16 or center_x > width * 0.84:
                continue
            if center_y < height * 0.16 or center_y > height * 0.84:
                continue
            if edge_distance is not None:
                cx = max(0, min(width - 1, int(round(center_x))))
                cy = max(0, min(height - 1, int(round(center_y))))
                if edge_distance[cy, cx] < min_edge_distance:
                    continue
                inner = edge_distance[int(y):int(y + h), int(x):int(x + w)]
                if inner.size <= 0 or float(np.percentile(inner, 70)) < min_edge_distance * 0.65:
                    continue
            candidates.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "length": int(length),
                "aspect_ratio": aspect_ratio,
            })

        line_detection = self.fast_cv_find_scratch_lines(gray, mask, edge_distance)
        bright_line_detection = self.fast_cv_find_bright_scratch_lines(gray, mask, edge_distance)
        if bright_line_detection is not None:
            if line_detection is None or bright_line_detection.get("total_line_length", 0) > line_detection.get("total_line_length", 0):
                line_detection = bright_line_detection
        bright_crosshatch_detection = self.fast_cv_find_bright_crosshatch_scratch(gray, mask, edge_distance)
        if bright_crosshatch_detection is not None:
            if line_detection is None or bright_crosshatch_detection.get("total_line_length", 0) > line_detection.get("total_line_length", 0):
                line_detection = bright_crosshatch_detection
        center_cross_detection = self.fast_cv_find_center_cross_scratch(gray, mask, edge_distance)
        if center_cross_detection is not None:
            if (
                line_detection is None
                or center_cross_detection.get("total_line_length", 0) > line_detection.get("total_line_length", 0) * 0.72
            ):
                line_detection = center_cross_detection
        if line_detection is not None:
            candidates.extend(line_detection.pop("candidates", []))

        strong = [item for item in candidates if item["length"] >= 45 and item["aspect_ratio"] >= 2.4]
        medium = [item for item in candidates if item["length"] >= 20 and item["aspect_ratio"] >= 1.8]
        if not strong and len(medium) < 3:
            if line_detection is not None:
                return line_detection
            if center_cross_detection is not None:
                return center_cross_detection
            return None

        selected = strong if strong else medium
        left = min(item["x"] for item in selected)
        top = min(item["y"] for item in selected)
        right = max(item["x"] + item["w"] for item in selected)
        bottom = max(item["y"] + item["h"] for item in selected)
        total_area = sum(item["area"] for item in selected)
        length = max(right - left, bottom - top)
        short_side = max(1, min(right - left, bottom - top))
        aggregate_aspect = float(length) / float(short_side)
        if aggregate_aspect < FAST_REVIEW_SCRATCH_MIN_AGGREGATE_ASPECT:
            if line_detection is not None:
                return line_detection
            return None
        mask_roi = mask[int(top):int(bottom), int(left):int(right)]
        if mask_roi.size <= 0:
            return None
        mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, (right - left) * (bottom - top)))
        if mask_overlap < FAST_REVIEW_SCRATCH_MIN_MASK_OVERLAP:
            if line_detection is not None:
                return line_detection
            return None
        mask_area = max(1, int(cv2.countNonZero(mask)))
        box_area_ratio = float((right - left) * (bottom - top)) / float(mask_area)
        if box_area_ratio > FAST_REVIEW_SCRATCH_MAX_BOX_AREA_RATIO:
            if line_detection is not None:
                return line_detection
            return None
            if (
                total_area / float(mask_area) < FAST_REVIEW_WEAK_SCRATCH_MAX_AREA_RATIO
                and box_area_ratio < FAST_REVIEW_WEAK_SCRATCH_MAX_BOX_RATIO
                and length < 60
            ):
                if line_detection is not None:
                    return line_detection
                return None
        if total_area < 160 and length < 70:
            if line_detection is not None:
                return line_detection
            return None
        if total_area < 180 and aggregate_aspect < 2.2:
            if line_detection is not None:
                return line_detection
            return None
        confidence = min(0.94, 0.78 + len(selected) * 0.035 + min(0.08, length / 500.0))
        return {
            "type": "scratch",
            "confidence": round(confidence, 2),
            "x": int(left),
            "y": int(top),
            "w": int(right - left),
            "h": int(bottom - top),
            "area": int(total_area),
            "length": int(length),
            "aspect_ratio": round(aggregate_aspect, 2),
            "level": "medium" if length >= 70 or total_area >= 180 else "light",
            "source": "pc_fast_review",
            "strong_internal_scratch": (
                length >= 48
                and short_side >= 16
                and aggregate_aspect >= 1.8
                and box_area_ratio <= 0.08
            ),
        }

    def fast_cv_find_scratch_lines(self, gray, mask, edge_distance=None):
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
        local_diff = cv2.absdiff(gray, local_background)
        lens_values = local_diff[mask > 0]
        if lens_values.size <= 0:
            return None

        diff_threshold = max(8.0, float(np.percentile(lens_values, 93.0)))
        diff_mask = np.where(local_diff >= diff_threshold, 255, 0).astype(np.uint8)
        edge_mask = cv2.Canny(blurred, 25, 70)
        candidate_mask = cv2.bitwise_or(edge_mask, diff_mask)
        candidate_mask = cv2.bitwise_and(candidate_mask, mask)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

        height, width = gray.shape[:2]
        min_line_length = max(14, int(min(width, height) * 0.045))
        lines = cv2.HoughLinesP(
            candidate_mask,
            1,
            np.pi / 180.0,
            threshold=12,
            minLineLength=min_line_length,
            maxLineGap=8,
        )
        if lines is None:
            return None

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        segments = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            center_x = max(0, min(width - 1, int((x1 + x2) / 2)))
            center_y = max(0, min(height - 1, int((y1 + y2) / 2)))
            if mask[center_y, center_x] == 0:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance:
                continue
            if edge_distance is not None:
                line_points = []
                samples = max(2, int(length / 6.0))
                for sample_index in range(samples + 1):
                    ratio = float(sample_index) / float(samples)
                    px = max(0, min(width - 1, int(round(x1 + (x2 - x1) * ratio))))
                    py = max(0, min(height - 1, int(round(y1 + (y2 - y1) * ratio))))
                    line_points.append(float(edge_distance[py, px]))
                if line_points and float(np.percentile(line_points, 40)) < min_edge_distance * 0.75:
                    continue
                if line_points and float(np.percentile(line_points, 10)) < min_edge_distance * 0.35:
                    continue
            segments.append((x1, y1, x2, y2, length))

        if not segments:
            return None

        clusters = self.cluster_scratch_line_segments(segments)
        mask_area = max(1, int(cv2.countNonZero(mask)))
        frame_area = max(1, int(width * height))
        best = None
        best_score = 0.0
        for cluster in clusters:
            xs = []
            ys = []
            total_line_length = 0.0
            for x1, y1, x2, y2, line_length in cluster:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                total_line_length += line_length

            left = max(0, min(xs) - 2)
            top = max(0, min(ys) - 2)
            right = min(width - 1, max(xs) + 3)
            bottom = min(height - 1, max(ys) + 3)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            box_area = box_w * box_h
            box_area_ratio = float(box_area) / float(mask_area)
            if box_area_ratio > FAST_REVIEW_SCRATCH_MAX_BOX_AREA_RATIO:
                continue

            mask_roi = mask[top:bottom, left:right]
            if mask_roi.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, box_area))
            if mask_overlap < FAST_REVIEW_SCRATCH_MIN_MASK_OVERLAP:
                continue

            candidate_roi = candidate_mask[top:bottom, left:right]
            candidate_area = int(cv2.countNonZero(candidate_roi))
            length = max(box_w, box_h)
            short_side = max(1, min(box_w, box_h))
            aspect_ratio = float(length) / float(short_side)
            gray_roi = gray[top:bottom, left:right]
            background_roi = local_background[top:bottom, left:right]
            active_pixels = candidate_roi > 0
            if np.any(active_pixels):
                signed_values = gray_roi[active_pixels].astype(np.int16) - background_roi[active_pixels].astype(np.int16)
                local_signed_delta = float(np.mean(signed_values))
                local_bright_delta = float(np.percentile(np.maximum(signed_values, 0), 85.0))
                local_dark_delta = float(np.percentile(np.maximum(-signed_values, 0), 85.0))
            else:
                local_signed_delta = 0.0
                local_bright_delta = 0.0
                local_dark_delta = 0.0
            if total_line_length < max(42.0, min_line_length * 1.9):
                continue
            if len(cluster) == 1 and total_line_length < 78.0:
                continue
            if total_line_length < 120.0 and len(cluster) < 5:
                continue
            if aspect_ratio < FAST_REVIEW_SCRATCH_MIN_AGGREGATE_ASPECT and len(cluster) < 4:
                continue

            line_density = total_line_length / float(max(1, length))
            fill_ratio = float(candidate_area) / float(max(1, box_area))
            crosshatch_scratch = (
                len(cluster) >= 4
                and total_line_length >= FAST_REVIEW_SCRATCH_CROSSHATCH_MIN_LINE_LENGTH
                and box_area_ratio <= 0.075
                and 0.06 <= fill_ratio <= 0.54
                and local_bright_delta >= max(4.5, local_dark_delta * 0.65)
            )
            star_scratch = (
                len(cluster) >= FAST_REVIEW_STAR_SCRATCH_MIN_LINES
                and total_line_length >= FAST_REVIEW_STAR_SCRATCH_MIN_LENGTH
                and box_area_ratio <= FAST_REVIEW_SCRATCH_MAX_BOX_AREA_RATIO
                and FAST_REVIEW_STAR_SCRATCH_MIN_FILL <= fill_ratio <= FAST_REVIEW_STAR_SCRATCH_MAX_FILL
            )
            if fill_ratio > FAST_REVIEW_SCRATCH_MAX_FILL_RATIO and aspect_ratio < 2.3:
                if not star_scratch and not crosshatch_scratch:
                    continue
            if aspect_ratio < FAST_REVIEW_SCRATCH_MIN_LINE_ASPECT and len(cluster) < 6:
                if not star_scratch and not crosshatch_scratch:
                    continue
            if aspect_ratio < FAST_REVIEW_SCRATCH_MIN_AGGREGATE_ASPECT and not star_scratch and not crosshatch_scratch:
                continue
            strong_internal = (
                total_line_length >= 78.0
                and len(cluster) >= 4
                and short_side >= 16
                and box_area_ratio <= 0.08
                and 0.05 <= fill_ratio <= 0.45
            ) or star_scratch or crosshatch_scratch
            score = total_line_length + candidate_area * 0.35 + line_density * 12.0
            if score > best_score:
                best_score = score
                confidence = min(0.94, 0.78 + min(0.10, total_line_length / 800.0) + min(0.06, len(cluster) * 0.015))
                strong_internal = strong_internal and self.scratch_box_has_safe_edge_distance(
                    edge_distance,
                    left,
                    top,
                    right,
                    bottom,
                    min_edge_distance,
                    0.90,
                )
                best = {
                    "type": "scratch",
                    "confidence": round(confidence, 2),
                    "x": int(left),
                    "y": int(top),
                    "w": int(box_w),
                    "h": int(box_h),
                    "area": int(candidate_area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if length >= 70 or total_line_length >= 110 else "light",
                    "source": "pc_fast_review_line",
                    "line_count": len(cluster),
                    "total_line_length": round(total_line_length, 1),
                    "fill_ratio": round(fill_ratio, 2),
                    "local_signed_delta": round(local_signed_delta, 1),
                    "local_bright_delta": round(local_bright_delta, 1),
                    "local_dark_delta": round(local_dark_delta, 1),
                    "star_scratch": bool(star_scratch),
                    "crosshatch_scratch": bool(crosshatch_scratch),
                    "strong_internal_scratch": strong_internal,
                }

        return best

    def fast_cv_find_bright_crosshatch_scratch(self, gray, mask, edge_distance=None):
        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=11, sigmaY=11)
        bright_diff = np.maximum(gray.astype(np.int16) - local_background.astype(np.int16), 0).astype(np.uint8)
        lens_values = bright_diff[mask > 0]
        if lens_values.size <= 0:
            return None

        diff_threshold = max(
            FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_BRIGHT_DELTA,
            float(np.percentile(lens_values, 88.0)),
        )
        bright_mask = np.where(bright_diff >= diff_threshold, 255, 0).astype(np.uint8)
        bright_mask = cv2.bitwise_and(bright_mask, mask)
        gray_values = gray[mask > 0]
        if gray_values.size > 0:
            glare_limit = max(
                FAST_REVIEW_GLARE_MASK_MIN_GRAY,
                float(np.percentile(gray_values, FAST_REVIEW_GLARE_MASK_PERCENTILE)),
            )
            glare_mask = np.where(gray >= glare_limit, 255, 0).astype(np.uint8)
            glare_mask = cv2.dilate(glare_mask, np.ones((17, 17), dtype=np.uint8), iterations=1)
            bright_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(glare_mask))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

        height, width = gray.shape[:2]
        min_line_length = max(8, int(min(width, height) * 0.028))
        lines = cv2.HoughLinesP(
            bright_mask,
            1,
            np.pi / 180.0,
            threshold=7,
            minLineLength=min_line_length,
            maxLineGap=8,
        )
        if lines is None:
            return None

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        segments = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            center_x = max(0, min(width - 1, int((x1 + x2) / 2)))
            center_y = max(0, min(height - 1, int((y1 + y2) / 2)))
            if mask[center_y, center_x] == 0:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance * 0.78:
                continue
            segments.append((x1, y1, x2, y2, length))

        if len(segments) < FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_LINE_COUNT:
            return None

        clusters = self.cluster_scratch_line_segments(segments)
        mask_area = max(1, int(cv2.countNonZero(mask)))
        frame_area = max(1, int(width * height))
        best = None
        best_score = 0.0
        for cluster in clusters:
            if len(cluster) < FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_LINE_COUNT:
                continue
            xs = []
            ys = []
            total_line_length = 0.0
            angle_groups = set()
            slender_count = 0
            blob_like_count = 0
            slender_total_length = 0.0
            for x1, y1, x2, y2, line_length in cluster:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                total_line_length += line_length
                angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
                angle_groups.add(int(angle // 25.0))
                segment_short_side = max(1, min(abs(x2 - x1) + 1, abs(y2 - y1) + 1))
                segment_aspect = float(line_length) / float(segment_short_side)
                if (
                    segment_aspect >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_ASPECT
                    and line_length >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LENGTH
                ):
                    slender_count += 1
                    slender_total_length += line_length
                elif segment_aspect < 2.0:
                    blob_like_count += 1
            if total_line_length < FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_TOTAL_LENGTH:
                continue
            if len(angle_groups) < 2:
                continue
            if slender_count < FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_SLENDER_LINES:
                continue
            blob_ratio = float(blob_like_count) / float(max(1, len(cluster)))
            if blob_ratio > FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_BLOB_LINE_RATIO:
                continue

            left = max(0, min(xs) - 5)
            top = max(0, min(ys) - 5)
            right = min(width - 1, max(xs) + 6)
            bottom = min(height - 1, max(ys) + 6)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            box_area = box_w * box_h
            box_area_ratio = float(box_area) / float(mask_area)
            strong_multi_angle_crosshatch = (
                len(angle_groups) >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_STRONG_ANGLE_GROUPS
                and total_line_length >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_STRONG_TOTAL_LENGTH
                and slender_count >= FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_STRONG_SLENDER_LINES
                and box_area_ratio <= FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_STRONG_BOX_AREA_RATIO
            )
            if (
                box_area_ratio > FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_BOX_AREA_RATIO
                and not strong_multi_angle_crosshatch
            ):
                continue

            mask_roi = mask[top:bottom, left:right]
            if mask_roi.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, box_area))
            if mask_overlap < 0.34:
                continue

            candidate_roi = bright_mask[top:bottom, left:right]
            candidate_area = int(cv2.countNonZero(candidate_roi))
            if candidate_area <= 0:
                continue
            bright_roi = bright_diff[top:bottom, left:right]
            local_peak = float(np.percentile(bright_roi[candidate_roi > 0], 85.0))
            if local_peak < FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_BRIGHT_DELTA:
                continue

            length = max(box_w, box_h)
            short_side = max(1, min(box_w, box_h))
            aspect_ratio = float(length) / float(short_side)
            fill_ratio = float(candidate_area) / float(max(1, box_area))
            if (
                aspect_ratio < FAST_REVIEW_BRIGHT_CROSSHATCH_MIN_ASPECT
                and len(angle_groups) < 4
                and not strong_multi_angle_crosshatch
            ):
                continue
            if fill_ratio > FAST_REVIEW_BRIGHT_CROSSHATCH_MAX_FILL_RATIO:
                continue
            if edge_distance is not None and not self.scratch_box_has_safe_edge_distance(
                edge_distance,
                left,
                top,
                right,
                bottom,
                min_edge_distance,
                0.98,
            ):
                continue
            score = total_line_length + len(angle_groups) * 42.0 + local_peak * 5.0
            if score > best_score:
                best_score = score
                confidence = min(0.94, 0.82 + min(0.08, total_line_length / 700.0) + min(0.04, local_peak / 160.0))
                best = {
                    "type": "scratch",
                    "confidence": round(confidence, 2),
                    "x": int(left),
                    "y": int(top),
                    "w": int(box_w),
                    "h": int(box_h),
                    "area": int(candidate_area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if total_line_length >= 120 or length >= 70 else "light",
                    "source": "pc_fast_bright_crosshatch",
                    "line_count": len(cluster),
                    "angle_groups": len(angle_groups),
                    "slender_line_count": int(slender_count),
                    "blob_line_ratio": round(blob_ratio, 2),
                    "slender_total_length": round(slender_total_length, 1),
                    "total_line_length": round(total_line_length, 1),
                    "local_bright_delta": round(local_peak, 1),
                    "local_signed_delta": round(local_peak, 1),
                    "local_dark_delta": 0.0,
                    "fill_ratio": round(fill_ratio, 2),
                    "crosshatch_scratch": True,
                    "strong_internal_scratch": True,
                    "candidates": [
                        {
                            "x": int(min(x1, x2)),
                            "y": int(min(y1, y2)),
                            "w": int(abs(x2 - x1) + 1),
                            "h": int(abs(y2 - y1) + 1),
                            "area": int(max(1, line_length)),
                            "length": int(line_length),
                            "aspect_ratio": round(float(line_length) / float(max(1, min(abs(x2 - x1) + 1, abs(y2 - y1) + 1))), 2),
                        }
                        for x1, y1, x2, y2, line_length in cluster
                    ],
                }

        return best

    def fast_cv_find_center_cross_scratch(self, gray, mask, edge_distance=None):
        height, width = gray.shape[:2]
        if width < 560 or height < 360:
            return None
        frame_area = max(1, int(width * height))
        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
        bright_diff = np.maximum(gray.astype(np.int16) - local_background.astype(np.int16), 0).astype(np.uint8)
        dark_diff = np.maximum(local_background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)

        left_limit = int(width * FAST_REVIEW_CENTER_CROSS_MIN_X_RATIO)
        right_limit = int(width * FAST_REVIEW_CENTER_CROSS_MAX_X_RATIO)
        top_limit = int(height * FAST_REVIEW_CENTER_CROSS_MIN_Y_RATIO)
        bottom_limit = int(height * FAST_REVIEW_CENTER_CROSS_MAX_Y_RATIO)
        if right_limit <= left_limit or bottom_limit <= top_limit:
            return None

        center_roi = np.zeros_like(mask)
        center_roi[top_limit:bottom_limit, left_limit:right_limit] = 255
        center_roi = cv2.bitwise_and(center_roi, mask)
        if cv2.countNonZero(center_roi) <= 0:
            return None

        roi_values = np.concatenate((
            bright_diff[center_roi > 0].reshape(-1),
            dark_diff[center_roi > 0].reshape(-1),
        ))
        if roi_values.size <= 0:
            return None
        diff_threshold = max(
            FAST_REVIEW_CENTER_CROSS_MIN_LOCAL_DELTA,
            float(np.percentile(roi_values, 88.5)),
        )
        contrast_mask = np.where(
            (bright_diff >= diff_threshold) | (dark_diff >= diff_threshold),
            255,
            0,
        ).astype(np.uint8)
        contrast_mask = cv2.bitwise_and(contrast_mask, center_roi)

        gray_values = gray[mask > 0]
        if gray_values.size > 0:
            glare_limit = max(FAST_REVIEW_GLARE_MASK_MIN_GRAY, float(np.percentile(gray_values, 99.0)))
            glare_mask = np.where(gray >= glare_limit, 255, 0).astype(np.uint8)
            glare_mask = cv2.dilate(glare_mask, np.ones((13, 13), dtype=np.uint8), iterations=1)
            contrast_mask = cv2.bitwise_and(contrast_mask, cv2.bitwise_not(glare_mask))

        edge_mask = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 18, 58)
        edge_mask = cv2.bitwise_and(edge_mask, center_roi)
        candidate_mask = cv2.bitwise_or(contrast_mask, edge_mask)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

        min_line_length = max(9, int(min(width, height) * 0.030))
        lines = cv2.HoughLinesP(
            candidate_mask,
            1,
            np.pi / 180.0,
            threshold=8,
            minLineLength=min_line_length,
            maxLineGap=9,
        )
        if lines is None:
            return None

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        segments = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            center_x = max(0, min(width - 1, int((x1 + x2) / 2)))
            center_y = max(0, min(height - 1, int((y1 + y2) / 2)))
            if center_roi[center_y, center_x] == 0:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance * 0.62:
                continue
            angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
            if (
                center_x > width * FAST_REVIEW_FRAME_SIDE_REJECT_RATIO
                and 65.0 <= angle <= 115.0
                and length >= min_line_length * 1.8
            ):
                continue
            short_side = max(1, min(abs(x2 - x1) + 1, abs(y2 - y1) + 1))
            aspect = length / float(short_side)
            if aspect < 1.8 and length < 22.0:
                continue
            segments.append((x1, y1, x2, y2, length))

        if len(segments) < FAST_REVIEW_CENTER_CROSS_MIN_LINE_COUNT:
            return None

        clusters = self.cluster_scratch_line_segments(segments)
        mask_area = max(1, int(cv2.countNonZero(mask)))
        best = None
        best_score = 0.0
        for cluster in clusters:
            if len(cluster) < FAST_REVIEW_CENTER_CROSS_MIN_LINE_COUNT:
                continue
            xs = []
            ys = []
            total_line_length = 0.0
            angle_groups = set()
            slender_count = 0
            vertical_length = 0.0
            for x1, y1, x2, y2, line_length in cluster:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                total_line_length += line_length
                angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
                angle_groups.add(int(angle // 25.0))
                if 65.0 <= angle <= 115.0:
                    vertical_length += line_length
                short_side = max(1, min(abs(x2 - x1) + 1, abs(y2 - y1) + 1))
                segment_aspect = float(line_length) / float(short_side)
                if segment_aspect >= 2.6 and line_length >= 14.0:
                    slender_count += 1

            if total_line_length < FAST_REVIEW_CENTER_CROSS_MIN_TOTAL_LENGTH:
                continue
            if len(angle_groups) < FAST_REVIEW_CENTER_CROSS_MIN_ANGLE_GROUPS:
                continue
            if slender_count < FAST_REVIEW_CENTER_CROSS_MIN_SLENDER_LINES:
                continue
            vertical_ratio = vertical_length / float(max(1.0, total_line_length))
            if vertical_ratio > FAST_REVIEW_CENTER_CROSS_MAX_VERTICAL_RATIO:
                continue

            left = max(0, min(xs) - 4)
            top = max(0, min(ys) - 4)
            right = min(width - 1, max(xs) + 5)
            bottom = min(height - 1, max(ys) + 5)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            center_x = left + box_w / 2.0
            center_y = top + box_h / 2.0
            if center_x < left_limit or center_x > right_limit or center_y < top_limit or center_y > bottom_limit:
                continue
            if (
                center_x < width * FAST_REVIEW_CENTER_CROSS_BOX_MIN_X_RATIO
                or center_x > width * FAST_REVIEW_CENTER_CROSS_BOX_MAX_X_RATIO
            ):
                continue

            box_area = box_w * box_h
            box_area_ratio = float(box_area) / float(frame_area)
            if box_area_ratio > FAST_REVIEW_CENTER_CROSS_MAX_BOX_AREA_RATIO:
                continue
            mask_roi = mask[top:bottom, left:right]
            if mask_roi.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, box_area))
            if mask_overlap < 0.36:
                continue
            candidate_roi = candidate_mask[top:bottom, left:right]
            candidate_area = int(cv2.countNonZero(candidate_roi))
            if candidate_area <= 0:
                continue
            fill_ratio = float(candidate_area) / float(max(1, box_area))
            if fill_ratio > FAST_REVIEW_CENTER_CROSS_MAX_FILL_RATIO:
                continue
            contrast_roi = np.maximum(bright_diff[top:bottom, left:right], dark_diff[top:bottom, left:right])
            local_peak = float(np.percentile(contrast_roi[candidate_roi > 0], 82.0))
            if local_peak < FAST_REVIEW_CENTER_CROSS_MIN_LOCAL_DELTA:
                continue
            if edge_distance is not None and not self.scratch_box_has_safe_edge_distance(
                edge_distance,
                left,
                top,
                right,
                bottom,
                min_edge_distance,
                0.54,
            ):
                continue

            length = max(box_w, box_h)
            aspect_ratio = float(length) / float(max(1, min(box_w, box_h)))
            score = total_line_length + len(angle_groups) * 45.0 + slender_count * 18.0 + local_peak * 5.0
            if score > best_score:
                best_score = score
                confidence = min(0.94, 0.83 + min(0.08, total_line_length / 900.0) + min(0.03, local_peak / 180.0))
                best = {
                    "type": "scratch",
                    "confidence": round(confidence, 2),
                    "x": int(left),
                    "y": int(top),
                    "w": int(box_w),
                    "h": int(box_h),
                    "area": int(candidate_area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if total_line_length >= 220 or length >= 70 else "light",
                    "source": "pc_fast_center_cross_scratch",
                    "line_count": len(cluster),
                    "angle_groups": len(angle_groups),
                    "slender_line_count": int(slender_count),
                    "total_line_length": round(total_line_length, 1),
                    "vertical_line_ratio": round(vertical_ratio, 2),
                    "local_bright_delta": round(local_peak, 1),
                    "local_signed_delta": round(local_peak, 1),
                    "local_dark_delta": 0.0,
                    "fill_ratio": round(fill_ratio, 2),
                    "crosshatch_scratch": True,
                    "strong_internal_scratch": True,
                    "candidates": [
                        {
                            "x": int(min(x1, x2)),
                            "y": int(min(y1, y2)),
                            "w": int(abs(x2 - x1) + 1),
                            "h": int(abs(y2 - y1) + 1),
                            "area": int(max(1, line_length)),
                            "length": int(line_length),
                            "aspect_ratio": round(float(line_length) / float(max(1, min(abs(x2 - x1) + 1, abs(y2 - y1) + 1))), 2),
                        }
                        for x1, y1, x2, y2, line_length in cluster
                    ],
                }

        return best

    def fast_cv_find_bright_scratch_lines(self, gray, mask, edge_distance=None):
        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=11, sigmaY=11)
        bright_diff = np.maximum(gray.astype(np.int16) - local_background.astype(np.int16), 0).astype(np.uint8)
        lens_values = bright_diff[mask > 0]
        if lens_values.size <= 0:
            return None

        diff_threshold = max(
            FAST_REVIEW_BRIGHT_SCRATCH_MIN_LOCAL_DELTA,
            float(np.percentile(lens_values, 90.0)),
        )
        bright_mask = np.where(bright_diff >= diff_threshold, 255, 0).astype(np.uint8)
        edge_mask = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 18, 58)
        candidate_mask = cv2.bitwise_or(bright_mask, edge_mask)
        candidate_mask = cv2.bitwise_and(candidate_mask, mask)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

        height, width = gray.shape[:2]
        min_line_length = max(9, int(min(width, height) * 0.032))
        lines = cv2.HoughLinesP(
            candidate_mask,
            1,
            np.pi / 180.0,
            threshold=8,
            minLineLength=min_line_length,
            maxLineGap=10,
        )
        if lines is None:
            return None

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        segments = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            center_x = max(0, min(width - 1, int((x1 + x2) / 2)))
            center_y = max(0, min(height - 1, int((y1 + y2) / 2)))
            if mask[center_y, center_x] == 0:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance:
                continue
            if edge_distance is not None:
                samples = max(2, int(length / 7.0))
                distances = []
                for sample_index in range(samples + 1):
                    ratio = float(sample_index) / float(samples)
                    px = max(0, min(width - 1, int(round(x1 + (x2 - x1) * ratio))))
                    py = max(0, min(height - 1, int(round(y1 + (y2 - y1) * ratio))))
                    distances.append(float(edge_distance[py, px]))
                if distances and float(np.percentile(distances, 45)) < min_edge_distance * 0.72:
                    continue
                if distances and float(np.percentile(distances, 10)) < min_edge_distance * 0.36:
                    continue
            segments.append((x1, y1, x2, y2, length))

        if len(segments) < FAST_REVIEW_BRIGHT_SCRATCH_MIN_LINE_COUNT:
            return None

        clusters = self.cluster_scratch_line_segments(segments)
        mask_area = max(1, int(cv2.countNonZero(mask)))
        best = None
        best_score = 0.0
        for cluster in clusters:
            if len(cluster) < FAST_REVIEW_BRIGHT_SCRATCH_MIN_LINE_COUNT:
                continue
            xs = []
            ys = []
            total_line_length = 0.0
            angle_groups = set()
            for x1, y1, x2, y2, line_length in cluster:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                total_line_length += line_length
                angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
                angle_groups.add(int(angle // 30.0))
            if total_line_length < FAST_REVIEW_BRIGHT_SCRATCH_MIN_TOTAL_LENGTH:
                continue
            if len(cluster) < 4 and total_line_length < 78.0:
                continue
            if total_line_length < 130.0 and len(cluster) < 6:
                continue

            left = max(0, min(xs) - 3)
            top = max(0, min(ys) - 3)
            right = min(width - 1, max(xs) + 4)
            bottom = min(height - 1, max(ys) + 4)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            box_area = box_w * box_h
            box_area_ratio = float(box_area) / float(mask_area)
            if box_area_ratio > FAST_REVIEW_BRIGHT_SCRATCH_MAX_BOX_AREA_RATIO:
                continue

            mask_roi = mask[top:bottom, left:right]
            if mask_roi.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, box_area))
            if mask_overlap < FAST_REVIEW_SCRATCH_MIN_MASK_OVERLAP:
                continue

            candidate_roi = candidate_mask[top:bottom, left:right]
            candidate_area = int(cv2.countNonZero(candidate_roi))
            if candidate_area <= 0:
                continue
            bright_roi = bright_diff[top:bottom, left:right]
            local_peak = float(np.percentile(bright_roi[candidate_roi > 0], 85.0))
            if local_peak < diff_threshold:
                continue

            length = max(box_w, box_h)
            short_side = max(1, min(box_w, box_h))
            aspect_ratio = float(length) / float(short_side)
            if length < 55 and total_line_length < 220.0:
                continue
            fill_ratio = float(candidate_area) / float(max(1, box_area))
            continuous_line = aspect_ratio >= 2.2 and total_line_length >= FAST_REVIEW_BRIGHT_SCRATCH_STRONG_MIN_LENGTH
            bright_enough = local_peak >= FAST_REVIEW_BRIGHT_SCRATCH_STRONG_MIN_DELTA
            if aspect_ratio < FAST_REVIEW_SCRATCH_MIN_BRIGHT_LINE_ASPECT and len(angle_groups) < FAST_REVIEW_BRIGHT_SCRATCH_MIN_ANGLE_GROUPS:
                continue
            if fill_ratio > FAST_REVIEW_BRIGHT_SCRATCH_MAX_FILL_RATIO:
                if not (continuous_line and aspect_ratio >= 2.8 and bright_enough):
                    continue
            if fill_ratio > FAST_REVIEW_SCRATCH_MAX_FILL_RATIO and aspect_ratio < 2.2:
                continue
            if not continuous_line and not (bright_enough and len(angle_groups) >= 2 and total_line_length >= 72.0):
                continue
            if edge_distance is not None and not self.scratch_box_has_safe_edge_distance(
                edge_distance,
                left,
                top,
                right,
                bottom,
                min_edge_distance,
                0.95,
            ):
                continue

            score = total_line_length + len(angle_groups) * 35.0 + local_peak * 4.0
            if score > best_score:
                best_score = score
                confidence = min(0.94, 0.80 + min(0.08, total_line_length / 600.0) + min(0.04, local_peak / 160.0))
                strong_internal = self.scratch_box_has_safe_edge_distance(
                    edge_distance,
                    left,
                    top,
                    right,
                    bottom,
                    min_edge_distance,
                    0.95,
                )
                strong_internal = strong_internal and (continuous_line or bright_enough)
                best = {
                    "type": "scratch",
                    "confidence": round(confidence, 2),
                    "x": int(left),
                    "y": int(top),
                    "w": int(box_w),
                    "h": int(box_h),
                    "area": int(candidate_area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if total_line_length >= 95 or length >= 65 else "light",
                    "source": "pc_fast_bright_scratch",
                    "line_count": len(cluster),
                    "angle_groups": len(angle_groups),
                    "total_line_length": round(total_line_length, 1),
                    "local_bright_delta": round(local_peak, 1),
                    "fill_ratio": round(fill_ratio, 2),
                    "strong_internal_scratch": strong_internal,
                    "candidates": [
                        {
                            "x": int(min(x1, x2)),
                            "y": int(min(y1, y2)),
                            "w": int(abs(x2 - x1) + 1),
                            "h": int(abs(y2 - y1) + 1),
                            "area": int(max(1, line_length)),
                            "length": int(line_length),
                            "aspect_ratio": round(float(line_length) / float(max(1, min(abs(x2 - x1) + 1, abs(y2 - y1) + 1))), 2),
                        }
                        for x1, y1, x2, y2, line_length in cluster
                    ],
                }

        return best

    def scratch_box_has_safe_edge_distance(self, edge_distance, left, top, right, bottom, min_edge_distance, ratio):
        if edge_distance is None:
            return True
        inner = edge_distance[int(top):int(bottom), int(left):int(right)]
        if inner.size <= 0:
            return False
        return float(np.percentile(inner, 55)) >= float(min_edge_distance) * float(ratio)

    def cluster_scratch_line_segments(self, segments):
        clusters = []
        for segment in sorted(segments, key=lambda item: item[4], reverse=True):
            sx1, sy1, sx2, sy2, _length = segment
            segment_center = ((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0)
            best_cluster = None
            best_distance = None
            for cluster in clusters:
                xs = []
                ys = []
                for x1, y1, x2, y2, _line_length in cluster:
                    xs.extend([x1, x2])
                    ys.extend([y1, y2])
                center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
                distance = max(abs(segment_center[0] - center[0]), abs(segment_center[1] - center[1]))
                if distance <= 48 and (best_distance is None or distance < best_distance):
                    best_cluster = cluster
                    best_distance = distance
            if best_cluster is None:
                clusters.append([segment])
            else:
                best_cluster.append(segment)
        return clusters

    def apply_stage2_to_live_image(self, image, payload):
        if not isinstance(self.latest_detection_result, dict):
            return
        if not self.stage2_enabled_var.get():
            return
        now = time.monotonic()
        if now - self.last_stage2_infer_time < STAGE2_MIN_INTERVAL_SECONDS:
            return

        payload_key = "%s:%s" % (payload.get("receive_time", ""), payload.get("byte_count", ""))
        if self.latest_detection_result.get("_stage2_payload_key") == payload_key:
            return

        model = self.get_stage2_model()
        if model is None or cv2 is None or np is None:
            return

        analysis_mask = self.build_stage2_analysis_mask(image, payload, self.latest_detection_result)
        if analysis_mask is None:
            return

        self.last_stage2_infer_time = now
        rgb = np.array(image)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        stage2_result = model.infer(frame, analysis_mask)
        merged_defects, stage2_result = merge_with_rule_defects(
            self.latest_detection_result.get("defects") or [],
            stage2_result,
        )

        refined = dict(self.latest_detection_result)
        refined["defects"] = merged_defects
        refined["_stage2_payload_key"] = payload_key
        refined["stage2"] = {
            "enabled": True,
            "available": bool(stage2_result.get("available", False)),
            "model_path": stage2_result.get("model_path", ""),
            "sample_count": int(stage2_result.get("sample_count", 0)),
            "threshold": float(stage2_result.get("threshold", 0.0)),
            "score_max": float(stage2_result.get("score_max", 0.0)),
            "score_mean": float(stage2_result.get("score_mean", 0.0)),
            "candidate_count": int(stage2_result.get("candidate_count", 0)),
            "confirmed_rule_count": int(stage2_result.get("confirmed_rule_count", 0)),
            "added_stage2_count": int(stage2_result.get("added_stage2_count", 0)),
        }
        refined = self.normalize_detection_result(refined)
        refined = self.stabilize_detection_result(refined)
        self.update_result_ui(refined, render_live_image=False)

    def build_detection_mask(self, image, payload, result):
        if cv2 is None or np is None:
            return None

        image_w, image_h = image.size
        result = result if isinstance(result, dict) else {}
        frame = result.get("frame") or {}
        source_w = self._positive_int(frame.get("w")) or self._positive_int(payload.get("width")) or image_w
        source_h = self._positive_int(frame.get("h")) or self._positive_int(payload.get("height")) or image_h

        roi = result.get("roi") if isinstance(result.get("roi"), dict) else None
        lens = result.get("lens") if isinstance(result.get("lens"), dict) else None
        if lens is not None and lens.get("found", True):
            roi = lens
        roi = self.valid_roi_or_none(roi, source_w, source_h)
        if roi is not None and self.roi_is_low_confidence(roi, source_w, source_h):
            roi = None
        used_auto_roi = False
        if roi is None or self.roi_is_full_frame(roi, source_w, source_h):
            auto_roi = self.detect_lens_roi_from_image(image)
            if auto_roi is not None:
                roi = {
                    "x": int(auto_roi["x"] * source_w / float(max(1, image_w))),
                    "y": int(auto_roi["y"] * source_h / float(max(1, image_h))),
                    "w": int(auto_roi["w"] * source_w / float(max(1, image_w))),
                    "h": int(auto_roi["h"] * source_h / float(max(1, image_h))),
                    "source": "pc_auto_roi",
                }
                used_auto_roi = True
            else:
                roi = self.center_fallback_roi(source_w, source_h)
        self._last_effective_roi = dict(roi)

        x_scale = float(image_w) / float(max(1, source_w))
        y_scale = float(image_h) / float(max(1, source_h))
        x = int(self._positive_int(roi.get("x")) * x_scale)
        y = int(self._positive_int(roi.get("y")) * y_scale)
        w = int(self._positive_int(roi.get("w")) * x_scale)
        h = int(self._positive_int(roi.get("h")) * y_scale)
        if w <= 4 or h <= 4:
            return None

        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))

        mask = self.build_lens_ellipse_mask(image, (x, y, w, h)) if used_auto_roi else None
        if mask is None and used_auto_roi:
            mask = self.build_lens_contour_mask(image, (x, y, w, h))
        if mask is None:
            mask = np.zeros((image_h, image_w), dtype=np.uint8)
            center = (x + w // 2, y + h // 2)
            axes = (max(2, int(w * 0.44)), max(2, int(h * 0.40)))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        edge_ignore = max(8, int(min(w, h) * FAST_REVIEW_INNER_MASK_MARGIN_RATIO))
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        return np.where(distance > edge_ignore, 255, 0).astype(np.uint8)

    def build_stage2_analysis_mask(self, image, payload, result):
        return self.build_detection_mask(image, payload, result)

    def detect_lens_roi_from_image(self, image):
        if cv2 is None or np is None:
            return None

        image_w, image_h = image.size
        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 90)
        edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=1)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = float(max(1, image_w * image_h))
        min_area = frame_area * FAST_REVIEW_AUTO_ROI_MIN_AREA_RATIO
        min_w = max(24, int(image_w * FAST_REVIEW_AUTO_ROI_MIN_WIDTH_RATIO))
        min_h = max(18, int(image_h * FAST_REVIEW_AUTO_ROI_MIN_HEIGHT_RATIO))
        best_box = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or h < min_h:
                continue
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            if aspect_ratio > 3.6:
                continue
            fill_ratio = area / float(max(1, w * h))
            if fill_ratio < 0.04:
                continue
            center_x = (x + w / 2.0) / float(max(1, image_w))
            center_y = (y + h / 2.0) / float(max(1, image_h))
            center_penalty = abs(center_x - 0.50) * 0.40 + abs(center_y - 0.52) * 0.25
            bottom_bonus = min(0.18, max(0.0, (y + h) / float(max(1, image_h)) - 0.55))
            score = (area / frame_area) + fill_ratio * 0.18 + bottom_bonus - center_penalty
            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        if best_box is None:
            return None

        x, y, w, h = best_box
        padding = max(6, int(min(w, h) * FAST_REVIEW_AUTO_ROI_PADDING_RATIO))
        x = max(0, x - padding)
        y = max(0, y - padding)
        right = min(image_w, x + w + padding * 2)
        bottom = min(image_h, y + h + padding * 2)
        w = max(1, right - x)
        h = max(1, bottom - y)
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    def build_lens_ellipse_mask(self, image, roi_rect):
        if cv2 is None or np is None:
            return None

        image_w, image_h = image.size
        x, y, w, h = roi_rect
        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        crop = gray[y:y + h, x:x + w]
        if crop.size <= 0:
            return None

        blur = cv2.GaussianBlur(crop, (5, 5), 0)
        edges = cv2.Canny(blur, 35, 95)
        edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ellipse = None
        best_score = 0.0
        roi_area = float(max(1, w * h))
        for contour in contours:
            if len(contour) < 5:
                continue
            contour_area = float(cv2.contourArea(contour))
            if contour_area < roi_area * 0.01:
                continue
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (axis_a, axis_b), angle = ellipse
            long_axis = max(axis_a, axis_b)
            short_axis = max(1.0, min(axis_a, axis_b))
            aspect = long_axis / short_axis
            if aspect > 3.4:
                continue
            ellipse_area = np.pi * axis_a * axis_b / 4.0
            area_ratio = ellipse_area / roi_area
            if area_ratio < 0.04 or area_ratio > 0.80:
                continue
            center_dx = abs(cx - w / 2.0) / float(max(1.0, w / 2.0))
            center_dy = abs(cy - h / 2.0) / float(max(1.0, h / 2.0))
            center_score = 1.0 - min(1.0, (center_dx + center_dy) / 2.0)
            fill_score = min(1.0, contour_area / max(1.0, ellipse_area))
            aspect_score = 1.0 - min(1.0, abs(aspect - 1.55) / 2.2)
            score = area_ratio * 0.45 + center_score * 0.25 + fill_score * 0.18 + aspect_score * 0.12
            if score > best_score:
                best_score = score
                best_ellipse = ((cx + x, cy + y), (axis_a, axis_b), angle)

        if best_ellipse is None:
            return None

        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.ellipse(mask, best_ellipse, 255, -1)
        return mask

    def lens_mask_is_reliable(self, mask, roi_rect):
        if mask is None:
            return False
        x, y, w, h = roi_rect
        roi_mask = mask[y:y + h, x:x + w]
        if roi_mask.size <= 0:
            return False
        points = cv2.findNonZero(roi_mask)
        if points is None:
            return False
        bx, by, bw, bh = cv2.boundingRect(points)
        area_ratio = float(cv2.countNonZero(roi_mask)) / float(max(1, w * h))
        center_dx = abs((bx + bw / 2.0) - (w / 2.0)) / float(max(1, w))
        center_dy = abs((by + bh / 2.0) - (h / 2.0)) / float(max(1, h))
        return (
            bw >= int(w * 0.45)
            and bh >= int(h * 0.45)
            and 0.18 <= area_ratio <= 0.82
            and center_dx <= 0.28
            and center_dy <= 0.28
        )

    def build_lens_contour_mask(self, image, roi_rect):
        if cv2 is None or np is None:
            return None

        image_w, image_h = image.size
        x, y, w, h = roi_rect
        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        crop = gray[y:y + h, x:x + w]
        if crop.size <= 0:
            return None

        blur = cv2.GaussianBlur(crop, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 90)
        edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))

        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        min_w = max(24, int(w * FAST_REVIEW_LENS_CONTOUR_MIN_WIDTH_RATIO))
        min_h = max(18, int(h * FAST_REVIEW_LENS_CONTOUR_MIN_HEIGHT_RATIO))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < min_w or bh < min_h:
                continue
            center_y_ratio = float(by + bh / 2.0) / float(max(1, h))
            if center_y_ratio < FAST_REVIEW_LENS_CONTOUR_MIN_CENTER_Y_RATIO:
                continue
            aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
            if aspect_ratio > 3.5:
                continue
            rect_area = float(max(1, bw * bh))
            fill = area / rect_area
            if fill < 0.08:
                continue
            score = area * (1.0 + center_y_ratio * 0.35)
            if score > best_score:
                best_score = score
                best = contour

        if best is None:
            return None

        hull = cv2.convexHull(best)
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(local_mask, [hull], -1, 255, -1)
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

        full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        full_mask[y:y + h, x:x + w] = local_mask
        return full_mask

    def draw_detection_boxes(self, image, payload):
        if ImageDraw is None or not isinstance(self.latest_detection_result, dict):
            return

        defects = self.latest_detection_result.get("defects") or []
        if not defects:
            return

        frame = self.latest_detection_result.get("frame") or {}
        source_w = self._positive_int(frame.get("w")) or self._positive_int(payload.get("width")) or image.width
        source_h = self._positive_int(frame.get("h")) or self._positive_int(payload.get("height")) or image.height
        x_scale = float(image.width) / float(source_w)
        y_scale = float(image.height) / float(source_h)
        draw = ImageDraw.Draw(image)
        outline_width = max(2, int(min(image.width, image.height) / 160))

        for defect in defects:
            if defect.get("type") not in SUPPORTED_DEFECT_TYPES:
                continue
            x = self._positive_int(defect.get("x"))
            y = self._positive_int(defect.get("y"))
            w = self._positive_int(defect.get("w"))
            h = self._positive_int(defect.get("h"))
            if w <= 0 or h <= 0:
                continue

            left = x * x_scale
            top = y * y_scale
            right = (x + w) * x_scale
            bottom = (y + h) * y_scale
            padding = max(4, min(image.width, image.height) * 0.01)
            box = (
                max(0, left - padding),
                max(0, top - padding),
                min(image.width - 1, right + padding),
                min(image.height - 1, bottom + padding),
            )
            color = (255, 64, 64) if defect.get("type") == "scratch" else (255, 176, 0)
            draw.rectangle(box, outline=color, width=outline_width)
            try:
                confidence = float(defect.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            label = "%s %.2f" % (defect_type_name(defect.get("type", "")), confidence)
            text_x = int(box[0])
            text_y = max(0, int(box[1]) - 18)
            text_box = (text_x, text_y, text_x + max(58, len(label) * 12), text_y + 17)
            draw.rectangle(text_box, fill=color)
            draw.text((text_x + 3, text_y + 1), label, fill=(255, 255, 255))

    def _positive_int(self, value):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def format_lens_track_text(self, lens):
        if not isinstance(lens, dict):
            return "镜片跟踪：暂无数据"

        found = bool(lens.get("found", False))
        status = "已锁定" if found else "未找到"
        source = lens.get("source", "")
        confidence = lens.get("confidence", 0)
        try:
            confidence_text = "%.2f" % float(confidence)
        except (TypeError, ValueError):
            confidence_text = str(confidence)

        return (
            "镜片跟踪：%s，中心 (%s, %s)，ROI %sx%s+%s+%s，置信度 %s，%s"
            % (
                status,
                lens.get("cx", "-"),
                lens.get("cy", "-"),
                lens.get("w", "-"),
                lens.get("h", "-"),
                lens.get("x", "-"),
                lens.get("y", "-"),
                confidence_text,
                source,
            )
        )

    def add_history(self, result, raw_line):
        if self.history_tree is None:
            return
        record = {
            "receive_time": now_text(),
            "result": result,
            "raw_json": raw_line,
        }
        self.history_records.insert(0, record)
        self.history_records = self.history_records[:300]
        self._append_history_file(record)
        self._insert_history_row(record)

    def _insert_history_row(self, record):
        if self.history_tree is None:
            return
        result = record["result"]
        status = "有缺陷" if result.get("has_defect") else "正常"
        values = (
            record["receive_time"],
            status,
            result.get("defect_count", 0),
            level_name(result.get("overall_level", "normal")),
            result.get("timestamp", ""),
        )
        self.history_tree.insert("", 0, values=values)

    def load_history(self):
        if self.history_tree is None:
            return
        if not HISTORY_JSONL.exists():
            return

        try:
            with open(HISTORY_JSONL, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.history_records.insert(0, record)
        except Exception:
            self.history_records.clear()

        self.history_records = self.history_records[:300]
        for record in reversed(self.history_records):
            self._insert_history_row(record)

    def _append_history_file(self, record):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_JSONL, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear_history(self):
        if not messagebox.askyesno("确认", "确定清空界面和历史记录吗？"):
            return
        self.history_records.clear()
        if self.history_tree is not None:
            self.history_tree.delete(*self.history_tree.get_children())
        if self.defect_tree is not None:
            self.defect_tree.delete(*self.defect_tree.get_children())
        self._reset_summary()
        self.has_defect_var.set("是否检测到缺陷：暂无数据")
        self.count_var.set("缺陷总数：0")
        self.level_var.set("整体严重程度：暂无数据")
        self.lens_track_var.set("镜片跟踪：暂无数据")
        self.latest_detection_result = None
        self.last_fast_review_time = 0.0
        self.last_fast_review_key = None
        self.fast_review_candidate = None
        self.fast_review_candidate_count = 0
        self.stable_detection_result = None
        self.pending_detection_class = None
        self.pending_detection_count = 0
        self._set_raw_text("暂无数据")
        if self.latest_live_image_payload is not None:
            self.render_live_image(self.latest_live_image_payload)
        if HISTORY_JSONL.exists():
            HISTORY_JSONL.unlink()

    def export_csv(self):
        if not self.history_records:
            messagebox.showinfo("提示", "暂无历史记录可导出。")
            return

        default_name = "lens_defect_history_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["接收时间", "是否有缺陷", "缺陷数量", "整体等级", "OpenMV时间戳", "原始JSON"])
            for record in self.history_records:
                writer.writerow([
                    record["receive_time"],
                    "是" if record["result"].get("has_defect") else "否",
                    record["result"].get("defect_count", 0),
                    level_name(record["result"].get("overall_level", "normal")),
                    record["result"].get("timestamp", ""),
                    record["raw_json"],
                ])
        messagebox.showinfo("完成", "已导出：%s" % path)

    def mock_one_result(self):
        sample = {
            "has_defect": True,
            "defect_count": 2,
            "summary": {
                "scratch": 1,
                "stain": 1,
            },
            "overall_level": "medium",
            "defects": [
                {"type": "scratch", "confidence": 0.86, "x": 120, "y": 80, "w": 70, "h": 5, "area": 350, "length": 70, "aspect_ratio": 14.0, "level": "medium"},
                {"type": "stain", "confidence": 0.68, "x": 88, "y": 112, "w": 22, "h": 18, "area": 180, "length": 22, "aspect_ratio": 1.22, "level": "light"},
            ],
            "timestamp": int(time.time() * 1000) % 10000000,
            "lens": {
                "found": True,
                "x": 38,
                "y": 28,
                "w": 244,
                "h": 182,
                "cx": 160,
                "cy": 119,
                "confidence": 0.83,
                "lost_frames": 0,
                "source": "mock",
            },
        }
        self.handle_json_line(json.dumps(sample, ensure_ascii=False))

    def _reset_summary(self):
        if self.summary_tree is None:
            return
        self.summary_tree.delete(*self.summary_tree.get_children())
        for defect_type in SUMMARY_TYPES:
            self.summary_tree.insert("", tk.END, values=(defect_type_name(defect_type), 0))

    def _set_raw_text(self, value):
        if self.raw_text is None:
            return
        pretty_value = value
        try:
            pretty_value = json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._set_text(self.raw_text, pretty_value)

    def _set_text(self, widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def on_close(self):
        self.auto_capture_running = False
        self.reader.stop()
        self.root.destroy()


def main():
    if not acquire_single_instance_lock():
        return
    root = tk.Tk()
    LensDefectHostApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
