import csv
import ctypes
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
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
SINGLE_INSTANCE_LOCK_FILE_HANDLE = None


APP_TITLE = "OpenMV N6 缺陷识别上位机"
DEFAULT_BAUDRATE = "115200"
READ_TIMEOUT_SECONDS = 0.02
CAPTURE_TIMEOUT_SECONDS = 12
FRAME_TIMEOUT_SECONDS = 2.0
SERIAL_READ_CHUNK_SIZE = 16384
SERIAL_IMAGE_BYTES_PER_SECOND = 60000
SERIAL_NO_DATA_WARNING_SECONDS = 5.0
SERIAL_NO_DATA_REPEAT_SECONDS = 5.0
SERIAL_MAX_IMAGE_BYTES = 2_000_000
SERIAL_MAX_TEXT_LINE_BYTES = 8192
SERIAL_SYNC_KEEP_BYTES = 32
SERIAL_SYNC_STATUS_SECONDS = 2.0
AUTO_START_RECEIVE = False
MCU_SEND_MIN_INTERVAL_SECONDS = 1.0
LOW_LATENCY_YOLO_MIN_INTERVAL_SECONDS = 0.12
CONVEYOR_CENTER_STABLE_FRAMES = 2
CONVEYOR_GATE_X_MIN = 0.20
CONVEYOR_GATE_X_MAX = 0.80
CONVEYOR_GATE_Y_MIN = 0.16
CONVEYOR_GATE_Y_MAX = 0.90
CONVEYOR_ROI_CENTER_X_MIN = 0.28
CONVEYOR_ROI_CENTER_X_MAX = 0.74
CONVEYOR_ROI_CENTER_Y_MIN = 0.24
CONVEYOR_ROI_CENTER_Y_MAX = 0.84
CONVEYOR_ROI_EDGE_MARGIN_RATIO = 0.035
CONVEYOR_FUSION_MIN_FRAMES = 3
CONVEYOR_FUSION_MAX_FRAMES = 5
CONVEYOR_FUSION_DEFECT_MIN_VOTES = 2
CONVEYOR_FUSION_NORMAL_MIN_VOTES = 3
CONVEYOR_FUSION_STRONG_CONFIDENCE = 0.92
CONVEYOR_MIN_WORKPIECE_CONFIDENCE = 0.20
CONVEYOR_FALLBACK_ROI_SOURCES = {
    "",
    "fallback",
    "fixed",
    "hold",
    "pc_center_fallback",
    "pc_yolo_center_guard",
}
INSPECTION_CLAHE_CLIP_LIMIT = 2.0
INSPECTION_CLAHE_TILE_GRID = (8, 8)
MORPH_STAIN_MIN_RESPONSE = 10.0
MORPH_SCRATCH_MIN_RESPONSE = 9.0

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name in ("windows_host", "dist", "release") else APP_DIR
SINGLE_INSTANCE_LOCK_FILE = Path(tempfile.gettempdir()) / "LensDefectHostOpenMVN6.lock"
HISTORY_DIR = APP_DIR / "history"
HISTORY_JSONL = HISTORY_DIR / "detection_history.jsonl"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SLIDE_DATASET_DIR = PROJECT_ROOT / "dataset_slide"
DEFAULT_STAGE2_MODEL = DEFAULT_MODELS_DIR / "lens_stage2_anomaly.npz"
DEFAULT_YOLO_MODEL = DEFAULT_MODELS_DIR / "lens_yolo.onnx"
DEFAULT_YOLO_SEG_MODEL = DEFAULT_MODELS_DIR / "lens_yolo_seg.onnx"
DEFAULT_YOLO_LABELS = DEFAULT_MODELS_DIR / "lens_yolo_labels.txt"
DEFAULT_YOLO_META = DEFAULT_MODELS_DIR / "lens_yolo_meta.json"
DEFAULT_SLIDE_MODEL = DEFAULT_MODELS_DIR / "slide_defect_classifier_int8.tflite"
DEFAULT_SLIDE_LABELS = DEFAULT_MODELS_DIR / "slide_defect_labels.txt"
DEFAULT_SLIDE_SUMMARY = DEFAULT_MODELS_DIR / "slide_training_summary.json"
DEFAULT_SLIDE_YOLO_MODEL = DEFAULT_MODELS_DIR / "slide_yolo.onnx"
DEFAULT_SLIDE_YOLO_LABELS = DEFAULT_MODELS_DIR / "slide_yolo_labels.txt"
CORRECTION_METADATA_FILENAME = "corrections.csv"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
METADATA_FILENAME = "metadata.csv"
SPLIT_TARGET_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
MIN_RECOMMENDED_IMAGES_PER_CLASS = 100
HOST_DEFECT_CONFIRM_UPDATES = 1
HOST_NORMAL_CONFIRM_UPDATES = 2
HOST_DEFECT_CLASS_CONFIRM_UPDATES = 2
HOST_BLACK_STAIN_TO_SCRATCH_CONFIRM_UPDATES = 3
STAGE2_MIN_INTERVAL_SECONDS = 1.2
FAST_REVIEW_ENABLED = True
FAST_REVIEW_PROMOTE_NORMAL = True
FAST_REVIEW_MIN_INTERVAL_SECONDS = 0.10
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
FAST_REVIEW_EDGE_FILTER_STAIN_STRONG_MIN_OVERLAP = 0.66
FAST_REVIEW_EDGE_FILTER_SCRATCH_MIN_OVERLAP = 0.52
FAST_REVIEW_EDGE_FILTER_STAIN_CENTER_DISTANCE_RATIO = 0.055
FAST_REVIEW_EDGE_FILTER_STAIN_STRONG_CENTER_DISTANCE_RATIO = 0.040
FAST_REVIEW_EDGE_FILTER_STAIN_BOX_DISTANCE_RATIO = 0.040
FAST_REVIEW_EDGE_FILTER_STAIN_STRONG_BOX_DISTANCE_RATIO = 0.032
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
FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_DELTA = 14.5
FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_ASPECT = 2.05
FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_SLENDER_RATIO = 0.60
FAST_REVIEW_REFLECTION_GLARE_MIN_GRAY = 182.0
FAST_REVIEW_REFLECTION_GLARE_MIN_LOCAL_DELTA = 48.0
FAST_REVIEW_REFLECTION_SHADOW_MIN_GLARE_DELTA = 72.0
FAST_REVIEW_REFLECTION_SHADOW_MAX_SIGNED_DELTA = -2.0
FAST_REVIEW_CENTER_CROSS_WEAK_MIN_TOTAL_LENGTH = 900.0
FAST_REVIEW_CENTER_CROSS_WEAK_MIN_SLENDER_RATIO = 0.68
FAST_REVIEW_CENTER_CROSS_WEAK_MIN_FILL_RATIO = 0.25
FAST_REVIEW_REFLECTION_SCRATCH_MAX_TOTAL_LENGTH = 860.0
FAST_REVIEW_REFLECTION_SCRATCH_MAX_LINE_COUNT = 24
FAST_REVIEW_YOLO_BOX_LOCK_MIN_CONFIDENCE = 0.10
FAST_REVIEW_YOLO_BOX_LOCK_MIN_IOU = 0.025
FAST_REVIEW_YOLO_BOX_LOCK_MAX_CENTER_DISTANCE_RATIO = 0.20
FAST_REVIEW_YOLO_BOX_LOCK_EXPAND_RATIO = 0.45
FAST_REVIEW_YOLO_BOX_LOCK_CLASS_MARGIN = 0.07
FAST_REVIEW_YOLO_ROI_HIGH_CONFIDENCE = 0.88
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
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_LOCAL_DELTA = 5.6
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MIN_SIGNED_DELTA = -11.5
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_SIGNED_DELTA = -4.0
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_ANGLE_GROUPS = 2
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_ASPECT = 1.90
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_LENGTH = 105
FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_AREA = 2600
FAST_REVIEW_DARK_STAIN_SAFE_MIN_CONFIDENCE = 0.90
FAST_REVIEW_DARK_STAIN_SAFE_MIN_AREA = 900
FAST_REVIEW_DARK_STAIN_SAFE_MIN_LENGTH = 48
FAST_REVIEW_DARK_STAIN_SAFE_MIN_DENSITY = 0.40
FAST_REVIEW_DARK_STAIN_SAFE_MAX_DENSITY = 0.68
FAST_REVIEW_DARK_STAIN_SAFE_STRONG_SIGNED_DELTA = -24.0
FAST_REVIEW_DARK_STAIN_SAFE_STRONG_LOCAL_DELTA = 12.0
FAST_REVIEW_DARK_STAIN_SAFE_STAR_SIGNED_DELTA = -30.0
FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_AREA = 1800
FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_LENGTH = 82
FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_ASPECT = 2.0
FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_DENSITY = 0.55
FAST_REVIEW_DARK_STAIN_SAFE_MIN_MASK_OVERLAP = 0.58
FAST_REVIEW_DARK_STAIN_SAFE_BOX_DISTANCE_RATIO = 0.014
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
FAST_REVIEW_CENTER_CROSS_MIN_TOTAL_LENGTH = 180.0
FAST_REVIEW_CENTER_CROSS_MIN_LOCAL_DELTA = 5.0
FAST_REVIEW_CENTER_CROSS_MAX_BOX_AREA_RATIO = 0.20
FAST_REVIEW_CENTER_CROSS_MAX_FILL_RATIO = 0.43
FAST_REVIEW_CENTER_CROSS_MAX_VERTICAL_RATIO = 0.48
FAST_REVIEW_DARK_STAR_STAIN_MAX_BRIGHT_DELTA = 13.0
FAST_REVIEW_DARK_STAR_STAIN_MIN_DARK_FRACTION = 0.34
FAST_REVIEW_BLACK_STAIN_MIN_DARK_FRACTION = 0.22
FAST_REVIEW_BLACK_STAIN_MIN_FILL_RATIO = 0.12
FAST_REVIEW_BLACK_STAIN_MAX_BRIGHT_DELTA = 20.0
FAST_REVIEW_BLACK_STAIN_MAX_ASPECT = 3.00
FAST_REVIEW_BLACK_STAIN_MIN_ANGLE_GROUPS = 3
FAST_REVIEW_BLACK_CLUSTER_MIN_COMPONENTS = 2
FAST_REVIEW_BLACK_CLUSTER_MIN_FILL_RATIO = 0.12
FAST_REVIEW_BLACK_CLUSTER_MAX_MERGED_ASPECT = 4.40
FAST_REVIEW_BLACK_CLUSTER_MAX_COMPONENT_ASPECT = 6.20
FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_DARK_FRACTION = 0.08
FAST_REVIEW_BLACK_STAIN_RADIAL_MAX_BRIGHT_FRACTION = 0.46
FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_LINES = 3
FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_ANGLE_GROUPS = 2
FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_TOTAL_LENGTH = 42.0
FAST_REVIEW_BLACK_STAIN_RADIAL_MAX_SIGNED_DELTA = 8.0
FAST_REVIEW_BLACK_STAIN_LOCK_SECONDS = 1.15
FAST_REVIEW_BLACK_STAIN_LOCK_MIN_IOU = 0.035
FAST_REVIEW_BLACK_STAIN_LOCK_MAX_CENTER_DISTANCE_RATIO = 0.62
FAST_REVIEW_DARK_X_STAIN_MIN_LINES = 4
FAST_REVIEW_DARK_X_STAIN_MIN_ANGLE_GROUPS = 3
FAST_REVIEW_DARK_X_STAIN_MIN_TOTAL_LENGTH = 130.0
FAST_REVIEW_DARK_X_STAIN_MIN_DARK_FRACTION = 0.22
FAST_REVIEW_DARK_X_STAIN_MAX_BRIGHT_FRACTION = 0.30
FAST_REVIEW_DARK_X_STAIN_MIN_BOX_RATIO = 0.018
FAST_REVIEW_DARK_X_STAIN_MAX_BOX_RATIO = 0.18
FAST_REVIEW_DARK_X_STAIN_MAX_SIGNED_DELTA = -4.5
FAST_REVIEW_DARK_X_STAIN_MAX_ASPECT = 1.95
FAST_REVIEW_DARK_STAR_LINE_MIN_LINES = 7
FAST_REVIEW_DARK_STAR_LINE_MIN_ANGLE_GROUPS = 3
FAST_REVIEW_DARK_STAR_LINE_MIN_TOTAL_LENGTH = 250.0
FAST_REVIEW_DARK_STAR_LINE_MIN_BOX_AREA_RATIO = 0.020
FAST_REVIEW_DARK_STAR_LINE_MAX_BOX_AREA_RATIO = 0.135
FAST_REVIEW_DARK_STAR_LINE_MIN_ASPECT = 0.45
FAST_REVIEW_DARK_STAR_LINE_MAX_ASPECT = 3.4
FAST_REVIEW_DARK_STAR_LINE_MIN_DARK_FRACTION = 0.18
FAST_REVIEW_DARK_STAR_LINE_MIN_SIGNED_DELTA = -5.0
FAST_REVIEW_DARK_STAR_LINE_BRIGHT_SCRATCH_MIN_P95 = 165.0
FAST_REVIEW_DARK_STAR_LINE_BRIGHT_SCRATCH_MIN_DELTA = 42.0
FAST_REVIEW_DARK_STAR_LINE_BRIGHT_SCRATCH_MIN_FRACTION = 0.14
FAST_REVIEW_DARK_STAR_LINE_MAX_X_RATIO = 0.78
FAST_REVIEW_DARK_STAR_LINE_MAX_Y_RATIO = 0.82
FAST_REVIEW_GLARE_MASK_PERCENTILE = 97.5
FAST_REVIEW_GLARE_MASK_MIN_GRAY = 175.0
FAST_REVIEW_STAR_SCRATCH_MIN_LINES = 4
FAST_REVIEW_STAR_SCRATCH_MIN_LENGTH = 72.0
FAST_REVIEW_STAR_SCRATCH_MIN_FILL = 0.12
FAST_REVIEW_STAR_SCRATCH_MAX_FILL = 0.62
FAST_REVIEW_BRIGHT_SCRATCH_MIN_SIGNED_DELTA = 1.2
FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA = -0.8
FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_ANGLE_GROUPS = 4
FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_LOCAL_DELTA = 9.0
FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_SIGNED_DELTA = 12.0
FAST_REVIEW_CENTER_RADIAL_STAIN_MAX_SIGNED_DELTA = -6.0
FAST_REVIEW_CENTER_RADIAL_STAIN_MAX_BOX_RATIO = 0.16
FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_TOTAL_LENGTH = 54.0
FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_COMPONENTS = 2
FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_ASPECT = 1.55
FAST_REVIEW_CURVILINEAR_SCRATCH_MAX_FILL_RATIO = 0.34
FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_LOCAL_DELTA = 4.5
FAST_REVIEW_CURVILINEAR_SCRATCH_MAX_BOX_RATIO = 0.090
FAST_REVIEW_KEEP_PC_RESULT_SECONDS = 2.20
YOLO_INPUT_SIZE = 640
YOLO_CONFIDENCE_THRESHOLD = 0.08
YOLO_NMS_THRESHOLD = 0.45
YOLO_MIN_INTERVAL_SECONDS = 0.08
YOLO_MISSING_MODEL_RECHECK_SECONDS = 2.0
YOLO_FALLBACK_INPUT_SIZES = (416, 512, 320, 640)
YOLO_ROI_INFERENCE_ENABLED = True
YOLO_ROI_PADDING_RATIO = 0.18
YOLO_ROI_MIN_SIZE_RATIO = 0.28
YOLO_ROI_MAX_AREA_RATIO = 0.70
YOLO_FULL_FRAME_FALLBACK_AFTER_ROI_MISSES = 8
YOLO_AUTO_ROI_SIDE_CLIP_MARGIN_RATIO = 0.025
YOLO_AUTO_ROI_EDGE_PARTIAL_SIZE_RATIO = 0.92
YOLO_AUTO_ROI_MAX_CENTER_X_DRIFT_RATIO = 0.27
YOLO_ROI_HIGH_RES_INPUT_SIZE = 640
YOLO_LOW_CONFIDENCE_TILE_THRESHOLD = 0.34
YOLO_TILE_OVERLAP_RATIO = 0.22
YOLO_TILE_MIN_ROI_SIZE = 180
YOLO_TILE_MAX_TILES = 4
YOLO_TILE_MERGE_IOU_THRESHOLD = 0.18
YOLO_TILE_MERGE_CENTER_DISTANCE_RATIO = 0.32
YOLO_TILE_MERGE_CONTAINMENT_THRESHOLD = 0.58

SUPPORTED_DEFECT_TYPES = (
    "scratch",
    "stain",
)

DEFECT_TYPE_ORDER = ["normal"] + list(SUPPORTED_DEFECT_TYPES)
SUMMARY_TYPES = list(SUPPORTED_DEFECT_TYPES)


def ensure_local_module_paths():
    candidates = [
        APP_DIR,
        APP_DIR / "windows_host",
        APP_DIR.parent / "windows_host",
        PROJECT_ROOT / "windows_host",
    ]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS))  # pylint: disable=protected-access
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except Exception:
            resolved = str(candidate)
        if candidate.exists() and resolved not in sys.path:
            sys.path.insert(0, resolved)

ensure_local_module_paths()

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


@dataclass(frozen=True)
class DetectionModeConfig:
    key: str
    display_name: str
    dataset_dir: Path
    model_file: Path
    labels_file: Path
    training_summary: Path
    deploy_model_name: str
    deploy_labels_name: str
    capture_script: str
    classifier_script: str
    normal_result_text: str
    track_status_name: str
    stage2_model: Path
    yolo_model: Path
    yolo_enabled_default: bool


DETECTION_MODE_CONFIGS = {
    "lens": DetectionModeConfig(
        key="lens",
        display_name="眼镜片检测",
        dataset_dir=DEFAULT_DATASET_DIR,
        model_file=DEFAULT_MODELS_DIR / "lens_defect_classifier_int8.tflite",
        labels_file=DEFAULT_MODELS_DIR / "lens_defect_labels.txt",
        training_summary=DEFAULT_MODELS_DIR / "training_summary.json",
        deploy_model_name="lens_defect_classifier_int8.tflite",
        deploy_labels_name="lens_defect_labels.txt",
        capture_script="openmv/n6_usb_image_capture.py",
        classifier_script="openmv/n6_classifier_main.py",
        normal_result_text="正常镜片",
        track_status_name="镜片跟踪",
        stage2_model=DEFAULT_STAGE2_MODEL,
        yolo_model=DEFAULT_YOLO_SEG_MODEL if DEFAULT_YOLO_SEG_MODEL.exists() else DEFAULT_YOLO_MODEL,
        yolo_enabled_default=True,
    ),
    "slide": DetectionModeConfig(
        key="slide",
        display_name="载玻片检测",
        dataset_dir=DEFAULT_SLIDE_DATASET_DIR,
        model_file=DEFAULT_SLIDE_MODEL,
        labels_file=DEFAULT_SLIDE_LABELS,
        training_summary=DEFAULT_SLIDE_SUMMARY,
        deploy_model_name="slide_defect_classifier_int8.tflite",
        deploy_labels_name="slide_defect_labels.txt",
        capture_script="openmv/n6_usb_slide_capture.py",
        classifier_script="openmv/n6_slide_classifier_main.py",
        normal_result_text="合格",
        track_status_name="载玻片定位",
        stage2_model=DEFAULT_MODELS_DIR / "slide_stage2_anomaly.npz",
        yolo_model=DEFAULT_SLIDE_YOLO_MODEL,
        yolo_enabled_default=False,
    ),
}

DETECTION_MODE_ORDER = ["lens", "slide"]
DETECTION_MODE_DISPLAY_ORDER = [DETECTION_MODE_CONFIGS[key].display_name for key in DETECTION_MODE_ORDER]
DETECTION_MODE_BY_LABEL = {
    DETECTION_MODE_CONFIGS[key].display_name: key
    for key in DETECTION_MODE_ORDER
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


def detection_mode_key(value):
    if value in DETECTION_MODE_CONFIGS:
        return value
    return DETECTION_MODE_BY_LABEL.get(value, "lens")


def detection_mode_label(value):
    return DETECTION_MODE_CONFIGS[detection_mode_key(value)].display_name


def detection_mode_config(value):
    return DETECTION_MODE_CONFIGS[detection_mode_key(value)]


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


def correction_metadata_path(dataset_dir):
    return dataset_dir / CORRECTION_METADATA_FILENAME


def acquire_single_instance_lock():
    global SINGLE_INSTANCE_MUTEX_HANDLE, SINGLE_INSTANCE_LOCK_FILE_HANDLE
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    lock_path = str(SINGLE_INSTANCE_LOCK_FILE)
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    CREATE_ALWAYS = 2
    FILE_ATTRIBUTE_TEMPORARY = 0x00000100
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    file_handle = kernel32.CreateFileW(
        ctypes.c_wchar_p(lock_path),
        GENERIC_READ | GENERIC_WRITE,
        0,
        None,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY,
        None,
    )
    if file_handle is None or file_handle == INVALID_HANDLE_VALUE:
        return False
    SINGLE_INSTANCE_LOCK_FILE_HANDLE = file_handle
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, True, SINGLE_INSTANCE_MUTEX_NAME)
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
    ensure_local_module_paths()
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


def cv_runtime_error_text():
    if cv2 is None:
        return "OpenCV 导入失败：%s" % STAGE2_IMPORT_ERROR
    if np is None:
        return "NumPy 导入失败：%s" % STAGE2_IMPORT_ERROR
    return ""


class YoloOnnxDetector:
    def __init__(self, model_path, labels_path=None, input_size=YOLO_INPUT_SIZE):
        if cv2 is None or np is None:
            raise RuntimeError("%s，不能加载 YOLO ONNX 模型。" % (cv_runtime_error_text() or "缺少 OpenCV 或 NumPy"))
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))
        self.labels = self._load_labels(labels_path)
        self.input_size = self._resolve_input_size(input_size)
        self.task = "segment" if "seg" in self.model_path.stem.lower() else "detect"
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

    def detect(self, image, input_size=None):
        rgb = np.array(image.convert("RGB"))
        frame_h, frame_w = rgb.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return []
        output_size, outputs = self._forward_with_compatible_size(rgb, input_size=input_size)
        return self._parse_outputs(outputs, frame_w, frame_h, output_size)

    def detect_roi(self, image, roi, input_size=None, source="yolo_onnx_roi"):
        if not isinstance(roi, dict):
            return self.detect(image, input_size=input_size)
        image_w, image_h = image.size
        x = max(0, int(roi.get("x", 0) or 0))
        y = max(0, int(roi.get("y", 0) or 0))
        w = max(1, int(roi.get("w", 0) or 0))
        h = max(1, int(roi.get("h", 0) or 0))
        x = min(x, max(0, image_w - 1))
        y = min(y, max(0, image_h - 1))
        w = min(w, image_w - x)
        h = min(h, image_h - y)
        if w <= 4 or h <= 4:
            return []
        crop = image.crop((x, y, x + w, y + h))
        detections = self.detect(crop, input_size=input_size)
        for detection in detections:
            detection["x"] = int(detection.get("x", 0) or 0) + x
            detection["y"] = int(detection.get("y", 0) or 0) + y
            if isinstance(detection.get("mask_polygon"), list):
                detection["mask_polygon"] = [
                    (int(point[0]) + x, int(point[1]) + y)
                    for point in detection["mask_polygon"]
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]
            detection["source"] = source
            detection["roi_inference"] = True
        return detections

    def detect_roi_tiled(self, image, roi, input_size=None):
        if not isinstance(roi, dict):
            return []
        image_w, image_h = image.size
        x = max(0, int(roi.get("x", 0) or 0))
        y = max(0, int(roi.get("y", 0) or 0))
        w = max(1, int(roi.get("w", 0) or 0))
        h = max(1, int(roi.get("h", 0) or 0))
        x = min(x, max(0, image_w - 1))
        y = min(y, max(0, image_h - 1))
        w = min(w, image_w - x)
        h = min(h, image_h - y)
        if w < YOLO_TILE_MIN_ROI_SIZE or h < YOLO_TILE_MIN_ROI_SIZE:
            return []

        tile_w = max(YOLO_TILE_MIN_ROI_SIZE, int(w * 0.62))
        tile_h = max(YOLO_TILE_MIN_ROI_SIZE, int(h * 0.62))
        step_x = max(1, int(tile_w * (1.0 - YOLO_TILE_OVERLAP_RATIO)))
        step_y = max(1, int(tile_h * (1.0 - YOLO_TILE_OVERLAP_RATIO)))
        xs = [x]
        ys = [y]
        if x + tile_w < x + w:
            xs.append(max(x, x + w - tile_w))
        if y + tile_h < y + h:
            ys.append(max(y, y + h - tile_h))
        if w > tile_w * 1.55:
            xs.insert(1, min(x + step_x, max(x, x + w - tile_w)))
        if h > tile_h * 1.55:
            ys.insert(1, min(y + step_y, max(y, y + h - tile_h)))

        detections = []
        tile_count = 0
        for top in dict.fromkeys(ys):
            for left in dict.fromkeys(xs):
                if tile_count >= YOLO_TILE_MAX_TILES:
                    break
                tile_roi = {
                    "x": int(left),
                    "y": int(top),
                    "w": int(min(tile_w, image_w - left)),
                    "h": int(min(tile_h, image_h - top)),
                }
                tile_detections = self.detect_roi(
                    image,
                    tile_roi,
                    input_size=input_size,
                    source="yolo_onnx_roi_tile",
                )
                for detection in tile_detections:
                    detection["tile_inference"] = True
                    detection["tile_roi"] = dict(tile_roi)
                detections.extend(tile_detections)
                tile_count += 1
            if tile_count >= YOLO_TILE_MAX_TILES:
                break
        return self._merge_tile_detections(detections)

    def _forward_with_compatible_size(self, rgb, input_size=None):
        sizes = []
        if input_size:
            sizes.append(int(input_size))
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

    def _nms_detections(self, detections):
        if not detections:
            return []
        boxes = []
        scores = []
        for detection in detections:
            boxes.append([
                int(detection.get("x", 0) or 0),
                int(detection.get("y", 0) or 0),
                int(detection.get("w", 0) or 0),
                int(detection.get("h", 0) or 0),
            ])
            scores.append(float(detection.get("confidence", 0) or 0))
        keep = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_THRESHOLD)
        if len(keep) == 0:
            return []
        return [detections[int(index)] for index in np.array(keep).reshape(-1)]

    def _merge_tile_detections(self, detections):
        if not detections:
            return []
        ordered = sorted(
            [dict(detection) for detection in detections if isinstance(detection, dict)],
            key=lambda detection: float(detection.get("confidence", 0) or 0),
            reverse=True,
        )
        groups = []
        for detection in ordered:
            target = None
            for group in groups:
                if self._tile_detections_should_merge(group[0], detection):
                    target = group
                    break
            if target is None:
                groups.append([detection])
            else:
                target.append(detection)

        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_detection_group(group))
        return self._nms_detections(merged)

    def _tile_detections_should_merge(self, first, second):
        if first.get("type") != second.get("type"):
            return False
        if self._box_iou(first, second) >= YOLO_TILE_MERGE_IOU_THRESHOLD:
            return True
        if self._box_containment(first, second) >= YOLO_TILE_MERGE_CONTAINMENT_THRESHOLD:
            return True
        return self._box_center_distance_ratio(first, second) <= YOLO_TILE_MERGE_CENTER_DISTANCE_RATIO

    def _merge_detection_group(self, group):
        weights = [max(0.01, float(detection.get("confidence", 0) or 0)) for detection in group]
        total_weight = sum(weights) or float(len(group))
        left = sum(float(detection.get("x", 0) or 0) * weight for detection, weight in zip(group, weights)) / total_weight
        top = sum(float(detection.get("y", 0) or 0) * weight for detection, weight in zip(group, weights)) / total_weight
        right = sum((float(detection.get("x", 0) or 0) + float(detection.get("w", 0) or 0)) * weight for detection, weight in zip(group, weights)) / total_weight
        bottom = sum((float(detection.get("y", 0) or 0) + float(detection.get("h", 0) or 0)) * weight for detection, weight in zip(group, weights)) / total_weight
        best = max(group, key=lambda detection: float(detection.get("confidence", 0) or 0))
        merged = dict(best)
        x = int(round(left))
        y = int(round(top))
        w = max(1, int(round(right - left)))
        h = max(1, int(round(bottom - top)))
        confidence = max(float(detection.get("confidence", 0) or 0) for detection in group)
        confidence = min(0.99, confidence + min(0.04, 0.01 * (len(group) - 1)))
        merged.update({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": int(w * h),
            "length": int(max(w, h)),
            "aspect_ratio": round(float(max(w, h)) / float(max(1, min(w, h))), 2),
            "confidence": round(confidence, 2),
            "source": "yolo_onnx_roi_tile_merged",
            "tile_inference": True,
            "tile_merged": True,
            "tile_merge_count": len(group),
        })
        tile_rois = [detection.get("tile_roi") for detection in group if isinstance(detection.get("tile_roi"), dict)]
        if tile_rois:
            merged["tile_rois"] = [dict(tile_roi) for tile_roi in tile_rois]
        return merged

    def _box_iou(self, first, second):
        ax, ay, aw, ah = self._box_values(first)
        bx, by, bw, bh = self._box_values(second)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = aw * ah + bw * bh - intersection
        if union <= 0:
            return 0.0
        return float(intersection) / float(union)

    def _box_containment(self, first, second):
        ax, ay, aw, ah = self._box_values(first)
        bx, by, bw, bh = self._box_values(second)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        smaller = min(aw * ah, bw * bh)
        if smaller <= 0:
            return 0.0
        return float(intersection) / float(smaller)

    def _box_center_distance_ratio(self, first, second):
        ax, ay, aw, ah = self._box_values(first)
        bx, by, bw, bh = self._box_values(second)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 1.0
        acx = ax + aw / 2.0
        acy = ay + ah / 2.0
        bcx = bx + bw / 2.0
        bcy = by + bh / 2.0
        scale = max(aw, ah, bw, bh, 1.0)
        return max(abs(acx - bcx), abs(acy - bcy)) / scale

    def _box_values(self, detection):
        try:
            return (
                float(detection.get("x", 0) or 0),
                float(detection.get("y", 0) or 0),
                float(detection.get("w", 0) or 0),
                float(detection.get("h", 0) or 0),
            )
        except (AttributeError, TypeError, ValueError):
            return 0.0, 0.0, 0.0, 0.0

    def _forward_at_size(self, rgb, input_size):
        blob = cv2.dnn.blobFromImage(
            rgb,
            1.0 / 255.0,
            (int(input_size), int(input_size)),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        if self.task == "segment":
            try:
                names = self.net.getUnconnectedOutLayersNames()
                if names:
                    return self.net.forward(names)
            except Exception:
                pass
        return self.net.forward()

    def _parse_outputs(self, outputs, frame_w, frame_h, input_size=None):
        input_size = int(input_size or self.input_size)
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            return self._parse_segmentation_outputs(outputs, frame_w, frame_h, input_size)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
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

    def _parse_segmentation_outputs(self, outputs, frame_w, frame_h, input_size):
        prediction, proto = self._split_segmentation_outputs(outputs)
        if prediction is None or proto is None:
            return self._parse_outputs(outputs[0], frame_w, frame_h, input_size)
        data = np.asarray(prediction)
        data = np.squeeze(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            data = data.reshape(-1, data.shape[-1])
        label_count = len(self.labels)
        if data.shape[0] < data.shape[1] and data.shape[0] >= 4 + label_count:
            data = data.T

        mask_count = self._seg_proto_channels(proto)
        if mask_count <= 0:
            return self._parse_outputs(prediction, frame_w, frame_h, input_size)
        class_start = 4
        class_end = data.shape[1] - mask_count
        if class_end <= class_start:
            return self._parse_outputs(prediction, frame_w, frame_h, input_size)

        boxes = []
        scores = []
        class_ids = []
        mask_coeffs = []
        for row in data:
            if row.size < class_end + mask_count:
                continue
            values = row.astype(float)
            class_scores = values[class_start:class_end]
            if class_scores.size <= 0:
                continue
            class_index = int(np.argmax(class_scores))
            confidence = float(class_scores[class_index])
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
            scores.append(confidence)
            class_ids.append((class_index, defect_type))
            mask_coeffs.append(values[class_end:class_end + mask_count])

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_THRESHOLD)
        if len(keep) == 0:
            return []

        detections = []
        for index in np.array(keep).reshape(-1):
            idx = int(index)
            left, top, width, height = boxes[idx]
            _class_index, defect_type = class_ids[idx]
            mask_info = self._segmentation_mask_info(proto, mask_coeffs[idx], boxes[idx], frame_w, frame_h)
            if mask_info:
                mx, my, mw, mh = mask_info["box"]
                left, top, width, height = mx, my, mw, mh
            area = int(width * height)
            length = int(max(width, height))
            detection = {
                "type": defect_type,
                "confidence": round(float(scores[idx]), 2),
                "x": int(left),
                "y": int(top),
                "w": int(width),
                "h": int(height),
                "area": area,
                "length": length,
                "aspect_ratio": round(float(length) / float(max(1, min(width, height))), 2),
                "level": "medium" if area >= 500 or length >= 70 else "light",
                "source": "yolo_onnx",
                "model_task": "segment",
                "mask_area": int(mask_info.get("area", 0)) if mask_info else 0,
            }
            if mask_info and mask_info.get("polygon"):
                detection["mask_polygon"] = mask_info["polygon"]
            detections.append(detection)
        return detections

    def _split_segmentation_outputs(self, outputs):
        arrays = [np.asarray(output) for output in outputs]
        prediction = None
        proto = None
        for array in arrays:
            squeezed = np.squeeze(array)
            if squeezed.ndim == 3:
                proto = squeezed
            elif squeezed.ndim in (2, 3):
                prediction = array if prediction is None else prediction
        if prediction is None and arrays:
            prediction = arrays[0]
        return prediction, proto

    def _seg_proto_channels(self, proto):
        shape = np.asarray(proto).shape
        if len(shape) != 3:
            return 0
        return int(shape[0] if shape[0] <= shape[-1] else shape[-1])

    def _normalize_proto(self, proto):
        proto = np.asarray(proto, dtype=np.float32)
        if proto.ndim != 3:
            return None
        if proto.shape[0] <= proto.shape[-1]:
            return proto
        return np.transpose(proto, (2, 0, 1))

    def _segmentation_mask_info(self, proto, coeffs, box, frame_w, frame_h):
        proto = self._normalize_proto(proto)
        if proto is None:
            return None
        coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        if proto.shape[0] != coeffs.shape[0]:
            return None
        mask = np.tensordot(coeffs, proto, axes=(0, 0))
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -40.0, 40.0)))
        mask = cv2.resize(mask, (int(frame_w), int(frame_h)), interpolation=cv2.INTER_LINEAR)
        binary = np.where(mask >= 0.50, 255, 0).astype(np.uint8)
        x, y, w, h = [int(value) for value in box]
        clipped = np.zeros_like(binary)
        clipped[y:y + h, x:x + w] = binary[y:y + h, x:x + w]
        if cv2.countNonZero(clipped) <= 0:
            return None
        clipped = cv2.morphologyEx(clipped, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        contours, _hierarchy = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= 3.0]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        mx, my, mw, mh = cv2.boundingRect(contour)
        if mw <= 0 or mh <= 0:
            return None
        polygon = self._contour_polygon(contour)
        return {
            "box": (int(mx), int(my), int(mw), int(mh)),
            "area": int(cv2.contourArea(contour)),
            "polygon": polygon,
        }

    def _contour_polygon(self, contour):
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, perimeter * 0.012)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx) > 28:
            epsilon *= 1.35
            approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            return []
        return [(int(point[0][0]), int(point[0][1])) for point in approx]


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
        last_sync_status_time = 0.0
        buffer = bytearray()

        while self.running:
            try:
                if self.serial_port is None:
                    break

                chunk = self.serial_port.read(SERIAL_READ_CHUNK_SIZE)
                if not chunk:
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
                buffer.extend(chunk)
                self._process_buffer(buffer)
                if len(buffer) > SERIAL_MAX_IMAGE_BYTES:
                    marker_index = buffer.rfind(b"IMG_BEGIN ")
                    if marker_index >= 0:
                        del buffer[:marker_index]
                    else:
                        keep = min(len(buffer), SERIAL_SYNC_KEEP_BYTES)
                        del buffer[:-keep]
                        now = time.time()
                        if now - last_sync_status_time >= SERIAL_SYNC_STATUS_SECONDS:
                            self.ui_queue.put(UiMessage("status", "正在同步 OpenMV 数据流，请稍等下一帧"))
                            last_sync_status_time = now
            except Exception as exc:
                self.ui_queue.put(UiMessage("error", "串口读取失败：%s" % exc))
                break

        self.running = False
        self.ui_queue.put(UiMessage("status", "检测接收已停止"))

    def _process_buffer(self, buffer):
        while self.running and buffer:
            img_index = buffer.find(b"IMG_BEGIN ")
            line_index = buffer.find(b"\n")

            if img_index >= 0 and (line_index < 0 or img_index <= line_index):
                if img_index > 0:
                    del buffer[:img_index]
                if not self._read_image_frame_from_buffer(buffer):
                    return
                continue

            if line_index >= 0:
                raw_line = bytes(buffer[:line_index])
                del buffer[:line_index + 1]
                self._handle_text_line(raw_line)
                continue

            if len(buffer) > SERIAL_MAX_TEXT_LINE_BYTES:
                marker_index = buffer.rfind(b"IMG_BEGIN ")
                if marker_index >= 0:
                    del buffer[:marker_index]
                else:
                    keep = min(len(buffer), SERIAL_SYNC_KEEP_BYTES)
                    del buffer[:-keep]
            return

    def _read_image_frame_from_buffer(self, buffer):
        header_end = buffer.find(b"\n")
        if header_end < 0:
            return False

        header_line = buffer[:header_end].decode("utf-8", errors="ignore").strip()
        parts = header_line.split()
        if len(parts) < 2:
            del buffer[:header_end + 1]
            return True
        try:
            size = int(parts[1])
            width = int(parts[2]) if len(parts) >= 4 else 0
            height = int(parts[3]) if len(parts) >= 4 else 0
        except ValueError:
            del buffer[:header_end + 1]
            self.ui_queue.put(UiMessage("status", "OpenMV 画面头格式错误：%s" % header_line))
            return True
        if size <= 0 or size > SERIAL_MAX_IMAGE_BYTES:
            del buffer[:header_end + 1]
            self.ui_queue.put(UiMessage("status", "OpenMV 画面大小异常：%d 字节" % size))
            return True

        payload_start = header_end + 1
        payload_end = payload_start + size
        end_marker = b"IMG_END\n"
        frame_end = payload_end + len(end_marker)
        if len(buffer) < frame_end:
            return False
        if bytes(buffer[payload_end:frame_end]) != end_marker:
            next_marker = buffer.find(b"IMG_BEGIN ", 1)
            if next_marker >= 0:
                del buffer[:next_marker]
            else:
                keep = min(len(buffer), SERIAL_SYNC_KEEP_BYTES)
                del buffer[:-keep]
            self.ui_queue.put(UiMessage("status", "OpenMV 画面帧尾丢失，正在重新同步"))
            return True

        data = bytes(buffer[payload_start:payload_end])
        del buffer[:frame_end]
        self.ui_queue.put(UiMessage("live_image", {
            "image_bytes": data,
            "width": width,
            "height": height,
            "byte_count": len(data),
            "receive_time": now_text(),
        }))
        return True

    def _handle_text_line(self, raw_line):
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            return
        if line.startswith("{"):
            self.ui_queue.put(UiMessage("line", line))
        elif line.startswith("ERR ") or line.startswith("BOOT "):
            self.ui_queue.put(UiMessage("status", line))


class MicrocontrollerSerialClient:
    def __init__(self):
        self.serial_port = None
        self.port_name = ""

    @property
    def connected(self):
        return self.serial_port is not None and self.serial_port.is_open

    def connect(self, port, baudrate):
        if serial is None:
            raise RuntimeError("缺少 pyserial，请先运行 run_windows_host.bat 安装依赖。")
        self.disconnect()
        self.serial_port = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            timeout=0,
            write_timeout=0.5,
        )
        self.port_name = port
        time.sleep(0.2)

    def disconnect(self):
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.port_name = ""

    def send_line(self, line):
        if not self.connected:
            raise RuntimeError("单片机串口未连接。")
        payload = (line.strip() + "\n").encode("utf-8")
        self.serial_port.write(payload)
        self.serial_port.flush()


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
                raise TimeoutError("等待 OpenMV 图片数据超时，请确认 N6 正在运行对应的 USB 采集脚本")

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
        self.mcu_client = MicrocontrollerSerialClient()
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
        self.yolo_roi_miss_count = 0
        self.last_stage2_infer_time = 0.0
        self.last_fast_review_time = 0.0
        self.last_fast_review_key = None
        self.fast_review_candidate = None
        self.fast_review_candidate_count = 0
        self.last_pc_defect_result = None
        self.last_pc_defect_time = 0.0
        self.last_black_stain_result = None
        self.last_black_stain_time = 0.0
        self.stable_detection_result = None
        self.pending_detection_class = None
        self.pending_detection_count = 0
        self.active_mode_key = "lens"
        self.mode_state = {}
        for mode_key in DETECTION_MODE_ORDER:
            config = detection_mode_config(mode_key)
            self.mode_state[mode_key] = {
                "dataset_dir": str(config.dataset_dir),
                "stage2_model": str(config.stage2_model),
                "stage2_enabled": False,
                "yolo_model": str(config.yolo_model),
                "yolo_enabled": config.yolo_enabled_default,
                "model_file": str(config.model_file),
                "labels_file": str(config.labels_file),
            }
        active_mode = detection_mode_config(self.active_mode_key)

        self.port_var = tk.StringVar()
        self.object_mode_var = tk.StringVar(value=active_mode.display_name)
        self.baud_var = tk.StringVar(value=DEFAULT_BAUDRATE)
        self.mcu_port_var = tk.StringVar()
        self.mcu_baud_var = tk.StringVar(value=DEFAULT_BAUDRATE)
        self.mcu_auto_send_var = tk.BooleanVar(value=True)
        self.mcu_status_var = tk.StringVar(value="单片机：未连接")
        self.last_mcu_send_key = None
        self.last_mcu_send_time = 0.0
        self.status_var = tk.StringVar(value="状态：未连接")
        self.stage2_enabled_var = tk.BooleanVar(value=False)
        self.stage2_model_var = tk.StringVar(value=str(active_mode.stage2_model))
        self.stage2_status_var = tk.StringVar(value="高速复核：已启用；二级模型未勾选")
        self.yolo_enabled_var = tk.BooleanVar(value=active_mode.yolo_enabled_default)
        self.yolo_model_var = tk.StringVar(value=str(active_mode.yolo_model))
        self.yolo_status_var = tk.StringVar(value="YOLO：未加载；无模型时用电脑快速复核兜底")
        self.yolo_high_res_roi_var = tk.BooleanVar(value=True)
        self.yolo_low_conf_tile_var = tk.BooleanVar(value=True)
        self.conveyor_center_gate_var = tk.BooleanVar(value=True)
        self.low_latency_mode_var = tk.BooleanVar(value=True)
        self.conveyor_centered_count = 0
        self.conveyor_fusion_results = []
        self.conveyor_workpiece_payload_key = None
        self.conveyor_workpiece_roi = None
        self._last_effective_roi = None

        self.has_defect_var = tk.StringVar(value="是否检测到缺陷：暂无数据")
        self.class_result_var = tk.StringVar(value="识别结果：暂无数据")
        self.confidence_result_var = tk.StringVar(value="置信度：--")
        self.count_var = tk.StringVar(value="缺陷总数：0")
        self.level_var = tk.StringVar(value="整体严重程度：暂无数据")
        self.lens_track_var = tk.StringVar(value="镜片跟踪：暂无数据")
        self.live_image_var = tk.StringVar(value="OpenMV 画面：请选择 OpenMV 的 COM 口，点击“开始接收识别结果和画面”")

        self.dataset_dir_var = tk.StringVar(value=str(active_mode.dataset_dir))
        self.dataset_class_var = tk.StringVar(value=defect_type_name("normal"))
        self.dataset_split_var = tk.StringVar(value="train")
        self.auto_split_var = tk.BooleanVar(value=True)
        self.capture_interval_var = tk.StringVar(value="1.0")
        self.capture_count_var = tk.StringVar(value="当前类别图片数：0")
        self.dataset_health_var = tk.StringVar(value="数据集体检：暂无统计")
        self.capture_preview_var = tk.StringVar(value="采集预览：暂无图片")
        self.capture_quality_var = tk.StringVar(value="质量提示：暂无图片")

        self.model_file_var = tk.StringVar(value=str(active_mode.model_file))
        self.labels_file_var = tk.StringVar(value=str(active_mode.labels_file))
        self.openmv_folder_var = tk.StringVar(value="")
        self.live_image_photo = None
        self.capture_preview_photo = None
        self.summary_tree = None
        self.defect_tree = None
        self.raw_text = None
        self.history_tree = None

        self._create_widgets()
        self.apply_mode_state(self.active_mode_key, reset_runtime=True)
        self.refresh_ports()
        self.load_history()
        self.root.after(80, self._poll_ui_queue)
        self.root.after(350, self.preload_yolo_model)
        self.root.after(700, self.preload_stage2_model)
        if AUTO_START_RECEIVE:
            self.root.after(500, self.start_receive)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def current_mode_config(self):
        return detection_mode_config(self.active_mode_key)

    def mode_uses_lens_postprocess(self):
        return self.active_mode_key == "lens"

    def save_mode_state(self, mode_key=None):
        if mode_key is None:
            mode_key = self.active_mode_key
        if mode_key not in self.mode_state:
            return
        self.mode_state[mode_key].update({
            "dataset_dir": self.dataset_dir_var.get().strip(),
            "stage2_model": self.stage2_model_var.get().strip(),
            "stage2_enabled": bool(self.stage2_enabled_var.get()),
            "yolo_model": self.yolo_model_var.get().strip(),
            "yolo_enabled": bool(self.yolo_enabled_var.get()),
            "model_file": self.model_file_var.get().strip(),
            "labels_file": self.labels_file_var.get().strip(),
        })

    def apply_mode_state(self, mode_key, reset_runtime=False):
        config = detection_mode_config(mode_key)
        state = self.mode_state[mode_key]
        self.object_mode_var.set(config.display_name)
        self.dataset_dir_var.set(state["dataset_dir"] or str(config.dataset_dir))
        self.stage2_model_var.set(state["stage2_model"] or str(config.stage2_model))
        self.stage2_enabled_var.set(bool(state["stage2_enabled"]))
        self.yolo_model_var.set(state["yolo_model"] or str(config.yolo_model))
        self.yolo_enabled_var.set(bool(state["yolo_enabled"]))
        self.model_file_var.set(state["model_file"] or str(config.model_file))
        self.labels_file_var.set(state["labels_file"] or str(config.labels_file))

        if self.mode_uses_lens_postprocess():
            self.stage2_status_var.set("高速复核：镜片模式可用，按需加载二级模型")
            self.yolo_status_var.set("YOLO：镜片模式可用，按需加载 ONNX 或使用快速复核")
            self.root.after(80, self.preload_yolo_model)
            self.root.after(160, self.preload_stage2_model)
        else:
            self.stage2_status_var.set("载玻片模式：保留 OpenMV 原始 JSON 显示，不启用镜片二级复核")
            self.yolo_status_var.set("载玻片模式：保留 OpenMV 原始 JSON 显示，不启用镜片 YOLO 复核")

        if reset_runtime:
            self.reset_mode_runtime_state()
        self.refresh_dataset_counts()
        self.load_training_summary()

    def reset_mode_runtime_state(self):
        self.latest_detection_result = None
        self.last_fast_review_time = 0.0
        self.last_fast_review_key = None
        self.fast_review_candidate = None
        self.fast_review_candidate_count = 0
        self.last_pc_defect_result = None
        self.last_pc_defect_time = 0.0
        self.last_black_stain_result = None
        self.last_black_stain_time = 0.0
        self.stable_detection_result = None
        self.pending_detection_class = None
        self.pending_detection_count = 0
        self.conveyor_centered_count = 0
        self.conveyor_fusion_results = []
        self.conveyor_workpiece_payload_key = None
        self.conveyor_workpiece_roi = None
        self._last_effective_roi = None
        self.has_defect_var.set("是否检测到缺陷：暂无数据")
        self.class_result_var.set("识别结果：暂无数据")
        self.confidence_result_var.set("置信度：--")
        self.count_var.set("缺陷总数：0")
        self.level_var.set("整体严重程度：暂无数据")
        self.lens_track_var.set("%s：暂无数据" % self.current_mode_config().track_status_name)
        if self.defect_tree is not None:
            self.defect_tree.delete(*self.defect_tree.get_children())
        self._reset_summary()
        if self.raw_text is not None:
            self._set_raw_text("暂无数据")
        if self.latest_live_image_payload is not None:
            self.render_live_image(self.latest_live_image_payload)

    def on_mode_changed(self, _event=None):
        new_mode_key = detection_mode_key(self.object_mode_var.get())
        if new_mode_key == self.active_mode_key:
            return
        self.save_mode_state(self.active_mode_key)
        self.active_mode_key = new_mode_key
        self.apply_mode_state(new_mode_key, reset_runtime=True)
        self.status_var.set("状态：已切换到%s" % self.current_mode_config().display_name)

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

        ttk.Label(serial_frame, text="检测对象").pack(side=tk.LEFT)
        self.object_mode_combo = ttk.Combobox(
            serial_frame,
            textvariable=self.object_mode_var,
            width=14,
            values=DETECTION_MODE_DISPLAY_ORDER,
            state="readonly",
        )
        self.object_mode_combo.pack(side=tk.LEFT, padx=(8, 12))
        self.object_mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)

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

        yolo_opt_frame = ttk.Frame(self.detect_tab)
        yolo_opt_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(
            yolo_opt_frame,
            text="中间 ROI 高分辨率推理",
            variable=self.yolo_high_res_roi_var,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            yolo_opt_frame,
            text="低置信度时 ROI 切片复核",
            variable=self.yolo_low_conf_tile_var,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            yolo_opt_frame,
            text="传送带中心触发",
            variable=self.conveyor_center_gate_var,
        ).pack(side=tk.LEFT, padx=(18, 4))
        ttk.Checkbutton(
            yolo_opt_frame,
            text="生产低延迟",
            variable=self.low_latency_mode_var,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            yolo_opt_frame,
            text="这是污渍",
            command=lambda: self.save_live_correction("stain"),
        ).pack(side=tk.LEFT, padx=(18, 4))
        ttk.Button(
            yolo_opt_frame,
            text="这是划痕",
            command=lambda: self.save_live_correction("scratch"),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            yolo_opt_frame,
            text="这是正常",
            command=lambda: self.save_live_correction("normal"),
        ).pack(side=tk.LEFT, padx=4)

        mcu_frame = ttk.LabelFrame(self.detect_tab, text="单片机输出", padding=6)
        mcu_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mcu_frame, text="COM 口").pack(side=tk.LEFT)
        self.mcu_port_combo = ttk.Combobox(
            mcu_frame,
            textvariable=self.mcu_port_var,
            width=28,
            state="readonly",
        )
        self.mcu_port_combo.pack(side=tk.LEFT, padx=(8, 10))
        ttk.Label(mcu_frame, text="波特率").pack(side=tk.LEFT)
        ttk.Combobox(
            mcu_frame,
            textvariable=self.mcu_baud_var,
            width=10,
            values=("9600", "19200", "38400", "57600", "115200", "921600"),
        ).pack(side=tk.LEFT, padx=(8, 10))
        ttk.Button(mcu_frame, text="连接单片机", command=self.connect_mcu).pack(side=tk.LEFT, padx=4)
        ttk.Button(mcu_frame, text="断开", command=self.disconnect_mcu).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            mcu_frame,
            text="自动发送结果",
            variable=self.mcu_auto_send_var,
        ).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(mcu_frame, textvariable=self.mcu_status_var, style="Status.TLabel").pack(side=tk.LEFT, padx=8)

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
        self._set_text(
            self.capture_log_text,
            "操作步骤：\n"
            "1. 眼镜片采集运行 openmv/n6_usb_image_capture.py；载玻片采集运行 openmv/n6_usb_slide_capture.py。\n"
            "2. 关闭 OpenMV IDE 串口占用，或确保上位机能打开 N6 的 COM 口。\n"
            "3. 在上方切换检测对象后，再开始采集并保存到对应数据集目录。",
        )

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
        ttk.Button(train_frame, text="生成 YOLO-seg 标注", command=self.run_yolo_seg_seed_export_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="预览 YOLO 标注", command=self.run_yolo_preview_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="训练 YOLO ONNX", command=self.run_yolo_training_script).pack(side=tk.LEFT, padx=4)
        ttk.Button(train_frame, text="训练 YOLO-seg ONNX", command=self.run_yolo_seg_training_script).pack(side=tk.LEFT, padx=4)
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
            "眼镜片采集：openmv/n6_usb_image_capture.py；载玻片采集：openmv/n6_usb_slide_capture.py。\n"
            "眼镜片模型检测：openmv/n6_classifier_main.py，会加载 /lens_defect_classifier_int8.tflite 和 /lens_defect_labels.txt。\n"
            "载玻片模型检测：openmv/n6_slide_classifier_main.py，会加载 /slide_defect_classifier_int8.tflite 和 /slide_defect_labels.txt。\n"
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
        if hasattr(self, "mcu_port_combo"):
            self.mcu_port_combo["values"] = ports
        if ports and not self.port_var.get():
            preferred = [
                item for item in ports
                if "OpenMV" in item or "USB" in item or "串行设备" in item
            ]
            self.port_var.set(preferred[0] if preferred else ports[0])
            self.status_var.set("状态：找到 %d 个串口，请选择 OpenMV N6 对应 COM 口" % len(ports))
        elif not ports:
            self.port_var.set("")
            if hasattr(self, "mcu_port_var"):
                self.mcu_port_var.set("")
            self.status_var.set("状态：未找到串口。请用 USB 连接 OpenMV N6")
            self.live_image_var.set("OpenMV 画面：未找到串口，请连接 N6 后点击“刷新串口”")
        if ports and hasattr(self, "mcu_port_var") and not self.mcu_port_var.get():
            openmv_port = self.port_var.get().split(" - ", 1)[0].strip()
            mcu_candidates = [
                item for item in ports
                if item.split(" - ", 1)[0].strip() != openmv_port
            ]
            if len(mcu_candidates) == 1:
                self.mcu_port_var.set(mcu_candidates[0])

    def selected_port(self):
        selected = self.port_var.get().strip()
        if not selected:
            raise RuntimeError("请先选择 COM 口。")
        return selected.split(" - ", 1)[0].strip()

    def selected_baudrate(self):
        baudrate = self.baud_var.get().strip()
        int(baudrate)
        return baudrate

    def selected_mcu_port(self):
        selected = self.mcu_port_var.get().strip()
        if not selected:
            raise RuntimeError("请先选择单片机 COM 口。")
        port = selected.split(" - ", 1)[0].strip()
        openmv_port = self.port_var.get().split(" - ", 1)[0].strip()
        if openmv_port and port == openmv_port:
            raise RuntimeError("单片机 COM 口不能和 OpenMV 接收口相同，请选择另一个串口。")
        return port

    def selected_mcu_baudrate(self):
        baudrate = self.mcu_baud_var.get().strip()
        int(baudrate)
        return baudrate

    def connect_mcu(self):
        try:
            port = self.selected_mcu_port()
            baudrate = self.selected_mcu_baudrate()
            self.mcu_client.connect(port, baudrate)
        except Exception as exc:
            messagebox.showerror("单片机连接失败", str(exc))
            self.mcu_status_var.set("单片机：连接失败")
            return
        self.last_mcu_send_key = None
        self.last_mcu_send_time = 0.0
        self.mcu_status_var.set("单片机：已连接 %s，等待识别结果" % port)

    def disconnect_mcu(self):
        self.mcu_client.disconnect()
        self.last_mcu_send_key = None
        self.last_mcu_send_time = 0.0
        self.mcu_status_var.set("单片机：已断开")

    def format_mcu_result_line(self, class_key, confidence):
        code = {
            "normal": "NORMAL",
            "scratch": "SCRATCH",
            "stain": "STAIN",
            "error": "ERROR",
        }.get(class_key, "UNKNOWN")
        ok_flag = 1 if class_key == "normal" else 0
        confidence_percent = 0
        if confidence is not None:
            try:
                confidence_value = float(confidence)
                if confidence_value <= 1.0:
                    confidence_value *= 100.0
                confidence_percent = max(0, min(100, int(round(confidence_value))))
            except (TypeError, ValueError):
                confidence_percent = 0
        return "RESULT,%s,%d,%d" % (code, ok_flag, confidence_percent)

    def maybe_send_mcu_result(self, result):
        if not self.mcu_auto_send_var.get() or not self.mcu_client.connected:
            return
        if isinstance(result, dict) and result.get("conveyor_waiting"):
            return
        class_key, _class_text, confidence = self.display_class_result(result)
        line = self.format_mcu_result_line(class_key, confidence)
        now = time.monotonic()
        send_key = line
        if (
            send_key == self.last_mcu_send_key
            and now - self.last_mcu_send_time < MCU_SEND_MIN_INTERVAL_SECONDS
        ):
            return
        try:
            self.mcu_client.send_line(line)
        except Exception as exc:
            self.mcu_client.disconnect()
            self.mcu_status_var.set("单片机：发送失败，已断开：%s" % exc)
            return
        self.last_mcu_send_key = send_key
        self.last_mcu_send_time = now
        self.mcu_status_var.set("单片机：已发送 %s" % line)

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
                    quality["warnings"].append("对比度偏低，透明玻璃表面缺陷可能不明显")
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

    def save_live_correction(self, corrected_class):
        corrected_class = defect_type_key(corrected_class)
        if self.latest_live_image_payload is None or "image_bytes" not in self.latest_live_image_payload:
            messagebox.showwarning("暂无画面", "还没有收到当前画面，先开始接收 OpenMV 画面。")
            return

        dataset_dir = Path(self.dataset_dir_var.get())
        split = self.choose_balanced_split(dataset_dir, corrected_class) if self.auto_split_var.get() else self.dataset_split_var.get()
        save_dir = dataset_dir / split / corrected_class
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = "correction_%s_%s_%s.jpg" % (corrected_class, split, safe_filename_time())
        save_path = save_dir / filename
        image_bytes = self.latest_live_image_payload["image_bytes"]
        with open(save_path, "wb") as file:
            file.write(image_bytes)

        width = self._positive_int(self.latest_live_image_payload.get("width"))
        height = self._positive_int(self.latest_live_image_payload.get("height"))
        quality = self.inspect_capture_image(image_bytes, width, height)
        self.append_capture_metadata(
            dataset_dir=dataset_dir,
            save_path=save_path,
            split=split,
            class_name=corrected_class,
            port=self.selected_port() if self.port_var.get().strip() else "",
            baudrate=self.baud_var.get().strip(),
            width=quality["width"],
            height=quality["height"],
            byte_count=len(image_bytes),
            quality=quality,
        )
        self.append_correction_metadata(dataset_dir, save_path, split, corrected_class, quality)
        self.refresh_dataset_counts()
        self.log_capture(
            "纠错样本已保存：%s，正确类别=%s，已加入下一次训练。"
            % (save_path, defect_type_name(corrected_class))
        )
        self.status_var.set("状态：已保存纠错样本 %s" % defect_type_name(corrected_class))

    def append_correction_metadata(self, dataset_dir, save_path, split, corrected_class, quality):
        path = correction_metadata_path(dataset_dir)
        is_new_file = not path.exists()
        result = self.latest_detection_result if isinstance(self.latest_detection_result, dict) else {}
        defects = result.get("defects") or []
        primary = max(defects, key=lambda item: self.confidence_float(item), default={})
        predicted_class, _class_text, predicted_confidence = self.display_class_result(result)
        try:
            relative_path = str(save_path.relative_to(dataset_dir))
        except ValueError:
            relative_path = str(save_path)
        sidecar = {
            "correction_time": now_text(),
            "relative_path": relative_path,
            "split": split,
            "corrected_label": corrected_class,
            "predicted_label": predicted_class,
            "predicted_confidence": predicted_confidence,
            "primary_box": {
                "x": primary.get("x"),
                "y": primary.get("y"),
                "w": primary.get("w"),
                "h": primary.get("h"),
                "source": primary.get("source"),
                "review_source": primary.get("review_source"),
            },
            "defects": defects,
            "roi": result.get("roi") if isinstance(result.get("roi"), dict) else {},
            "lens": result.get("lens") if isinstance(result.get("lens"), dict) else {},
            "hard_negative_tags": self.correction_tags(corrected_class, predicted_class, primary),
            "quality": quality,
        }
        json_path = save_path.with_suffix(".correction.json")
        json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

        with open(path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow([
                    "correction_time",
                    "relative_path",
                    "split",
                    "corrected_label",
                    "predicted_label",
                    "predicted_confidence",
                    "box_x",
                    "box_y",
                    "box_w",
                    "box_h",
                    "source",
                    "tags",
                    "sidecar_json",
                ])
            writer.writerow([
                sidecar["correction_time"],
                relative_path,
                split,
                corrected_class,
                predicted_class,
                "" if predicted_confidence is None else predicted_confidence,
                primary.get("x", ""),
                primary.get("y", ""),
                primary.get("w", ""),
                primary.get("h", ""),
                primary.get("source", primary.get("review_source", "")),
                "|".join(sidecar["hard_negative_tags"]),
                str(json_path),
            ])

    def correction_tags(self, corrected_class, predicted_class, primary):
        tags = []
        if corrected_class == "normal":
            tags.append("normal_reflection_negative")
            if predicted_class in SUPPORTED_DEFECT_TYPES:
                tags.append("false_positive_%s" % predicted_class)
            if self.defect_source_looks_edge_related(primary):
                tags.append("frame_or_lens_edge_negative")
            else:
                tags.append("center_white_reflection_negative")
        elif corrected_class == "stain":
            tags.append("center_black_stain_positive")
            if predicted_class == "scratch":
                tags.append("scratch_to_stain_correction")
        elif corrected_class == "scratch":
            tags.append("center_thin_white_scratch_positive")
            if predicted_class == "stain":
                tags.append("stain_to_scratch_correction")
        return tags

    def defect_source_looks_edge_related(self, defect):
        if not isinstance(defect, dict):
            return False
        source = "%s %s" % (defect.get("source", ""), defect.get("review_source", ""))
        return "edge" in source or "frame" in source or "roi" in source

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

        mode_config = self.current_mode_config()
        summary_path = mode_config.training_summary
        if not summary_path.exists():
            self._set_text(
                self.training_result_text,
                "暂无训练结果。训练结束后这里会读取 %s、training_history.csv 和混淆矩阵文件。"
                % summary_path.name,
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

        mode_config = self.current_mode_config()
        dataset_dir = Path(self.dataset_dir_var.get())
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        python_exe = find_training_python()
        command = (
            '"%s" "%s" --dataset "%s" --output "%s" --artifact-prefix "%s" --summary-name "%s" --epochs 20 --image-size 128'
            % (
                python_exe,
                script,
                dataset_dir,
                DEFAULT_MODELS_DIR,
                "lens_defect" if mode_config.key == "lens" else "slide_defect",
                mode_config.training_summary.name,
            )
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开%s训练命令窗口" % mode_config.display_name)

    def run_yolo_seed_export_script(self):
        if not self.mode_uses_lens_postprocess():
            messagebox.showinfo("提示", "载玻片模式当前复用分类训练流程，不启用镜片 YOLO 种子导出。")
            return
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
        if not self.mode_uses_lens_postprocess():
            messagebox.showinfo("提示", "载玻片模式当前复用分类训练流程，不启用镜片 YOLO 训练。")
            return
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

    def run_yolo_seg_training_script(self):
        if not self.mode_uses_lens_postprocess():
            messagebox.showinfo("提示", "载玻片模式当前复用分类训练流程，不启用镜片 YOLO-seg 训练。")
            return
        script = find_yolo_training_script()
        if script is None:
            messagebox.showerror("找不到脚本", "未找到 training/train_yolo_from_seed.py")
            return

        python_exe = find_training_python()
        data_yaml = PROJECT_ROOT / "dataset_yolo_seed_refined" / "data.yaml"
        output_model = DEFAULT_YOLO_SEG_MODEL
        command = (
            '"%s" "%s" --data "%s" --output "%s" --base-model yolov8n-seg.pt --task segment --epochs 80 --image-size 640 --batch 4 --name lens_yolo_seg'
            % (python_exe, script, data_yaml, output_model)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开 YOLO-seg ONNX 训练窗口")

    def run_yolo_preview_script(self):
        if not self.mode_uses_lens_postprocess():
            messagebox.showinfo("提示", "载玻片模式当前复用分类训练流程，不启用镜片 YOLO 标注预览。")
            return
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

    def run_yolo_seg_seed_export_script(self):
        if not self.mode_uses_lens_postprocess():
            messagebox.showinfo("提示", "载玻片模式当前复用分类训练流程，不启用镜片 YOLO-seg 标注导出。")
            return
        script = find_yolo_seed_export_script()
        if script is None:
            messagebox.showerror("找不到脚本", "未找到 training/export_yolo_seed_labels.py")
            return

        dataset_dir = Path(self.dataset_dir_var.get())
        output_dir = PROJECT_ROOT / "dataset_yolo_seed_refined"
        python_exe = find_training_python()
        command = (
            '"%s" "%s" --dataset "%s" --output "%s" --copy-images --min-confidence 0.70 --label-format segment --clean-output'
            % (python_exe, script, dataset_dir, output_dir)
        )
        subprocess.Popen(["cmd", "/k", command], cwd=str(PROJECT_ROOT))
        self.status_var.set("状态：已打开 YOLO-seg 分割标注生成窗口")

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
        mode_config = self.current_mode_config()
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

        target_model = target_dir / mode_config.deploy_model_name
        target_labels = target_dir / mode_config.deploy_labels_name
        shutil.copy2(model_path, target_model)
        shutil.copy2(labels_path, target_labels)
        messagebox.showinfo("完成", "已复制：\n%s\n%s" % (target_model, target_labels))
        self.status_var.set("状态：%s模型已复制到 OpenMV N6" % mode_config.display_name)

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
        signature = raw_text or str(DEFAULT_YOLO_SEG_MODEL if DEFAULT_YOLO_SEG_MODEL.exists() else DEFAULT_YOLO_MODEL)
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
        candidates.append(DEFAULT_YOLO_SEG_MODEL)
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
        self.yolo_status_var.set("YOLO：已加载 %s（%s）" % (path.name, self.yolo_detector.task))
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
        if self.mode_uses_lens_postprocess():
            result = self.final_filter_result_for_current_image(
                result,
                self.latest_live_image_payload,
                clear_recent=False,
            )
            result = self.merge_recent_pc_defect_result(result)
            result = self.final_filter_result_for_current_image(result, self.latest_live_image_payload)
        result = self.stabilize_detection_result(result)
        self.update_result_ui(result)
        final_result = self.latest_detection_result if isinstance(self.latest_detection_result, dict) else result
        if final_result.get("conveyor_waiting"):
            self.status_var.set("状态：等待工件进入中心检测区")
        elif final_result.get("error"):
            self.add_history(final_result, raw_line)
            self.status_var.set("状态：OpenMV 错误：%s" % final_result.get("error"))
        else:
            self.add_history(final_result, raw_line)
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

    def final_filter_result_for_current_image(self, result, payload, clear_recent=True):
        filtered = self.filter_edge_defects_for_live_image(result, payload)
        if clear_recent and (not isinstance(filtered, dict) or not filtered.get("has_defect")):
            self.last_pc_defect_result = None
            self.last_pc_defect_time = 0.0
        return filtered

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
        roi, source_w, source_h = self.resolve_detection_roi_for_image(display_image, payload, result)
        if isinstance(result, dict) and isinstance(roi, dict):
            result = dict(result)
            result["roi"] = dict(roi)
            result["frame"] = {"w": int(source_w), "h": int(source_h)}
        mask = self.build_detection_mask(display_image, payload, result)
        if mask is None or cv2.countNonZero(mask) <= 0:
            refined = dict(result)
            refined["defects"] = []
            refined["edge_filter"] = {"removed": len(defects), "reason": "empty_lens_mask"}
            return self.normalize_detection_result(refined)
        edge_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        display_gray = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2GRAY)

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
        roi, _source_w, _source_h = self.resolve_detection_roi_for_image(display_image, payload, result)
        return roi

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

        if self.defect_is_safe_center_radial_stain(defect, source_w, source_h):
            return True
        if self.defect_is_safe_dark_star_line_stain(defect, source_w, source_h):
            return True
        if self.defect_is_safe_dark_x_stain(defect, source_w, source_h):
            return True

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
        defect_source = str(defect.get("source", ""))
        high_confidence_yolo_roi = (
            self.is_yolo_roi_source(defect_source)
            and self.confidence_float(defect) >= FAST_REVIEW_YOLO_ROI_HIGH_CONFIDENCE
        )
        high_confidence_yolo_roi_scratch = high_confidence_yolo_roi and defect_type == "scratch"
        safe_middle_dark_scratch = (
            defect_type == "scratch"
            and defect_source == "pc_fast_dark_scratch"
            and self.confidence_float(defect) >= 0.90
            and self.dark_stain_candidate_looks_like_weak_scratch_shadow(defect, mask)
        )
        strong_middle_stain = (
            defect_type == "stain"
            and (defect_source == "pc_fast_dark_stain" or self.is_yolo_roi_source(defect_source))
            and self.confidence_float(defect) >= 0.88
        )
        strong_middle_dark_stain = (
            defect_type == "stain"
            and defect_source == "pc_fast_dark_stain"
            and self.confidence_float(defect) >= 0.90
            and float(defect.get("local_dark_delta", 0) or 0) >= 12.0
            and float(defect.get("brightness_signed_delta", 0) or 0) <= -12.0
        )
        safe_middle_dark_stain = (
            defect_type == "stain"
            and self.dark_stain_candidate_is_safe_middle_defect(defect, roi, source_w, source_h)
        )
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
        if not high_confidence_yolo_roi_scratch and self.defect_looks_like_internal_reflection(
            defect,
            display_gray=display_gray,
            image_w=image_w,
            image_h=image_h,
            scaled_box=(x, y, w, h),
        ):
            return False
        if (
            not high_confidence_yolo_roi_scratch
            and not safe_middle_dark_stain
            and self.defect_looks_like_reflection_shadow(
            defect,
            display_gray=display_gray,
            image_w=image_w,
            image_h=image_h,
            scaled_box=(x, y, w, h),
            )
        ):
            return False
        if not safe_middle_dark_stain and not self.defect_center_is_inside_inner_roi(defect, roi, source_w, source_h):
            return False
        if not safe_middle_dark_stain and self.defect_box_near_roi_edge(defect, roi, source_w, source_h):
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
            if safe_middle_dark_stain:
                min_overlap = FAST_REVIEW_DARK_STAIN_SAFE_MIN_MASK_OVERLAP
            elif strong_middle_dark_stain and not touches_roi_edge:
                min_overlap = FAST_REVIEW_EDGE_FILTER_STAIN_STRONG_MIN_OVERLAP
        if overlap < min_overlap and not high_confidence_yolo_roi:
            return False

        if distance is None:
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        center_x = max(0, min(image_w - 1, int(x + w / 2)))
        center_y = max(0, min(image_h - 1, int(y + h / 2)))
        if mask[center_y, center_x] == 0:
            if not high_confidence_yolo_roi:
                return False
            if not self.defect_box_is_inside_roi(defect, roi, source_w, source_h):
                return False

        min_center_distance = max(6, int(min(image_w, image_h) * 0.018))
        if defect_type == "stain":
            stain_distance_ratio = (
                FAST_REVIEW_EDGE_FILTER_STAIN_STRONG_CENTER_DISTANCE_RATIO
                if strong_middle_stain
                else FAST_REVIEW_EDGE_FILTER_STAIN_CENTER_DISTANCE_RATIO
            )
            min_center_distance = max(min_center_distance, int(min(image_w, image_h) * stain_distance_ratio))
        elif defect_type == "scratch":
            min_center_distance = max(
                min_center_distance,
                int(min(image_w, image_h) * FAST_REVIEW_EDGE_FILTER_SCRATCH_CENTER_DISTANCE_RATIO),
            )
        if distance[center_y, center_x] < min_center_distance and not high_confidence_yolo_roi:
            if not safe_middle_dark_stain:
                return False

        if defect_type == "stain":
            inner = distance[y:y + h, x:x + w]
            if inner.size <= 0:
                return False
            stain_box_distance_ratio = (
                FAST_REVIEW_DARK_STAIN_SAFE_BOX_DISTANCE_RATIO
                if safe_middle_dark_stain
                else (
                    FAST_REVIEW_EDGE_FILTER_STAIN_STRONG_BOX_DISTANCE_RATIO
                    if strong_middle_dark_stain and not touches_roi_edge
                    else FAST_REVIEW_EDGE_FILTER_STAIN_BOX_DISTANCE_RATIO
                )
            )
            min_box_distance = max(4, int(min(image_w, image_h) * stain_box_distance_ratio))
            if float(np.percentile(inner, 60)) < min_box_distance:
                return False
        elif defect_type == "scratch":
            inner = distance[y:y + h, x:x + w]
            if inner.size <= 0:
                return False
            min_box_distance = max(5, int(min(image_w, image_h) * FAST_REVIEW_EDGE_FILTER_SCRATCH_BOX_DISTANCE_RATIO))
            if safe_middle_dark_scratch:
                min_box_distance = max(4, int(min(image_w, image_h) * 0.025))
            source = str(defect.get("source", ""))
            is_line_source = self.is_yolo_source(source) or source in (
                "pc_fast_review_line",
                "pc_fast_bright_scratch",
                "pc_fast_bright_crosshatch",
                "pc_fast_center_cross_scratch",
                "pc_fast_curvilinear_scratch",
            )
            percentile = 58 if is_line_source else 50
            if float(np.percentile(inner, percentile)) < min_box_distance and not high_confidence_yolo_roi:
                return False
            if touches_roi_edge and is_line_source:
                return False

        return True

    def dark_stain_candidate_is_safe_middle_defect(self, defect, roi, source_w, source_h):
        if not isinstance(defect, dict):
            return False
        if defect.get("type") != "stain" or str(defect.get("source", "")) != "pc_fast_dark_stain":
            return False
        if self.confidence_float(defect) < FAST_REVIEW_DARK_STAIN_SAFE_MIN_CONFIDENCE:
            return False
        if not self.defect_center_is_inside_middle_zone(defect, roi, source_w, source_h):
            return False

        area = float(defect.get("area", 0) or 0)
        length = float(defect.get("length", 0) or 0)
        density = float(defect.get("density", 0) or 0)
        if area < FAST_REVIEW_DARK_STAIN_SAFE_MIN_AREA or length < FAST_REVIEW_DARK_STAIN_SAFE_MIN_LENGTH:
            return False
        if density < FAST_REVIEW_DARK_STAIN_SAFE_MIN_DENSITY or density > FAST_REVIEW_DARK_STAIN_SAFE_MAX_DENSITY:
            return False

        signed_delta = float(defect.get("brightness_signed_delta", 0) or 0)
        local_dark = float(defect.get("local_dark_delta", 0) or 0)
        angle_groups = int(defect.get("angle_groups", 0) or 0)
        aspect_ratio = float(defect.get("aspect_ratio", 0) or 0)
        compact_dark_stain = (
            signed_delta <= FAST_REVIEW_DARK_STAIN_SAFE_STRONG_SIGNED_DELTA
            and local_dark >= FAST_REVIEW_DARK_STAIN_SAFE_STRONG_LOCAL_DELTA
            and angle_groups >= 1
        )
        star_shadow_stain = (
            signed_delta <= FAST_REVIEW_DARK_STAIN_SAFE_STAR_SIGNED_DELTA
            and area >= FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_AREA
            and length >= FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_LENGTH
            and aspect_ratio >= FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_ASPECT
            and density >= FAST_REVIEW_DARK_STAIN_SAFE_STAR_MIN_DENSITY
        )
        return compact_dark_stain or star_shadow_stain

    def defect_is_safe_dark_star_line_stain(self, defect, source_w, source_h):
        if defect.get("type") != "stain":
            return False
        if str(defect.get("source", "")) != "pc_fast_dark_star_stain_lines":
            return False
        if self.confidence_float(defect) < 0.90:
            return False
        if source_w <= 0 or source_h <= 0:
            return False

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        if w <= 0 or h <= 0:
            return False
        center_x = (x + w / 2.0) / float(source_w)
        center_y = (y + h / 2.0) / float(source_h)
        if center_x < 0.04 or center_x > FAST_REVIEW_DARK_STAR_LINE_MAX_X_RATIO:
            return False
        if center_y < 0.03 or center_y > FAST_REVIEW_DARK_STAR_LINE_MAX_Y_RATIO:
            return False
        if x <= source_w * 0.01 or y <= source_h * 0.01:
            return False
        if x + w >= source_w * 0.90 or y + h >= source_h * 0.90:
            return False

        line_count = int(defect.get("line_count", 0) or 0)
        angle_groups = int(defect.get("angle_groups", 0) or 0)
        total_length = float(defect.get("total_line_length", 0) or 0)
        if line_count < FAST_REVIEW_DARK_STAR_LINE_MIN_LINES:
            return False
        if angle_groups < FAST_REVIEW_DARK_STAR_LINE_MIN_ANGLE_GROUPS:
            return False
        if total_length < FAST_REVIEW_DARK_STAR_LINE_MIN_TOTAL_LENGTH:
            return False

        box_ratio = float(w * h) / float(max(1, source_w * source_h))
        if box_ratio < FAST_REVIEW_DARK_STAR_LINE_MIN_BOX_AREA_RATIO or box_ratio > FAST_REVIEW_DARK_STAR_LINE_MAX_BOX_AREA_RATIO:
            return False
        aspect_ratio = float(defect.get("aspect_ratio", 0) or 0)
        if aspect_ratio < FAST_REVIEW_DARK_STAR_LINE_MIN_ASPECT or aspect_ratio > FAST_REVIEW_DARK_STAR_LINE_MAX_ASPECT:
            return False
        dark_fraction = float(defect.get("density", 0) or 0)
        signed_delta = float(defect.get("brightness_signed_delta", 0) or 0)
        if (
            signed_delta > FAST_REVIEW_DARK_STAR_LINE_MIN_SIGNED_DELTA
            and dark_fraction < FAST_REVIEW_DARK_STAR_LINE_MIN_DARK_FRACTION
        ):
            return False
        return True

    def defect_is_safe_center_radial_stain(self, defect, source_w, source_h):
        if defect.get("type") != "stain":
            return False
        if str(defect.get("source", "")) != "pc_fast_center_radial_stain":
            return False
        if self.confidence_float(defect) < 0.90:
            return False
        if source_w <= 0 or source_h <= 0:
            return False

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        if w <= 0 or h <= 0:
            return False

        center_x = (x + w / 2.0) / float(source_w)
        center_y = (y + h / 2.0) / float(source_h)
        if center_x < 0.07 or center_x > 0.76:
            return False
        if center_y < 0.08 or center_y > 0.82:
            return False
        if x <= source_w * 0.015 or y <= source_h * 0.015:
            return False
        if x + w >= source_w * 0.90 or y + h >= source_h * 0.90:
            return False

        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        if aspect_ratio > 3.2:
            return False
        box_ratio = float(w * h) / float(max(1, source_w * source_h))
        if box_ratio < 0.0012 or box_ratio > FAST_REVIEW_CENTER_RADIAL_STAIN_MAX_BOX_RATIO:
            return False

        density = float(defect.get("density", 0) or 0)
        if density < 0.06 or density > 0.76:
            return False
        angle_groups = int(defect.get("angle_groups", 0) or 0)
        if angle_groups < FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_ANGLE_GROUPS:
            return False
        local_dark = float(defect.get("local_dark_delta", 0) or 0)
        signed_delta = float(defect.get("brightness_signed_delta", 0) or 0)
        if signed_delta > FAST_REVIEW_CENTER_RADIAL_STAIN_MAX_SIGNED_DELTA:
            return False
        if (
            local_dark < FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_LOCAL_DELTA
            and signed_delta > -FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_SIGNED_DELTA
        ):
            return False
        if local_dark < 12.0 and signed_delta > -18.0:
            return False
        return True

    def defect_is_safe_dark_x_stain(self, defect, source_w, source_h):
        if defect.get("type") != "stain":
            return False
        if str(defect.get("source", "")) != "pc_fast_dark_x_stain":
            return False
        if self.confidence_float(defect) < 0.88:
            return False
        if source_w <= 0 or source_h <= 0:
            return False

        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        if w <= 0 or h <= 0:
            return False

        center_x = (x + w / 2.0) / float(source_w)
        center_y = (y + h / 2.0) / float(source_h)
        if center_x < 0.22 or center_x > 0.76:
            return False
        if center_y < 0.24 or center_y > 0.82:
            return False
        if x <= source_w * 0.025 or y <= source_h * 0.025:
            return False
        if x + w >= source_w * 0.92 or y + h >= source_h * 0.92:
            return False

        angle_groups = int(defect.get("angle_groups", 0) or 0)
        line_count = int(defect.get("line_count", 0) or 0)
        total_length = float(defect.get("total_line_length", 0) or 0)
        density = float(defect.get("density", 0) or 0)
        signed_delta = float(defect.get("brightness_signed_delta", 0) or 0)
        local_dark = float(defect.get("local_dark_delta", 0) or 0)
        return (
            line_count >= FAST_REVIEW_DARK_X_STAIN_MIN_LINES
            and angle_groups >= FAST_REVIEW_DARK_X_STAIN_MIN_ANGLE_GROUPS
            and total_length >= FAST_REVIEW_DARK_X_STAIN_MIN_TOTAL_LENGTH
            and density >= FAST_REVIEW_DARK_X_STAIN_MIN_DARK_FRACTION
            and signed_delta <= FAST_REVIEW_DARK_X_STAIN_MAX_SIGNED_DELTA
            and local_dark >= 4.0
            and float(defect.get("aspect_ratio", 99) or 99) <= FAST_REVIEW_DARK_X_STAIN_MAX_ASPECT
        )

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
        if (
            not self.is_yolo_source(source)
            and source not in (
                "pc_fast_review_line",
                "pc_fast_bright_scratch",
                "pc_fast_bright_crosshatch",
                "pc_fast_center_cross_scratch",
                "pc_fast_curvilinear_scratch",
            )
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

        if self.is_yolo_source(source):
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

    def defect_looks_like_internal_reflection(self, defect, display_gray, image_w, image_h, scaled_box=None):
        if defect.get("type") != "scratch" or display_gray is None:
            return False
        source = str(defect.get("source", ""))
        if (
            not self.is_yolo_source(source)
            and source not in (
                "pc_fast_center_cross_scratch",
                "pc_fast_bright_scratch",
                "pc_fast_bright_crosshatch",
                "pc_fast_curvilinear_scratch",
            )
        ):
            return False

        if scaled_box is None:
            x = self._positive_int(defect.get("x"))
            y = self._positive_int(defect.get("y"))
            w = self._positive_int(defect.get("w"))
            h = self._positive_int(defect.get("h"))
        else:
            x, y, w, h = scaled_box
        if w <= 0 or h <= 0:
            return False

        x = max(0, min(int(x), image_w - 1))
        y = max(0, min(int(y), image_h - 1))
        w = max(1, min(int(w), image_w - x))
        h = max(1, min(int(h), image_h - y))
        box = display_gray[y:y + h, x:x + w]
        if box.size <= 0:
            return False

        pad = max(10, int(min(image_w, image_h) * 0.035))
        lx1 = max(0, x - pad)
        ly1 = max(0, y - pad)
        lx2 = min(image_w, x + w + pad)
        ly2 = min(image_h, y + h + pad)
        local = display_gray[ly1:ly2, lx1:lx2]
        if local.size <= 0:
            return False

        box_p98 = float(np.percentile(box, 98.0))
        local_mean = float(np.mean(local))
        local_p995 = float(np.percentile(local, 99.5))
        local_peak_delta = local_p995 - local_mean

        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        local_delta = float(defect.get("local_bright_delta", 0) or 0)
        line_count = int(defect.get("line_count", 0) or 0)
        slender_count = int(defect.get("slender_line_count", 0) or 0)
        slender_ratio = float(slender_count) / float(max(1, line_count))

        weak_wide_center_cross = (
            source == "pc_fast_center_cross_scratch"
            and aspect <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_ASPECT
            and local_delta <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_DELTA
            and slender_ratio <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_SLENDER_RATIO
        )
        strong_glare_near_box = (
            local_p995 >= FAST_REVIEW_REFLECTION_GLARE_MIN_GRAY
            and local_peak_delta >= FAST_REVIEW_REFLECTION_GLARE_MIN_LOCAL_DELTA
            and box_p98 >= FAST_REVIEW_GLARE_LINE_MIN_BRIGHTNESS
        )
        if weak_wide_center_cross and strong_glare_near_box:
            return True

        if self.is_yolo_source(source) and strong_glare_near_box and aspect <= 2.0:
            return True
        return False

    def defect_looks_like_reflection_shadow(self, defect, display_gray, image_w, image_h, scaled_box=None):
        if display_gray is None:
            return False
        source = str(defect.get("source", ""))
        if (
            not self.is_yolo_source(source)
            and source not in (
                "pc_fast_dark_stain",
                "pc_fast_review",
                "pc_fast_center_cross_scratch",
                "pc_fast_bright_scratch",
                "pc_fast_bright_crosshatch",
                "pc_fast_curvilinear_scratch",
            )
        ):
            return False

        if scaled_box is None:
            x = self._positive_int(defect.get("x"))
            y = self._positive_int(defect.get("y"))
            w = self._positive_int(defect.get("w"))
            h = self._positive_int(defect.get("h"))
        else:
            x, y, w, h = scaled_box
        if w <= 0 or h <= 0:
            return False

        x = max(0, min(int(x), image_w - 1))
        y = max(0, min(int(y), image_h - 1))
        w = max(1, min(int(w), image_w - x))
        h = max(1, min(int(h), image_h - y))
        box = display_gray[y:y + h, x:x + w]
        if box.size <= 0:
            return False

        pad = max(12, int(min(image_w, image_h) * 0.045))
        lx1 = max(0, x - pad)
        ly1 = max(0, y - pad)
        lx2 = min(image_w, x + w + pad)
        ly2 = min(image_h, y + h + pad)
        local = display_gray[ly1:ly2, lx1:lx2]
        if local.size <= 0:
            return False

        box_mean = float(np.mean(box))
        local_mean = float(np.mean(local))
        local_p995 = float(np.percentile(local, 99.5))
        glare_delta = local_p995 - local_mean
        signed_delta = box_mean - local_mean
        box_p98 = float(np.percentile(box, 98.0))
        defect_type = defect.get("type")
        if defect_type == "stain":
            local_dark = float(defect.get("local_dark_delta", 0) or 0)
            density = float(defect.get("density", 0) or 0)
            angle_groups = int(defect.get("angle_groups", 0) or 0)
            box_area = float(max(1, w * h))
            weak_small_dark_patch = (
                box_area <= float(image_w * image_h) * 0.025
                and abs(signed_delta) <= 1.8
                and local_dark <= 10.5
                and box_p98 >= 96.0
                and angle_groups <= 4
            )
            if weak_small_dark_patch:
                return True
            if (
                glare_delta >= FAST_REVIEW_REFLECTION_SHADOW_MIN_GLARE_DELTA * 1.30
                and signed_delta >= -7.0
                and local_dark <= 9.5
                and density <= 0.62
                and (angle_groups <= 1 or local_dark <= 6.0)
            ):
                return True
            if (
                glare_delta >= FAST_REVIEW_REFLECTION_SHADOW_MIN_GLARE_DELTA * 0.44
                and signed_delta >= -2.5
                and local_dark <= 3.5
                and angle_groups <= 0
            ):
                return True
            if (
                glare_delta >= FAST_REVIEW_REFLECTION_SHADOW_MIN_GLARE_DELTA * 0.82
                and signed_delta >= -6.0
                and local_dark <= 8.5
                and angle_groups <= 2
            ):
                return True
        elif defect_type == "scratch":
            local_delta = float(defect.get("local_bright_delta", 0) or 0)
            line_count = int(defect.get("line_count", 0) or 0)
            total_length = float(defect.get("total_line_length", 0) or 0)
            if (
                source == "pc_fast_center_cross_scratch"
                and glare_delta >= FAST_REVIEW_REFLECTION_SHADOW_MIN_GLARE_DELTA * 1.10
                and signed_delta >= 2.0
                and local_delta <= 28.0
            ):
                return True
            if (
                glare_delta >= FAST_REVIEW_REFLECTION_SHADOW_MIN_GLARE_DELTA
                and local_delta <= 16.0
                and box_p98 < FAST_REVIEW_REFLECTION_GLARE_MIN_GRAY
                and (
                    line_count <= FAST_REVIEW_REFLECTION_SCRATCH_MAX_LINE_COUNT
                    or total_length <= FAST_REVIEW_REFLECTION_SCRATCH_MAX_TOTAL_LENGTH
                )
            ):
                return True
        return False

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
        normal_text = self.current_mode_config().normal_result_text
        if result.get("conveyor_waiting"):
            return "waiting", "等待中心检测区", None
        if result.get("error"):
            return "error", "模型错误", None

        if not result.get("has_defect", False):
            return "normal", normal_text, None

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
        return "normal", normal_text, None

    def conveyor_gate_rect(self, image_w, image_h):
        return {
            "x": int(image_w * CONVEYOR_GATE_X_MIN),
            "y": int(image_h * CONVEYOR_GATE_Y_MIN),
            "w": max(1, int(image_w * (CONVEYOR_GATE_X_MAX - CONVEYOR_GATE_X_MIN))),
            "h": max(1, int(image_h * (CONVEYOR_GATE_Y_MAX - CONVEYOR_GATE_Y_MIN))),
        }

    def conveyor_wait_result(self, result, reason, reset_center=True):
        if reset_center:
            self.conveyor_centered_count = 0
            self.conveyor_fusion_results = []
        held = dict(result) if isinstance(result, dict) else {}
        held["defects"] = []
        held["summary"] = {defect_type: 0 for defect_type in SUMMARY_TYPES}
        held["defect_count"] = 0
        held["has_defect"] = False
        held["overall_level"] = "normal"
        held["conveyor_waiting"] = True
        held["conveyor_reason"] = reason
        return held

    def clone_detection_result(self, result):
        try:
            return json.loads(json.dumps(result, ensure_ascii=False))
        except Exception:
            return dict(result) if isinstance(result, dict) else {}

    def primary_defect_for_class(self, result, class_key):
        defects = [
            defect for defect in (result.get("defects") or [])
            if isinstance(defect, dict) and defect.get("type") == class_key
        ]
        if not defects:
            return None
        return max(defects, key=lambda item: self.confidence_float(item))

    def fused_conveyor_result_from_buffer(self, current_result):
        frame_count = len(self.conveyor_fusion_results)
        if frame_count < CONVEYOR_FUSION_MIN_FRAMES:
            return self.conveyor_wait_result(current_result, "fusion_wait_frames", reset_center=False)

        scores = {"normal": 0.0, "scratch": 0.0, "stain": 0.0}
        votes = {"normal": 0, "scratch": 0, "stain": 0}
        best_by_class = {}
        for item in self.conveyor_fusion_results:
            class_key, _class_text, confidence = self.display_class_result(item)
            if class_key not in scores:
                continue
            votes[class_key] += 1
            confidence_value = 0.0 if confidence is None else max(0.0, min(1.0, float(confidence)))
            scores[class_key] += 1.0 + confidence_value
            if class_key in SUPPORTED_DEFECT_TYPES:
                defect = self.primary_defect_for_class(item, class_key)
                if defect is not None and self.confidence_float(defect) >= self.confidence_float(best_by_class.get(class_key, {})):
                    best_by_class[class_key] = defect

        defect_candidates = [key for key in SUPPORTED_DEFECT_TYPES if votes[key] > 0]
        if defect_candidates:
            best_defect_class = max(defect_candidates, key=lambda key: (votes[key], scores[key]))
            best_defect = best_by_class.get(best_defect_class)
            best_confidence = self.confidence_float(best_defect)
            if (
                votes[best_defect_class] < CONVEYOR_FUSION_DEFECT_MIN_VOTES
                and best_confidence < CONVEYOR_FUSION_STRONG_CONFIDENCE
            ):
                return self.conveyor_wait_result(current_result, "fusion_wait_defect_votes", reset_center=False)
            fused = self.clone_detection_result(current_result)
            fused["defects"] = [dict(best_defect)] if isinstance(best_defect, dict) else []
            fused["conveyor_fusion"] = {
                "frames": frame_count,
                "votes": votes,
                "scores": {key: round(value, 3) for key, value in scores.items()},
            }
            return self.normalize_detection_result(fused)

        if votes["normal"] < CONVEYOR_FUSION_NORMAL_MIN_VOTES:
            return self.conveyor_wait_result(current_result, "fusion_wait_normal_votes", reset_center=False)
        fused = self.clone_detection_result(current_result)
        fused["defects"] = []
        fused["conveyor_fusion"] = {
            "frames": frame_count,
            "votes": votes,
            "scores": {key: round(value, 3) for key, value in scores.items()},
        }
        return self.normalize_detection_result(fused)

    def apply_conveyor_frame_fusion(self, result):
        if not self.conveyor_center_gate_var.get() or not isinstance(result, dict):
            self.conveyor_fusion_results = []
            return result
        if result.get("conveyor_waiting") or result.get("error"):
            if result.get("conveyor_waiting") and result.get("conveyor_reason") != "roi_wait_stable":
                self.conveyor_fusion_results = []
            return result
        self.conveyor_fusion_results.append(self.clone_detection_result(result))
        self.conveyor_fusion_results = self.conveyor_fusion_results[-CONVEYOR_FUSION_MAX_FRAMES:]
        return self.fused_conveyor_result_from_buffer(result)

    def roi_is_inside_conveyor_gate(self, roi, image_w, image_h):
        if not isinstance(roi, dict) or image_w <= 0 or image_h <= 0:
            return False
        x = self._positive_int(roi.get("x"))
        y = self._positive_int(roi.get("y"))
        w = self._positive_int(roi.get("w"))
        h = self._positive_int(roi.get("h"))
        if w <= 0 or h <= 0:
            return False

        frame_area = float(max(1, image_w * image_h))
        area_ratio = float(w * h) / frame_area
        if area_ratio < 0.035 or area_ratio > 0.74:
            return False

        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        if aspect_ratio > 3.8:
            return False

        center_x = (x + w / 2.0) / float(image_w)
        center_y = (y + h / 2.0) / float(image_h)
        return (
            CONVEYOR_ROI_CENTER_X_MIN <= center_x <= CONVEYOR_ROI_CENTER_X_MAX
            and CONVEYOR_ROI_CENTER_Y_MIN <= center_y <= CONVEYOR_ROI_CENTER_Y_MAX
        )

    def defect_center_is_inside_conveyor_gate(self, defect, image_w, image_h):
        if not isinstance(defect, dict) or image_w <= 0 or image_h <= 0:
            return False
        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        if w <= 0 or h <= 0:
            return False
        center_x = (x + w / 2.0) / float(image_w)
        center_y = (y + h / 2.0) / float(image_h)
        return (
            CONVEYOR_GATE_X_MIN <= center_x <= CONVEYOR_GATE_X_MAX
            and CONVEYOR_GATE_Y_MIN <= center_y <= CONVEYOR_GATE_Y_MAX
        )

    def defect_center_is_inside_workpiece_roi(self, defect, roi, source_w, source_h):
        if not isinstance(defect, dict) or not isinstance(roi, dict) or source_w <= 0 or source_h <= 0:
            return False
        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        rx = self._positive_int(roi.get("x"))
        ry = self._positive_int(roi.get("y"))
        rw = self._positive_int(roi.get("w"))
        rh = self._positive_int(roi.get("h"))
        if w <= 0 or h <= 0 or rw <= 0 or rh <= 0:
            return False
        margin = max(6, int(min(rw, rh) * 0.08))
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        return (
            rx - margin <= center_x <= rx + rw + margin
            and ry - margin <= center_y <= ry + rh + margin
        )

    def conveyor_roi_source_is_fallback(self, roi):
        if not isinstance(roi, dict):
            return True
        source = str(roi.get("source", "") or "").strip().lower()
        return source in CONVEYOR_FALLBACK_ROI_SOURCES

    def conveyor_roi_requires_live_confirmation(self, roi, source_w, source_h):
        roi = self.valid_roi_or_none(roi, source_w, source_h)
        if roi is None or self.roi_is_full_frame(roi, source_w, source_h):
            return True
        if self.conveyor_roi_source_is_fallback(roi):
            return True
        try:
            confidence = float(roi.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return confidence < CONVEYOR_MIN_WORKPIECE_CONFIDENCE

    def conveyor_live_workpiece_roi(self, source_w, source_h):
        if Image is None or cv2 is None or np is None:
            return None
        payload = self.latest_live_image_payload if isinstance(self.latest_live_image_payload, dict) else {}
        image_bytes = payload.get("image_bytes")
        if not image_bytes:
            return None
        payload_key = "%s:%s" % (payload.get("receive_time", ""), payload.get("byte_count", len(image_bytes)))

        if payload_key != self.conveyor_workpiece_payload_key:
            detected = None
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    roi = self.detect_lens_roi_from_image(image)
                    if roi is None:
                        roi = self.detect_conveyor_workpiece_roi_from_image(image)
                    roi = self.valid_roi_or_none(roi, image.width, image.height)
                    if (
                        roi is not None
                        and not self.roi_is_full_frame(roi, image.width, image.height)
                        and self.roi_is_inside_conveyor_gate(roi, image.width, image.height)
                    ):
                        roi["source"] = "pc_live_workpiece"
                        detected = {"roi": roi, "w": image.width, "h": image.height}
            except Exception:
                detected = None
            self.conveyor_workpiece_payload_key = payload_key
            self.conveyor_workpiece_roi = detected

        cached = self.conveyor_workpiece_roi
        if not isinstance(cached, dict):
            return None
        roi = cached.get("roi")
        image_w = self._positive_int(cached.get("w"))
        image_h = self._positive_int(cached.get("h"))
        if not isinstance(roi, dict) or image_w <= 0 or image_h <= 0 or source_w <= 0 or source_h <= 0:
            return None
        scaled = {
            "x": int(self._positive_int(roi.get("x")) * float(source_w) / float(image_w)),
            "y": int(self._positive_int(roi.get("y")) * float(source_h) / float(image_h)),
            "w": int(self._positive_int(roi.get("w")) * float(source_w) / float(image_w)),
            "h": int(self._positive_int(roi.get("h")) * float(source_h) / float(image_h)),
            "source": "pc_live_workpiece",
            "confidence": 1.0,
            "found": True,
        }
        return self.valid_roi_or_none(scaled, source_w, source_h)

    def apply_conveyor_center_gate(self, result):
        if not self.conveyor_center_gate_var.get() or not isinstance(result, dict):
            self.conveyor_centered_count = 0
            self.conveyor_fusion_results = []
            return result
        if result.get("error"):
            return result

        payload = self.latest_live_image_payload if isinstance(self.latest_live_image_payload, dict) else {}
        frame = result.get("frame") if isinstance(result.get("frame"), dict) else {}
        source_w = self._positive_int(frame.get("w")) or self._positive_int(payload.get("width"))
        source_h = self._positive_int(frame.get("h")) or self._positive_int(payload.get("height"))
        if source_w <= 0 or source_h <= 0:
            return result

        requires_workpiece = self.mode_uses_lens_postprocess()
        lens = result.get("lens") if isinstance(result.get("lens"), dict) else None
        roi = lens if isinstance(lens, dict) else None
        if roi is None:
            roi = result.get("roi") if isinstance(result.get("roi"), dict) else None
        roi = self.valid_roi_or_none(roi, source_w, source_h)

        lens_reported_missing = bool(isinstance(lens, dict) and lens.get("found") is False)
        if requires_workpiece and (
            lens_reported_missing
            or self.conveyor_roi_requires_live_confirmation(roi, source_w, source_h)
        ):
            live_roi = self.conveyor_live_workpiece_roi(source_w, source_h)
            if live_roi is None:
                reason = "no_lens_found" if lens_reported_missing else "no_live_workpiece"
                return self.conveyor_wait_result(result, reason)
            roi = live_roi

        if roi is not None and not self.roi_is_full_frame(roi, source_w, source_h):
            if not self.roi_is_inside_conveyor_gate(roi, source_w, source_h):
                return self.conveyor_wait_result(result, "roi_outside_center_gate")
            self.conveyor_centered_count += 1
            if self.conveyor_centered_count < CONVEYOR_CENTER_STABLE_FRAMES:
                return self.conveyor_wait_result(result, "roi_wait_stable", reset_center=False)
        elif requires_workpiece:
            return self.conveyor_wait_result(result, "no_live_workpiece")
        else:
            self.conveyor_centered_count = min(
                CONVEYOR_CENTER_STABLE_FRAMES,
                self.conveyor_centered_count + 1,
            )

        defects = result.get("defects") or []
        if not defects:
            return result
        gated_defects = [
            defect for defect in defects
            if (
                self.defect_center_is_inside_conveyor_gate(defect, source_w, source_h)
                and (
                    not requires_workpiece
                    or self.defect_center_is_inside_workpiece_roi(defect, roi, source_w, source_h)
                )
            )
        ]
        if len(gated_defects) == len(defects):
            return result
        filtered = dict(result)
        filtered["defects"] = gated_defects
        filtered["conveyor_filtered_defects"] = len(defects) - len(gated_defects)
        return self.normalize_detection_result(filtered)

    def stabilize_detection_result(self, result):
        result = self.apply_conveyor_center_gate(result)
        if isinstance(result, dict) and result.get("has_defect"):
            locked_defects = []
            changed_by_black_lock = False
            for defect in result.get("defects") or []:
                locked = self.apply_black_stain_class_lock(defect)
                locked_defects.append(locked)
                changed_by_black_lock = changed_by_black_lock or locked is not defect
            if changed_by_black_lock:
                result = dict(result)
                result["defects"] = locked_defects
                result = self.normalize_detection_result(result)

        result = self.apply_conveyor_frame_fusion(result)

        class_key, _class_text, _confidence = self.display_class_result(result)
        if self.stable_detection_result is None:
            self.stable_detection_result = result
            self.pending_detection_class = None
            self.pending_detection_count = 0
            self.remember_black_stain_detection(result)
            return result

        stable_key, _stable_text, _stable_confidence = self.display_class_result(self.stable_detection_result)
        if class_key == stable_key:
            self.stable_detection_result = result
            self.pending_detection_class = None
            self.pending_detection_count = 0
            self.remember_black_stain_detection(result)
            return result

        if class_key != self.pending_detection_class:
            self.pending_detection_class = class_key
            self.pending_detection_count = 1
        else:
            self.pending_detection_count += 1

        if class_key == "normal":
            required_count = HOST_NORMAL_CONFIRM_UPDATES
        elif stable_key in SUPPORTED_DEFECT_TYPES and class_key in SUPPORTED_DEFECT_TYPES:
            required_count = HOST_DEFECT_CLASS_CONFIRM_UPDATES
            if stable_key == "stain" and class_key == "scratch":
                required_count = HOST_BLACK_STAIN_TO_SCRATCH_CONFIRM_UPDATES
        else:
            required_count = HOST_DEFECT_CONFIRM_UPDATES
        if self.pending_detection_count >= required_count:
            self.stable_detection_result = result
            self.pending_detection_class = None
            self.pending_detection_count = 0
            self.remember_black_stain_detection(result)
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
                "waiting": "#1E6B8C",
            }.get(class_key, "#333333")
            self.class_result_label.configure(fg=color)
        self.has_defect_var.set("是否检测到缺陷：%s" % ("是" if has_defect else "否"))
        if class_key == "waiting":
            self.has_defect_var.set("是否检测到缺陷：等待检测")
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
            if self.latest_detection_result is not result:
                return
        self.maybe_send_mcu_result(result)

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
                if self.mode_uses_lens_postprocess():
                    self.apply_yolo_to_live_image(display_image, payload)
                    self.apply_fast_cv_review_to_live_image(display_image, payload)
                    if not self.low_latency_mode_var.get():
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

    def detection_center_distance_ratio(self, first, second):
        if not isinstance(first, dict) or not isinstance(second, dict):
            return 1.0
        ax = float(self._positive_int(first.get("x"))) + float(self._positive_int(first.get("w"))) / 2.0
        ay = float(self._positive_int(first.get("y"))) + float(self._positive_int(first.get("h"))) / 2.0
        bx = float(self._positive_int(second.get("x"))) + float(self._positive_int(second.get("w"))) / 2.0
        by = float(self._positive_int(second.get("y"))) + float(self._positive_int(second.get("h"))) / 2.0
        aw = float(max(1, self._positive_int(first.get("w"))))
        ah = float(max(1, self._positive_int(first.get("h"))))
        bw = float(max(1, self._positive_int(second.get("w"))))
        bh = float(max(1, self._positive_int(second.get("h"))))
        scale = max(aw, ah, bw, bh, 1.0)
        return max(abs(ax - bx), abs(ay - by)) / scale

    def expand_detection_box(self, detection, ratio, image_w, image_h):
        if not isinstance(detection, dict):
            return None
        x = self._positive_int(detection.get("x"))
        y = self._positive_int(detection.get("y"))
        w = self._positive_int(detection.get("w"))
        h = self._positive_int(detection.get("h"))
        if w <= 0 or h <= 0:
            return None
        grow_x = max(2, int(w * ratio))
        grow_y = max(2, int(h * ratio))
        left = max(0, x - grow_x)
        top = max(0, y - grow_y)
        right = min(max(1, image_w), x + w + grow_x)
        bottom = min(max(1, image_h), y + h + grow_y)
        return {
            "x": int(left),
            "y": int(top),
            "w": int(max(1, right - left)),
            "h": int(max(1, bottom - top)),
        }

    def confidence_float(self, detection, default=0.0):
        try:
            return float(detection.get("confidence", default) or default)
        except (AttributeError, TypeError, ValueError):
            return float(default)

    def detection_source(self, detection_or_source):
        if isinstance(detection_or_source, dict):
            return str(detection_or_source.get("source", ""))
        return str(detection_or_source or "")

    def is_yolo_source(self, detection_or_source):
        return self.detection_source(detection_or_source).startswith("yolo_onnx")

    def is_yolo_roi_source(self, detection_or_source):
        return self.detection_source(detection_or_source).startswith("yolo_onnx_roi")

    def detection_area_ratio(self, detection, image_w, image_h):
        if not isinstance(detection, dict):
            return 0.0
        width = self._positive_int(detection.get("w"))
        height = self._positive_int(detection.get("h"))
        frame_area = float(max(1, int(image_w) * int(image_h)))
        return float(width * height) / frame_area

    def review_detection_is_strong_scratch(self, detection):
        if not isinstance(detection, dict) or detection.get("type") != "scratch":
            return False
        confidence = self.confidence_float(detection)
        if confidence < 0.72:
            return False
        source = self.detection_source(detection)
        line_source = source in (
            "pc_fast_review",
            "pc_fast_review_line",
            "pc_fast_bright_scratch",
            "pc_fast_bright_crosshatch",
            "pc_fast_center_cross_scratch",
            "pc_fast_curvilinear_scratch",
        )
        aspect_ratio = float(detection.get("aspect_ratio", 0) or 0)
        line_count = int(detection.get("line_count", 0) or 0)
        total_line_length = float(detection.get("total_line_length", detection.get("length", 0)) or 0)
        fill_ratio = float(detection.get("fill_ratio", 0) or 0)
        strong_internal = bool(detection.get("strong_internal_scratch"))
        if strong_internal and aspect_ratio >= 1.6:
            return True
        if (
            line_source
            and aspect_ratio >= 1.85
            and total_line_length >= 60.0
            and (line_count >= 3 or total_line_length >= 110.0)
            and (fill_ratio <= 0.55 if fill_ratio > 0 else True)
        ):
            return True
        return aspect_ratio >= 3.2 and total_line_length >= 82.0

    def review_detection_is_strong_stain(self, detection):
        if not isinstance(detection, dict) or detection.get("type") != "stain":
            return False
        if detection.get("black_stain_guard") or detection.get("black_radial_stain"):
            return True
        source = self.detection_source(detection)
        confidence = self.confidence_float(detection)
        density = float(detection.get("density", 0) or 0)
        signed_delta = float(detection.get("brightness_signed_delta", 0) or 0)
        local_dark_delta = float(detection.get("local_dark_delta", 0) or 0)
        aspect_ratio = float(detection.get("aspect_ratio", 0) or 0)
        area = int(detection.get("area", 0) or 0)
        if (
            source in (
                "pc_fast_dark_star_stain",
                "pc_fast_dark_star_stain_lines",
                "pc_fast_center_radial_stain",
                "pc_fast_dark_stain",
                "pc_fast_dark_x_stain",
            )
            and confidence >= 0.80
        ):
            return True
        if (
            source == "pc_fast_review"
            and confidence >= 0.84
            and area >= 1600
            and aspect_ratio <= 4.6
        ):
            return True
        return (
            confidence >= 0.84
            and aspect_ratio <= 4.8
            and (density >= 0.16 or local_dark_delta >= 8.0 or signed_delta <= -6.0)
        )

    def review_detection_should_override_yolo(self, yolo_detection, review_detection, iou, center_distance):
        if not isinstance(yolo_detection, dict) or not isinstance(review_detection, dict):
            return False
        yolo_confidence = self.confidence_float(yolo_detection)
        review_confidence = self.confidence_float(review_detection)
        if (
            yolo_confidence <= FAST_REVIEW_YOLO_BOX_LOCK_MIN_CONFIDENCE
            and review_confidence >= yolo_confidence + 0.08
        ):
            return True
        if (
            iou >= FAST_REVIEW_YOLO_BOX_LOCK_MIN_IOU
            or center_distance <= FAST_REVIEW_YOLO_BOX_LOCK_MAX_CENTER_DISTANCE_RATIO
        ):
            return False
        if review_detection.get("type") == "scratch":
            return self.review_detection_is_strong_scratch(review_detection) and review_confidence >= max(0.74, yolo_confidence + 0.02)
        if review_detection.get("type") == "stain":
            return self.review_detection_is_strong_stain(review_detection) and (
                review_confidence >= max(0.80, yolo_confidence - 0.04)
                or yolo_detection.get("type") == "scratch"
            )
        return False

    def review_box_should_replace_locked_box(self, yolo_detection, review_detection, iou, center_distance):
        if not isinstance(yolo_detection, dict) or not isinstance(review_detection, dict):
            return False
        yolo_area = max(1, self._positive_int(yolo_detection.get("w")) * self._positive_int(yolo_detection.get("h")))
        review_area = max(1, self._positive_int(review_detection.get("w")) * self._positive_int(review_detection.get("h")))
        if self.review_detection_is_strong_scratch(review_detection):
            return (
                center_distance >= 0.04
                or iou <= 0.38
                or review_area <= yolo_area * 0.78
                or self.confidence_float(yolo_detection) <= FAST_REVIEW_YOLO_BOX_LOCK_MIN_CONFIDENCE
            )
        if self.review_detection_is_strong_stain(review_detection):
            return (
                center_distance >= 0.03
                or iou <= 0.44
                or review_area <= yolo_area * 0.88
                or self.confidence_float(yolo_detection) <= FAST_REVIEW_YOLO_BOX_LOCK_MIN_CONFIDENCE
            )
        return False

    def apply_review_geometry_to_detection(self, merged_detection, review_detection):
        if not isinstance(merged_detection, dict) or not isinstance(review_detection, dict):
            return merged_detection
        refined = dict(merged_detection)
        for key in (
            "x",
            "y",
            "w",
            "h",
            "area",
            "length",
            "aspect_ratio",
            "mask_polygon",
            "line_count",
            "total_line_length",
            "fill_ratio",
            "local_signed_delta",
            "local_bright_delta",
            "local_dark_delta",
            "density",
            "brightness_signed_delta",
            "angle_groups",
            "slender_line_count",
            "star_scratch",
            "crosshatch_scratch",
            "strong_internal_scratch",
        ):
            if key in review_detection:
                refined[key] = review_detection.get(key)
        refined["box_refined_by_review"] = True
        refined["review_box_source"] = review_detection.get("source", "")
        return refined

    def should_run_fast_review_for_yolo(self, result):
        if not isinstance(result, dict):
            return True
        defects = [item for item in (result.get("defects") or []) if isinstance(item, dict)]
        if not defects:
            return True
        best = max(defects, key=lambda item: self.confidence_float(item))
        confidence = self.confidence_float(best)
        if confidence < 0.58:
            return True
        frame = result.get("frame") or {}
        frame_w = self._positive_int(frame.get("w"))
        frame_h = self._positive_int(frame.get("h"))
        area_ratio = self.detection_area_ratio(best, frame_w or 1, frame_h or 1)
        defect_type = best.get("type")
        if defect_type == "stain":
            if self.review_detection_is_strong_stain(best):
                return False
            return confidence < 0.82
        if defect_type == "scratch":
            if best.get("black_stain_guard"):
                return False
            if self.review_detection_is_strong_scratch(best):
                return False
            aspect_ratio = float(best.get("aspect_ratio", 0) or 0)
            if confidence >= 0.84 and aspect_ratio >= 2.6 and area_ratio <= 0.08:
                return False
            return True
        return confidence < 0.78

    def merge_yolo_locked_box_with_review(self, yolo_detection, review_detection, image_w, image_h):
        if (
            not isinstance(yolo_detection, dict)
            or not self.is_yolo_source(yolo_detection)
        ):
            return review_detection
        if not isinstance(review_detection, dict):
            return yolo_detection
        yolo_confidence = self.confidence_float(yolo_detection)
        review_confidence = self.confidence_float(review_detection)
        expanded_yolo = self.expand_detection_box(
            yolo_detection,
            FAST_REVIEW_YOLO_BOX_LOCK_EXPAND_RATIO,
            image_w,
            image_h,
        )
        if expanded_yolo is None:
            return review_detection
        iou = self.fast_review_iou(expanded_yolo, review_detection)
        center_distance = self.detection_center_distance_ratio(yolo_detection, review_detection)
        if (
            iou < FAST_REVIEW_YOLO_BOX_LOCK_MIN_IOU
            and center_distance > FAST_REVIEW_YOLO_BOX_LOCK_MAX_CENTER_DISTANCE_RATIO
        ):
            if self.review_detection_should_override_yolo(yolo_detection, review_detection, iou, center_distance):
                overridden = dict(review_detection)
                overridden["pc_fast_review"] = True
                overridden["yolo_override_source"] = yolo_detection.get("source", "")
                overridden["yolo_override_confidence"] = round(yolo_confidence, 2)
                overridden["review_iou"] = round(iou, 3)
                overridden["review_center_distance"] = round(center_distance, 3)
                return self.apply_black_stain_class_lock(overridden)
            return yolo_detection

        merged = dict(yolo_detection)
        yolo_type = yolo_detection.get("type")
        review_type = review_detection.get("type")
        review_source = self.detection_source(review_detection)
        if (
            yolo_type == "scratch"
            and review_type == "stain"
            and self.review_detection_is_strong_stain(review_detection)
            and review_confidence >= 0.80
        ):
            merged["type"] = "stain"
        if (
            review_type in SUPPORTED_DEFECT_TYPES
            and review_type != yolo_type
            and merged.get("type") == yolo_type
            and review_confidence >= yolo_confidence + FAST_REVIEW_YOLO_BOX_LOCK_CLASS_MARGIN
        ):
            merged["type"] = review_type
        if (
            merged.get("type") == review_type
            and self.review_box_should_replace_locked_box(yolo_detection, review_detection, iou, center_distance)
        ):
            merged = self.apply_review_geometry_to_detection(merged, review_detection)
        merged["confidence"] = round(max(yolo_confidence, min(0.96, review_confidence)), 2)
        merged["level"] = review_detection.get("level", yolo_detection.get("level", "medium"))
        merged["pc_fast_review"] = True
        merged["review_source"] = review_detection.get("source", "")
        merged["review_iou"] = round(iou, 3)
        merged["review_center_distance"] = round(center_distance, 3)
        merged = self.apply_black_stain_class_lock(merged)
        return merged

    def detection_looks_like_locked_black_stain(self, detection):
        if not isinstance(detection, dict):
            return False
        if detection.get("type") != "stain":
            return False
        source = str(detection.get("source", ""))
        review_source = str(detection.get("review_source", ""))
        if detection.get("black_stain_guard") or detection.get("black_radial_stain"):
            return True
        return source in (
            "pc_fast_dark_star_stain",
            "pc_fast_dark_star_stain_lines",
            "pc_fast_center_radial_stain",
            "pc_fast_dark_x_stain",
        ) or review_source in (
            "pc_fast_dark_star_stain",
            "pc_fast_dark_star_stain_lines",
            "pc_fast_center_radial_stain",
            "pc_yolo_black_stain_guard",
            "pc_fast_dark_x_stain",
        )

    def remember_black_stain_detection(self, result):
        if not isinstance(result, dict) or not result.get("has_defect"):
            return
        for defect in result.get("defects") or []:
            if self.detection_looks_like_locked_black_stain(defect):
                self.last_black_stain_result = dict(defect)
                self.last_black_stain_time = time.monotonic()
                return

    def apply_black_stain_class_lock(self, detection):
        if not isinstance(detection, dict) or detection.get("type") != "scratch":
            return detection
        recent = self.last_black_stain_result
        if not isinstance(recent, dict):
            return detection
        if time.monotonic() - float(self.last_black_stain_time or 0.0) > FAST_REVIEW_BLACK_STAIN_LOCK_SECONDS:
            return detection
        iou = self.fast_review_iou(recent, detection)
        center_distance = self.detection_center_distance_ratio(recent, detection)
        if (
            iou < FAST_REVIEW_BLACK_STAIN_LOCK_MIN_IOU
            and center_distance > FAST_REVIEW_BLACK_STAIN_LOCK_MAX_CENTER_DISTANCE_RATIO
        ):
            return detection
        locked = dict(detection)
        locked["type"] = "stain"
        locked["class_locked_from"] = "scratch"
        locked["review_source"] = "pc_black_stain_temporal_lock"
        locked["black_stain_lock"] = True
        locked["confidence"] = max(self.confidence_float(detection), min(0.96, self.confidence_float(recent, 0.90)))
        locked["level"] = recent.get("level", detection.get("level", "medium"))
        if "density" in recent:
            locked["density"] = recent.get("density")
        if "local_dark_delta" in recent:
            locked["local_dark_delta"] = recent.get("local_dark_delta")
        return locked

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
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        detection = self.fast_cv_detect_defect(image, mask, gray=gray)
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
        current_yolo_defects = [
            defect for defect in (self.latest_detection_result.get("defects") or [])
            if isinstance(defect, dict) and self.is_yolo_source(defect)
        ]
        if current_yolo_defects:
            best_yolo = max(current_yolo_defects, key=lambda item: self.confidence_float(item))
            detection = self.merge_yolo_locked_box_with_review(
                best_yolo,
                detection,
                image.width,
                image.height,
            )
            refined["yolo_box_locked"] = True
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
        min_interval = LOW_LATENCY_YOLO_MIN_INTERVAL_SECONDS if self.low_latency_mode_var.get() else YOLO_MIN_INTERVAL_SECONDS
        if now - self.last_yolo_infer_time < min_interval:
            return
        detector = self.get_yolo_detector()
        if detector is None:
            return

        self.last_yolo_payload_key = payload_key
        self.last_yolo_infer_time = now
        base = self.latest_detection_result if isinstance(self.latest_detection_result, dict) else {}
        inference_roi = self.yolo_inference_roi_for_image(image, payload, base)
        if inference_roi is not None:
            input_size = YOLO_ROI_HIGH_RES_INPUT_SIZE if self.yolo_high_res_roi_var.get() else None
            detections = detector.detect_roi(image, inference_roi, input_size=input_size)
            yolo_mode = "roi"
            if self.should_run_yolo_tile_review(detections):
                tile_detections = detector.detect_roi_tiled(image, inference_roi, input_size=input_size)
                if tile_detections:
                    detections = self.merge_yolo_detections(detections, tile_detections)
                    yolo_mode = "roi_tile_review"
            if detections:
                self.yolo_roi_miss_count = 0
            else:
                self.yolo_roi_miss_count += 1
                if self.yolo_roi_miss_count >= YOLO_FULL_FRAME_FALLBACK_AFTER_ROI_MISSES:
                    full_detections = detector.detect(image)
                    if full_detections:
                        detections = full_detections
                        yolo_mode = "roi_full_fallback"
                    self.yolo_roi_miss_count = 0
        else:
            detections = detector.detect(image)
            yolo_mode = "full"
            self.yolo_roi_miss_count = 0
        if not detections:
            return
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        detections = self.correct_yolo_black_stain_classes(image, detections, gray=gray)

        refined = dict(base)
        refined["frame"] = {"w": image.width, "h": image.height}
        refined["roi"] = self.yolo_filter_roi_for_image(image, payload, base, inference_roi)
        refined["defects"] = detections
        refined["model"] = "yolo_onnx"
        refined["yolo"] = {
            "enabled": True,
            "model_path": str(detector.model_path),
            "mode": yolo_mode,
            "inference_roi": inference_roi or {},
            "high_res_roi": bool(inference_roi is not None and self.yolo_high_res_roi_var.get()),
            "tile_review": yolo_mode == "roi_tile_review",
        }
        refined = self.normalize_detection_result(refined)
        refined = self.filter_edge_defects_for_image(refined, payload, image)
        fallback_detection = None
        if self.should_run_fast_review_for_yolo(refined):
            mask = self.build_detection_mask(image, payload, refined)
            fallback_detection = self.fast_cv_detect_defect(image, mask, gray=gray)
        if refined.get("has_defect"):
            yolo_defects = [
                defect for defect in (refined.get("defects") or [])
                if isinstance(defect, dict) and self.is_yolo_source(defect)
            ]
            if yolo_defects and fallback_detection is not None:
                best_yolo = max(yolo_defects, key=lambda item: self.confidence_float(item))
                merged_detection = self.merge_yolo_locked_box_with_review(
                    best_yolo,
                    fallback_detection,
                    image.width,
                    image.height,
                )
                locked_result = dict(refined)
                locked_result["defects"] = [merged_detection]
                locked_result["pc_fast_review"] = True
                locked_result["yolo_box_locked"] = True
                locked_result = self.normalize_detection_result(locked_result)
                locked_result = self.filter_edge_defects_for_image(locked_result, payload, image)
                if locked_result.get("has_defect"):
                    refined = locked_result
        elif fallback_detection is not None and fallback_detection.get("type") == "scratch":
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

    def should_run_yolo_tile_review(self, detections):
        if self.low_latency_mode_var.get():
            return False
        if not self.yolo_low_conf_tile_var.get():
            return False
        if not detections:
            return True
        best = max(detections, key=lambda item: self.confidence_float(item))
        if self.confidence_float(best) < YOLO_LOW_CONFIDENCE_TILE_THRESHOLD:
            return True
        detection_type = best.get("type")
        aspect_ratio = float(best.get("aspect_ratio", 0) or 0)
        return (
            detection_type == "scratch"
            and self.confidence_float(best) < 0.46
            and aspect_ratio < 2.2
        )

    def merge_yolo_detections(self, base_detections, extra_detections):
        merged = [dict(defect) for defect in (base_detections or []) if isinstance(defect, dict)]
        for detection in extra_detections or []:
            if not isinstance(detection, dict):
                continue
            duplicate = False
            for old in merged:
                if old.get("type") == detection.get("type") and self.fast_review_iou(old, detection) >= 0.32:
                    duplicate = True
                    if self.confidence_float(detection) > self.confidence_float(old):
                        old.update(detection)
                    break
            if not duplicate:
                merged.append(dict(detection))
        merged.sort(key=lambda item: self.confidence_float(item), reverse=True)
        return merged[:12]

    def correct_yolo_black_stain_classes(self, image, detections, gray=None):
        if cv2 is None or np is None or not detections:
            return detections
        if gray is None:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=13, sigmaY=13)
        dark_response = np.maximum(background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
        bright_response = np.maximum(gray.astype(np.int16) - background.astype(np.int16), 0).astype(np.uint8)
        corrected = []
        for detection in detections:
            corrected.append(self.correct_single_yolo_black_stain(gray, dark_response, bright_response, detection))
        return corrected

    def correct_single_yolo_black_stain(self, gray, dark_response, bright_response, detection):
        if not isinstance(detection, dict) or detection.get("type") != "scratch":
            return detection
        source = self.detection_source(detection)
        if not self.is_yolo_source(source):
            return detection
        image_h, image_w = gray.shape[:2]
        x = self._positive_int(detection.get("x"))
        y = self._positive_int(detection.get("y"))
        w = self._positive_int(detection.get("w"))
        h = self._positive_int(detection.get("h"))
        if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
            return detection
        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))
        center_x = (x + w / 2.0) / float(image_w)
        center_y = (y + h / 2.0) / float(image_h)
        if center_x < 0.18 or center_x > 0.78 or center_y < 0.22 or center_y > 0.84:
            return detection

        box = gray[y:y + h, x:x + w]
        if box.size <= 0:
            return detection
        pad = max(10, int(min(image_w, image_h) * 0.035))
        lx1 = max(0, x - pad)
        ly1 = max(0, y - pad)
        lx2 = min(image_w, x + w + pad)
        ly2 = min(image_h, y + h + pad)
        local = gray[ly1:ly2, lx1:lx2]
        if local.size <= 0:
            return detection
        box_dark_response = dark_response[y:y + h, x:x + w]
        box_bright_response = bright_response[y:y + h, x:x + w]
        local_dark_response = dark_response[ly1:ly2, lx1:lx2]
        local_bright_response = bright_response[ly1:ly2, lx1:lx2]
        if (
            box_dark_response.size <= 0
            or box_bright_response.size <= 0
            or local_dark_response.size <= 0
            or local_bright_response.size <= 0
        ):
            return detection
        box_mean = float(np.mean(box))
        local_mean = float(np.mean(local))
        signed_delta = box_mean - local_mean
        dark_limit = min(
            float(np.percentile(local, 42.0)),
            local_mean - max(4.0, float(np.std(local)) * 0.18),
        )
        dark_fraction = float(np.mean(box <= dark_limit))
        dark_response_limit = max(4.0, float(np.percentile(local_dark_response, 76.0)))
        bright_response_limit = max(8.0, float(np.percentile(local_bright_response, 84.0)))
        dark_response_fraction = float(np.mean(box_dark_response >= dark_response_limit))
        bright_response_fraction = float(np.mean(box_bright_response >= bright_response_limit))
        dark_response_p85 = float(np.percentile(box_dark_response, 85.0))
        bright_response_p85 = float(np.percentile(box_bright_response, 85.0))
        combined_dark_fraction = max(dark_fraction, dark_response_fraction)
        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        radial_metrics = self.yolo_black_radial_stain_metrics(
            box,
            box_dark_response,
            box_bright_response,
            dark_response_limit,
            bright_response_limit,
        )
        cluster_metrics = self.yolo_black_cluster_stain_metrics(
            box,
            box_dark_response,
            dark_limit,
            dark_response_limit,
        )
        dark_is_dominant = (
            bright_response_fraction <= 0.16
            and dark_response_p85 >= bright_response_p85 * 0.75
            and signed_delta <= 1.5
        ) or (
            bright_response_fraction <= 0.20
            and dark_response_p85 >= bright_response_p85 * 1.10
            and signed_delta <= 3.0
        )
        radial_black_stain = (
            radial_metrics["line_count"] >= FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_LINES
            and radial_metrics["angle_groups"] >= FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_ANGLE_GROUPS
            and radial_metrics["total_length"] >= FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_TOTAL_LENGTH
            and radial_metrics["dark_fraction"] >= FAST_REVIEW_BLACK_STAIN_RADIAL_MIN_DARK_FRACTION
            and radial_metrics["bright_fraction"] <= FAST_REVIEW_BLACK_STAIN_RADIAL_MAX_BRIGHT_FRACTION
            and signed_delta <= FAST_REVIEW_BLACK_STAIN_RADIAL_MAX_SIGNED_DELTA
            and (
                dark_response_p85 >= max(3.5, bright_response_p85 * 0.55)
                or radial_metrics["dark_fraction"] >= radial_metrics["bright_fraction"] * 0.42
            )
            and aspect_ratio <= 4.8
        )
        cluster_black_stain = (
            cluster_metrics["component_count"] >= FAST_REVIEW_BLACK_CLUSTER_MIN_COMPONENTS
            and cluster_metrics["total_fill_ratio"] >= FAST_REVIEW_BLACK_CLUSTER_MIN_FILL_RATIO
            and cluster_metrics["merged_aspect"] <= FAST_REVIEW_BLACK_CLUSTER_MAX_MERGED_ASPECT
            and cluster_metrics["largest_component_aspect"] <= FAST_REVIEW_BLACK_CLUSTER_MAX_COMPONENT_ASPECT
            and combined_dark_fraction >= 0.20
            and bright_response_fraction <= 0.34
            and dark_response_p85 >= 4.5
        )
        compact_black_stain = (
            cluster_metrics["dense_component_count"] >= 2
            and cluster_metrics["largest_area_ratio"] >= 0.06
            and aspect_ratio <= 2.2
            and combined_dark_fraction >= 0.24
            and signed_delta <= 6.0
        )
        if (
            (
                combined_dark_fraction >= 0.16
                and dark_response_p85 >= 4.5
                and dark_is_dominant
                and aspect_ratio <= 4.2
            )
            or radial_black_stain
            or cluster_black_stain
            or compact_black_stain
        ) and w * h >= max(450, int(image_w * image_h * 0.006)):
            corrected = dict(detection)
            corrected["type"] = "stain"
            corrected["source"] = source
            corrected["review_source"] = "pc_yolo_black_stain_guard"
            corrected["class_corrected_from"] = "scratch"
            corrected["black_stain_guard"] = True
            corrected["density"] = round(combined_dark_fraction, 2)
            corrected["brightness_signed_delta"] = round(signed_delta, 1)
            corrected["local_dark_delta"] = round(dark_response_p85, 1)
            corrected["black_radial_stain"] = bool(radial_black_stain)
            corrected["black_cluster_stain"] = bool(cluster_black_stain or compact_black_stain)
            if radial_black_stain:
                corrected["line_count"] = int(radial_metrics["line_count"])
                corrected["angle_groups"] = int(radial_metrics["angle_groups"])
                corrected["total_line_length"] = round(float(radial_metrics["total_length"]), 1)
            if cluster_black_stain or compact_black_stain:
                corrected["cluster_count"] = int(cluster_metrics["component_count"])
                corrected["cluster_fill_ratio"] = round(float(cluster_metrics["total_fill_ratio"]), 2)
            corrected["level"] = "medium"
            return corrected
        return detection

    def yolo_black_radial_stain_metrics(
        self,
        box_gray,
        box_dark_response,
        box_bright_response,
        dark_response_limit,
        bright_response_limit,
    ):
        metrics = {
            "line_count": 0,
            "angle_groups": 0,
            "total_length": 0.0,
            "dark_fraction": 0.0,
            "bright_fraction": 1.0,
        }
        if cv2 is None or np is None:
            return metrics
        if box_gray is None or box_dark_response is None or box_bright_response is None:
            return metrics
        if box_gray.size <= 0 or box_dark_response.size <= 0 or box_bright_response.size <= 0:
            return metrics

        dark_mask = np.where(box_dark_response >= dark_response_limit, 255, 0).astype(np.uint8)
        bright_mask = np.where(box_bright_response >= bright_response_limit, 255, 0).astype(np.uint8)
        metrics["dark_fraction"] = float(cv2.countNonZero(dark_mask)) / float(max(1, dark_mask.size))
        metrics["bright_fraction"] = float(cv2.countNonZero(bright_mask)) / float(max(1, bright_mask.size))

        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        edges = cv2.Canny(box_gray, 25, 70)
        edges = cv2.bitwise_and(edges, dark_mask)
        min_side = min(box_gray.shape[:2])
        min_line_length = max(10, int(min_side * 0.16))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=8,
            minLineLength=min_line_length,
            maxLineGap=8,
        )
        if lines is None:
            return metrics

        angle_groups = set()
        total_length = 0.0
        line_count = 0
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
            angle_groups.add(int(angle // 30.0))
            total_length += length
            line_count += 1

        metrics["line_count"] = int(line_count)
        metrics["angle_groups"] = int(len(angle_groups))
        metrics["total_length"] = float(total_length)
        return metrics

    def yolo_black_cluster_stain_metrics(self, box_gray, box_dark_response, dark_limit, dark_response_limit):
        metrics = {
            "component_count": 0,
            "dense_component_count": 0,
            "largest_area_ratio": 0.0,
            "total_fill_ratio": 0.0,
            "merged_aspect": 0.0,
            "largest_component_aspect": 0.0,
        }
        if cv2 is None or np is None:
            return metrics
        if box_gray is None or box_dark_response is None:
            return metrics
        if box_gray.size <= 0 or box_dark_response.size <= 0:
            return metrics

        dark_from_gray = np.where(box_gray <= dark_limit, 255, 0).astype(np.uint8)
        dark_from_response = np.where(box_dark_response >= dark_response_limit, 255, 0).astype(np.uint8)
        dark_mask = cv2.bitwise_or(dark_from_gray, dark_from_response)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
        if component_count <= 1:
            return metrics

        box_h, box_w = box_gray.shape[:2]
        box_area = float(max(1, box_w * box_h))
        min_component_area = max(18.0, box_area * 0.008)
        components = []
        for index in range(1, component_count):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area < min_component_area or w <= 0 or h <= 0:
                continue
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            components.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": float(area),
                "aspect_ratio": aspect_ratio,
            })
        if not components:
            return metrics

        metrics["component_count"] = int(len(components))
        total_area = sum(item["area"] for item in components)
        largest_area = max(item["area"] for item in components)
        metrics["dense_component_count"] = int(sum(1 for item in components if item["area"] >= min_component_area * 1.8))
        metrics["largest_area_ratio"] = float(largest_area) / box_area
        metrics["total_fill_ratio"] = float(total_area) / box_area
        metrics["largest_component_aspect"] = float(max(item["aspect_ratio"] for item in components))

        left = min(item["x"] for item in components)
        top = min(item["y"] for item in components)
        right = max(item["x"] + item["w"] for item in components)
        bottom = max(item["y"] + item["h"] for item in components)
        merged_w = max(1, right - left)
        merged_h = max(1, bottom - top)
        metrics["merged_aspect"] = float(max(merged_w, merged_h)) / float(max(1, min(merged_w, merged_h)))
        return metrics

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

    def yolo_center_guard_roi(self, image_w, image_h):
        roi = self.center_fallback_roi(image_w, image_h)
        roi["source"] = "pc_yolo_center_guard"
        return roi

    def yolo_filter_roi_for_image(self, image, payload, result, inference_roi=None):
        roi = self.scaled_effective_detection_roi_for_image(image, payload, result)
        if roi is not None and not self.yolo_roi_is_edge_biased(roi, image.width, image.height):
            return roi
        if isinstance(inference_roi, dict):
            roi = self.valid_roi_or_none(inference_roi, image.width, image.height)
            if roi is not None and not self.yolo_roi_is_edge_biased(roi, image.width, image.height):
                roi["source"] = "pc_yolo_filter_roi"
                return roi
        fallback = self.best_live_detection_roi(result, image.width, image.height)
        fallback_source = str(fallback.get("source", "")).lower() if isinstance(fallback, dict) else ""
        if fallback_source == "pc_center_fallback" or self.yolo_roi_is_edge_biased(fallback, image.width, image.height):
            return self.yolo_center_guard_roi(image.width, image.height)
        return fallback

    def yolo_roi_is_edge_biased(self, roi, image_w, image_h):
        if not isinstance(roi, dict):
            return False
        source = str(roi.get("source", "")).lower()
        if source not in ("pc_auto_roi", "pc_yolo_roi", "pc_yolo_filter_roi"):
            return False
        x = self._positive_int(roi.get("x"))
        y = self._positive_int(roi.get("y"))
        w = self._positive_int(roi.get("w"))
        h = self._positive_int(roi.get("h"))
        if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
            return True
        side_margin = max(2, int(image_w * YOLO_AUTO_ROI_SIDE_CLIP_MARGIN_RATIO))
        touches_side = x <= side_margin or x + w >= image_w - side_margin
        partial_width = w < int(image_w * YOLO_AUTO_ROI_EDGE_PARTIAL_SIZE_RATIO)
        center_x = (x + w / 2.0) / float(image_w)
        if touches_side and partial_width:
            return True
        return abs(center_x - 0.50) > YOLO_AUTO_ROI_MAX_CENTER_X_DRIFT_RATIO

    def scaled_effective_detection_roi_for_image(self, image, payload, result):
        roi, source_w, source_h = self.resolve_detection_roi_for_image(image, payload, result)
        if not isinstance(roi, dict):
            return None
        image_w, image_h = image.size
        x_scale = float(image_w) / float(max(1, source_w))
        y_scale = float(image_h) / float(max(1, source_h))
        scaled = {
            "x": int(self._positive_int(roi.get("x")) * x_scale),
            "y": int(self._positive_int(roi.get("y")) * y_scale),
            "w": int(self._positive_int(roi.get("w")) * x_scale),
            "h": int(self._positive_int(roi.get("h")) * y_scale),
            "source": str(roi.get("source", "pc_effective_roi") or "pc_effective_roi"),
        }
        return self.valid_roi_or_none(scaled, image_w, image_h)

    def yolo_inference_roi_for_image(self, image, payload, result):
        if not YOLO_ROI_INFERENCE_ENABLED:
            return None
        roi, source_w, source_h = self.resolve_detection_roi_for_image(image, payload, result)
        if not isinstance(roi, dict):
            return self.yolo_center_guard_roi(image.width, image.height)
        image_w, image_h = image.size
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
        source = str(roi.get("source", "")).lower()
        if source == "pc_center_fallback":
            return self.yolo_center_guard_roi(image_w, image_h)
        if self.yolo_roi_is_edge_biased({"x": x, "y": y, "w": w, "h": h, "source": source}, image_w, image_h):
            return self.yolo_center_guard_roi(image_w, image_h)

        frame_area = float(max(1, image_w * image_h))
        roi_area_ratio = float(w * h) / frame_area
        min_side = min(image_w, image_h)
        if (
            roi_area_ratio >= YOLO_ROI_MAX_AREA_RATIO
            or w < int(min_side * YOLO_ROI_MIN_SIZE_RATIO)
            or h < int(min_side * YOLO_ROI_MIN_SIZE_RATIO)
        ):
            return self.yolo_center_guard_roi(image_w, image_h)

        pad = max(8, int(min(w, h) * YOLO_ROI_PADDING_RATIO))
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(image_w, x + w + pad)
        bottom = min(image_h, y + h + pad)
        return {
            "x": int(left),
            "y": int(top),
            "w": int(max(1, right - left)),
            "h": int(max(1, bottom - top)),
            "source": "pc_yolo_roi",
        }

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

    def enhance_gray_for_inspection(self, gray):
        if cv2 is None or np is None or gray is None:
            return gray
        try:
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=INSPECTION_CLAHE_CLIP_LIMIT,
                tileGridSize=INSPECTION_CLAHE_TILE_GRID,
            )
            return clahe.apply(gray)
        except Exception:
            return gray

    def fast_cv_detect_defect(self, image, mask=None, gray=None):
        if gray is None:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        if width < 32 or height < 32:
            return None

        if mask is None:
            mask = self.build_detection_mask(image, {}, self.latest_detection_result)
        if mask is None:
            return None
        if cv2.countNonZero(mask) <= 0:
            return None

        analysis_gray = self.enhance_gray_for_inspection(gray)
        mean_value = float(np.mean(analysis_gray[mask > 0]))
        std_value = float(np.std(analysis_gray[mask > 0]))
        contrast_delta = max(18.0, std_value * 1.2)
        edge_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        stain = self.fast_cv_find_stain(analysis_gray, mask, mean_value, contrast_delta, edge_distance)
        morph_stain = self.fast_cv_find_morphology_stain(analysis_gray, mask, edge_distance)
        if morph_stain is not None and (stain is None or self.confidence_float(morph_stain) >= self.confidence_float(stain) - 0.03):
            stain = morph_stain
        dark_cluster = self.fast_cv_find_dark_stain_cluster(analysis_gray, mask, mean_value, std_value, edge_distance)
        if dark_cluster is not None and (stain is None or dark_cluster.get("area", 0) >= stain.get("area", 0)):
            stain = dark_cluster
        dark_x_stain = self.fast_cv_find_dark_x_stain(analysis_gray, mask, edge_distance)
        if dark_x_stain is None:
            center_mask = self.build_center_defect_fallback_mask(width, height)
            if cv2.countNonZero(center_mask) > 0:
                center_edge_distance = cv2.distanceTransform(center_mask, cv2.DIST_L2, 3)
                dark_x_stain = self.fast_cv_find_dark_x_stain(analysis_gray, center_mask, center_edge_distance)
        if dark_x_stain is not None:
            if stain is None or dark_x_stain.get("total_line_length", 0) >= stain.get("length", 0) * 1.35:
                stain = dark_x_stain
        scratch = self.fast_cv_find_scratch(analysis_gray, mask, edge_distance, None)
        morph_scratch = self.fast_cv_find_morphology_scratch(analysis_gray, mask, edge_distance)
        if morph_scratch is not None and (scratch is None or self.confidence_float(morph_scratch) >= self.confidence_float(scratch) - 0.04):
            scratch = morph_scratch
        detection = self.choose_fast_cv_defect(scratch, stain, mask)
        if dark_x_stain is not None and (
            detection is None
            or detection.get("type") != "stain"
            or dark_x_stain.get("confidence", 0) >= detection.get("confidence", 0) - 0.04
        ):
            detection = dark_x_stain
        if detection is None:
            detection = self.fast_cv_find_center_radial_stain(analysis_gray)
        if detection is None:
            detection = self.fast_cv_find_dark_star_stain_lines(analysis_gray)
        return detection

    def inspection_kernel_size(self, width, height, ratio, minimum=7, maximum=31):
        size = int(min(width, height) * ratio)
        size = max(minimum, min(maximum, size))
        if size % 2 == 0:
            size += 1
        return max(3, size)

    def fast_cv_find_morphology_stain(self, gray, mask, edge_distance=None):
        height, width = gray.shape[:2]
        if width < 64 or height < 64 or mask is None:
            return None
        mask_area = max(1, int(cv2.countNonZero(mask)))
        if mask_area <= 0:
            return None

        kernel_size = self.inspection_kernel_size(width, height, 0.085, minimum=23, maximum=61)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        values = blackhat[mask > 0]
        if values.size <= 0:
            return None
        threshold = max(6.0, min(MORPH_STAIN_MIN_RESPONSE, float(np.percentile(values, 99.2))))
        stain_mask = np.where(blackhat >= threshold, 255, 0).astype(np.uint8)
        stain_mask = cv2.bitwise_and(stain_mask, mask)
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(stain_mask, 8)
        if component_count <= 1:
            return None

        min_area = max(18, int(mask_area * 0.0004))
        max_area = max(min_area + 1, int(mask_area * 0.26))
        min_edge_distance = max(FAST_REVIEW_STAIN_MIN_EDGE_DISTANCE, int(min(width, height) * 0.018))
        best = None
        best_score = 0.0
        for index in range(1, component_count):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area < min_area or area > max_area or w <= 0 or h <= 0:
                continue
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            if aspect_ratio > 4.4:
                continue
            box_area = max(1, w * h)
            fill_ratio = float(area) / float(box_area)
            if fill_ratio < 0.08:
                continue
            center_x = max(0, min(width - 1, int(x + w / 2)))
            center_y = max(0, min(height - 1, int(y + h / 2)))
            if mask[center_y, center_x] == 0:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance:
                continue
            response_roi = blackhat[y:y + h, x:x + w]
            active_response = response_roi[stain_mask[y:y + h, x:x + w] > 0]
            if active_response.size <= 0:
                continue
            local_dark_delta = float(np.percentile(active_response, 85.0))
            score = area + local_dark_delta * 12.0 + fill_ratio * 160.0
            if score > best_score:
                best_score = score
                confidence = min(0.93, 0.76 + min(0.10, local_dark_delta / 110.0) + min(0.07, area / float(mask_area) * 4.0))
                best = {
                    "type": "stain",
                    "confidence": round(confidence, 2),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "length": int(max(w, h)),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if area >= mask_area * 0.018 or local_dark_delta >= 24.0 else "light",
                    "source": "pc_morph_blackhat_stain",
                    "density": round(fill_ratio, 2),
                    "local_dark_delta": round(local_dark_delta, 1),
                    "brightness_signed_delta": round(-local_dark_delta, 1),
                }
        return best

    def fast_cv_find_morphology_scratch(self, gray, mask, edge_distance=None):
        height, width = gray.shape[:2]
        if width < 96 or height < 80 or mask is None:
            return None
        mask_area = max(1, int(cv2.countNonZero(mask)))
        if mask_area <= 0:
            return None

        kernel_size = self.inspection_kernel_size(width, height, 0.030, minimum=9, maximum=25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        line_response = np.maximum(blackhat, tophat)
        values = line_response[mask > 0]
        if values.size <= 0:
            return None
        threshold = max(MORPH_SCRATCH_MIN_RESPONSE, float(np.percentile(values, 94.0)))
        candidate_mask = np.where(line_response >= threshold, 255, 0).astype(np.uint8)
        candidate_mask = cv2.bitwise_and(candidate_mask, mask)

        line_length = self.inspection_kernel_size(width, height, 0.024, minimum=7, maximum=17)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
        diagonal_kernel = np.eye(line_length, dtype=np.uint8)
        anti_diagonal_kernel = np.fliplr(diagonal_kernel)
        line_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, vertical_kernel)
        line_mask = cv2.bitwise_or(line_mask, cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, horizontal_kernel))
        line_mask = cv2.bitwise_or(line_mask, cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, diagonal_kernel))
        line_mask = cv2.bitwise_or(line_mask, cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, anti_diagonal_kernel))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
        if cv2.countNonZero(line_mask) <= 0 and cv2.countNonZero(candidate_mask) > 0:
            line_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(line_mask, 8)
        if component_count <= 1:
            return None

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        best = None
        best_score = 0.0
        for index in range(1, component_count):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area < 18 or w <= 0 or h <= 0:
                continue
            length = max(w, h)
            short_side = max(1, min(w, h))
            aspect_ratio = float(length) / float(short_side)
            if length < max(18, int(min(width, height) * 0.035)) or aspect_ratio < 2.2:
                continue
            box_area = max(1, w * h)
            box_ratio = float(box_area) / float(mask_area)
            if box_ratio > max(FAST_REVIEW_SCRATCH_MAX_BOX_AREA_RATIO, 0.22):
                continue
            fill_ratio = float(area) / float(box_area)
            if fill_ratio > 0.42:
                continue
            center_x = max(0, min(width - 1, int(x + w / 2)))
            center_y = max(0, min(height - 1, int(y + h / 2)))
            if mask[center_y, center_x] == 0:
                continue
            if edge_distance is not None and not self.scratch_box_has_safe_edge_distance(
                edge_distance,
                x,
                y,
                min(width - 1, x + w),
                min(height - 1, y + h),
                min_edge_distance,
                0.76,
            ):
                continue
            response_roi = line_response[y:y + h, x:x + w]
            active_response = response_roi[line_mask[y:y + h, x:x + w] > 0]
            if active_response.size <= 0:
                continue
            local_delta = float(np.percentile(active_response, 85.0))
            if local_delta < MORPH_SCRATCH_MIN_RESPONSE:
                continue
            score = length * 1.4 + area * 0.45 + local_delta * 7.0 + aspect_ratio * 6.0
            if score > best_score:
                best_score = score
                confidence = min(0.92, 0.75 + min(0.09, length / 360.0) + min(0.06, local_delta / 120.0))
                best = {
                    "type": "scratch",
                    "confidence": round(confidence, 2),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if length >= 70 or area >= 130 else "light",
                    "source": "pc_morph_line_scratch",
                    "fill_ratio": round(fill_ratio, 2),
                    "local_bright_delta": round(float(np.percentile(tophat[y:y + h, x:x + w][line_mask[y:y + h, x:x + w] > 0], 80.0)), 1),
                    "local_dark_delta": round(float(np.percentile(blackhat[y:y + h, x:x + w][line_mask[y:y + h, x:x + w] > 0], 80.0)), 1),
                    "strong_internal_scratch": bool(length >= 45 and aspect_ratio >= 2.6 and fill_ratio <= 0.32),
                }
        return best

    def choose_fast_cv_defect(self, scratch, stain, mask):
        if scratch is None:
            if self.stain_candidate_looks_like_scratch(stain, mask):
                return self.promote_stain_candidate_to_scratch(stain)
            return stain
        if stain is None:
            if self.scratch_candidate_looks_like_dark_stain(scratch, mask):
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
        if self.scratch_candidate_looks_like_dark_stain(scratch, mask):
            return self.promote_scratch_candidate_to_stain(scratch)
        scratch_is_long_slender = (
            scratch_aspect >= 3.5
            and scratch_length >= 80
            and scratch_area_ratio <= 0.018
        )
        if scratch_is_long_slender:
            return scratch
        scratch_is_crosshatch = (
            scratch_source in ("pc_fast_review_line", "pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch", "pc_fast_curvilinear_scratch")
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
            scratch_source in ("pc_fast_review_line", "pc_fast_bright_scratch", "pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch", "pc_fast_curvilinear_scratch")
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
            scratch_source in ("pc_fast_review_line", "pc_fast_bright_scratch", "pc_fast_bright_crosshatch", "pc_fast_center_cross_scratch", "pc_fast_curvilinear_scratch")
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
        if self.dark_stain_candidate_looks_like_weak_scratch_shadow(stain, mask):
            return True
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

    def dark_stain_candidate_looks_like_weak_scratch_shadow(self, stain, mask):
        if not isinstance(stain, dict) or mask is None:
            return False
        angle_groups = int(stain.get("angle_groups", 0) or 0)
        local_delta = float(stain.get("local_dark_delta", 0) or 0)
        signed_delta = float(stain.get("brightness_signed_delta", 0) or 0)
        aspect_ratio = float(stain.get("aspect_ratio", 0) or 0)
        density = float(stain.get("density", 0) or 0)
        area = int(stain.get("area", 0) or 0)
        length = int(stain.get("length", 0) or 0)
        if angle_groups > FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_ANGLE_GROUPS:
            return False
        if local_delta > FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_LOCAL_DELTA:
            return False
        if (
            signed_delta < FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MIN_SIGNED_DELTA
            or signed_delta > FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_SIGNED_DELTA
        ):
            return False
        if aspect_ratio <= 0 or aspect_ratio > FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_ASPECT:
            return False
        if density < 0.40 or density > 0.58:
            return False
        if area <= 0 or area > FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_AREA:
            return False
        if length <= 0 or length > FAST_REVIEW_DARK_STAIN_WEAK_SCRATCH_MAX_LENGTH:
            return False

        image_h, image_w = mask.shape[:2]
        x = self._positive_int(stain.get("x"))
        y = self._positive_int(stain.get("y"))
        w = self._positive_int(stain.get("w"))
        h = self._positive_int(stain.get("h"))
        if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
            return False
        center_x = (x + w / 2.0) / float(image_w)
        center_y = (y + h / 2.0) / float(image_h)
        return 0.25 <= center_x <= 0.68 and 0.28 <= center_y <= 0.72

    def promote_stain_candidate_to_scratch(self, stain):
        promoted = dict(stain)
        promoted["type"] = "scratch"
        promoted["source"] = "pc_fast_dark_scratch"
        promoted["confidence"] = max(0.84, min(0.93, float(stain.get("confidence", 0.84) or 0.84)))
        promoted["strong_internal_scratch"] = True
        promoted["level"] = "medium" if self._positive_int(stain.get("length")) >= 65 else "light"
        return promoted

    def scratch_candidate_looks_like_dark_stain(self, scratch, mask=None):
        if not isinstance(scratch, dict):
            return False
        source = str(scratch.get("source", ""))
        if source not in ("pc_fast_review", "pc_fast_review_line", "pc_fast_center_cross_scratch"):
            return False
        signed_delta = float(scratch.get("local_signed_delta", 0) or 0)
        dark_delta = float(scratch.get("local_dark_delta", 0) or 0)
        bright_delta = float(scratch.get("local_bright_delta", 0) or 0)
        line_count = int(scratch.get("line_count", 0) or 0)
        angle_groups = int(scratch.get("angle_groups", 0) or 0)
        fill_ratio = float(scratch.get("fill_ratio", 0) or 0)
        aspect_ratio = float(scratch.get("aspect_ratio", 0) or 0)
        dark_fraction = float(scratch.get("dark_fraction", 0) or 0)
        component_count = int(scratch.get("component_count", 0) or 0)
        compact_dark_cluster = (
            fill_ratio >= 0.24
            and aspect_ratio <= 2.25
            and component_count >= 3
            and (
                dark_delta >= max(4.5, bright_delta * 0.70)
                or signed_delta <= FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA
            )
        )
        if compact_dark_cluster:
            return True
        black_dominant_middle_defect = (
            line_count >= FAST_REVIEW_STAR_SCRATCH_MIN_LINES
            and angle_groups >= FAST_REVIEW_BLACK_STAIN_MIN_ANGLE_GROUPS
            and aspect_ratio <= FAST_REVIEW_BLACK_STAIN_MAX_ASPECT
            and fill_ratio >= FAST_REVIEW_BLACK_STAIN_MIN_FILL_RATIO
            and dark_fraction >= FAST_REVIEW_BLACK_STAIN_MIN_DARK_FRACTION
            and bright_delta <= FAST_REVIEW_BLACK_STAIN_MAX_BRIGHT_DELTA
            and (
                dark_delta >= max(3.0, bright_delta * 0.75)
                or signed_delta <= FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA
            )
        )
        if black_dominant_middle_defect:
            return True
        if (
            line_count >= FAST_REVIEW_STAR_SCRATCH_MIN_LINES
            and signed_delta <= FAST_REVIEW_DARK_STAR_STAIN_MAX_SIGNED_DELTA
            and dark_delta > bright_delta * 1.35
            and fill_ratio >= FAST_REVIEW_STAR_SCRATCH_MIN_FILL
        ):
            return True
        if source not in ("pc_fast_center_cross_scratch", "pc_fast_review_line") or mask is None:
            return False
        x = self._positive_int(scratch.get("x"))
        y = self._positive_int(scratch.get("y"))
        w = self._positive_int(scratch.get("w"))
        h = self._positive_int(scratch.get("h"))
        if w <= 0 or h <= 0:
            return False
        image_h, image_w = mask.shape[:2]
        compact_dark_star_stain = (
            dark_fraction >= 0.38
            and fill_ratio >= 0.36
            and bright_delta <= 16.0
            and line_count <= 25
            and aspect_ratio <= 2.70
            and angle_groups >= 4
        )
        if compact_dark_star_stain:
            return True
        box_center_x = x + w / 2.0
        if box_center_x < image_w * 0.56 or w < image_w * 0.18:
            return False
        if aspect_ratio > 2.25 or bright_delta > FAST_REVIEW_DARK_STAR_STAIN_MAX_BRIGHT_DELTA:
            return False
        if angle_groups < 4 or fill_ratio > 0.32:
            return False
        x1 = max(0, min(image_w - 1, int(x)))
        y1 = max(0, min(image_h - 1, int(y)))
        x2 = max(x1 + 1, min(image_w, int(x + w)))
        y2 = max(y1 + 1, min(image_h, int(y + h)))
        mask_roi = mask[y1:y2, x1:x2]
        if mask_roi.size <= 0:
            return False
        mask_overlap = float(cv2.countNonZero(mask_roi)) / float(max(1, (x2 - x1) * (y2 - y1)))
        if mask_overlap < 0.72:
            return False
        return bool(scratch.get("dark_star_stain_like"))

    def promote_scratch_candidate_to_stain(self, scratch):
        promoted = dict(scratch)
        promoted["type"] = "stain"
        promoted["source"] = "pc_fast_dark_star_stain"
        promoted["confidence"] = max(0.84, min(0.94, float(scratch.get("confidence", 0.84) or 0.84)))
        promoted["density"] = float(scratch.get("fill_ratio", 0) or 0)
        promoted["level"] = "medium" if self._positive_int(scratch.get("length")) >= 55 else "light"
        promoted.pop("strong_internal_scratch", None)
        return promoted

    def fast_cv_find_dark_x_stain(self, gray, mask, edge_distance=None):
        height, width = gray.shape[:2]
        if width < 120 or height < 90 or mask is None:
            return None

        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=11, sigmaY=11)
        dark_response = np.maximum(background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
        bright_response = np.maximum(gray.astype(np.int16) - background.astype(np.int16), 0).astype(np.uint8)
        lens_dark_values = dark_response[mask > 0]
        lens_gray_values = gray[mask > 0]
        if lens_dark_values.size <= 0 or lens_gray_values.size <= 0:
            return None

        dark_threshold = max(5.0, float(np.percentile(lens_dark_values, 82.0)))
        gray_threshold = min(
            float(np.percentile(lens_gray_values, 32.0)),
            float(np.mean(lens_gray_values) - max(5.0, np.std(lens_gray_values) * 0.24)),
        )
        dark_mask = np.where(
            (dark_response >= dark_threshold) | (gray <= gray_threshold),
            255,
            0,
        ).astype(np.uint8)
        dark_mask = cv2.bitwise_and(dark_mask, mask)

        glare_limit = max(178.0, float(np.percentile(lens_gray_values, 99.2)))
        glare_mask = np.where(gray >= glare_limit, 255, 0).astype(np.uint8)
        glare_mask = cv2.dilate(glare_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(glare_mask))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

        edges = cv2.Canny(gray, 24, 72)
        edges = cv2.bitwise_and(edges, dark_mask)
        min_line_length = max(14, int(min(width, height) * 0.045))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=9,
            minLineLength=min_line_length,
            maxLineGap=10,
        )
        if lines is None:
            return None

        min_edge_distance = max(
            FAST_REVIEW_STAIN_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_STAIN_EDGE_MARGIN_RATIO),
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
            if center_x < width * 0.18 or center_x > width * 0.82:
                continue
            if center_y < height * 0.18 or center_y > height * 0.84:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance * 0.70:
                continue
            segments.append((x1, y1, x2, y2, length))

        if len(segments) < FAST_REVIEW_DARK_X_STAIN_MIN_LINES:
            return None

        best = None
        best_score = 0.0
        frame_area = float(max(1, width * height))
        for cluster in self.cluster_scratch_line_segments(segments):
            if len(cluster) < FAST_REVIEW_DARK_X_STAIN_MIN_LINES:
                continue
            xs = []
            ys = []
            angle_groups = set()
            total_length = 0.0
            for x1, y1, x2, y2, length in cluster:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
                angle_groups.add(int(angle // 30.0))
                total_length += length
            if len(angle_groups) < FAST_REVIEW_DARK_X_STAIN_MIN_ANGLE_GROUPS:
                continue
            if total_length < FAST_REVIEW_DARK_X_STAIN_MIN_TOTAL_LENGTH:
                continue

            left = max(0, min(xs) - 4)
            top = max(0, min(ys) - 4)
            right = min(width, max(xs) + 5)
            bottom = min(height, max(ys) + 5)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            box_ratio = float(box_w * box_h) / frame_area
            if box_ratio < FAST_REVIEW_DARK_X_STAIN_MIN_BOX_RATIO or box_ratio > FAST_REVIEW_DARK_X_STAIN_MAX_BOX_RATIO:
                continue
            aspect_ratio = float(max(box_w, box_h)) / float(max(1, min(box_w, box_h)))
            if aspect_ratio > FAST_REVIEW_DARK_X_STAIN_MAX_ASPECT:
                continue
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            if center_x < width * 0.32 or center_x > width * 0.73:
                continue
            if center_y < height * 0.28 or center_y > height * 0.80:
                continue

            roi_mask = mask[top:bottom, left:right]
            if roi_mask.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(roi_mask)) / float(max(1, box_w * box_h))
            if mask_overlap < 0.54:
                continue
            dark_roi = dark_response[top:bottom, left:right]
            bright_roi = bright_response[top:bottom, left:right]
            gray_roi = gray[top:bottom, left:right]
            local_pad = max(8, int(min(width, height) * 0.025))
            local_left = max(0, left - local_pad)
            local_top = max(0, top - local_pad)
            local_right = min(width, right + local_pad)
            local_bottom = min(height, bottom + local_pad)
            local_gray = gray[local_top:local_bottom, local_left:local_right]
            if dark_roi.size <= 0 or bright_roi.size <= 0 or gray_roi.size <= 0 or local_gray.size <= 0:
                continue
            dark_fraction = float(np.mean(dark_roi >= dark_threshold))
            bright_threshold = max(8.0, float(np.percentile(bright_response[mask > 0], 86.0)))
            bright_fraction = float(np.mean(bright_roi >= bright_threshold))
            signed_delta = float(np.mean(gray_roi)) - float(np.mean(local_gray))
            dark_p85 = float(np.percentile(dark_roi, 85.0))
            bright_p85 = float(np.percentile(bright_roi, 85.0))
            if dark_fraction < FAST_REVIEW_DARK_X_STAIN_MIN_DARK_FRACTION:
                continue
            if bright_fraction > FAST_REVIEW_DARK_X_STAIN_MAX_BRIGHT_FRACTION:
                continue
            if signed_delta > FAST_REVIEW_DARK_X_STAIN_MAX_SIGNED_DELTA:
                continue
            if dark_p85 < max(4.0, bright_p85 * 0.65) and dark_fraction < bright_fraction * 0.55:
                continue

            score = total_length + len(angle_groups) * 130.0 + dark_fraction * 420.0 + dark_p85 * 24.0
            if score <= best_score:
                continue
            best_score = score
            confidence = min(0.96, 0.88 + min(0.05, total_length / 1500.0) + min(0.03, dark_fraction * 0.08))
            best = {
                "type": "stain",
                "confidence": round(confidence, 2),
                "x": int(left),
                "y": int(top),
                "w": int(box_w),
                "h": int(box_h),
                "area": int(box_w * box_h * dark_fraction),
                "length": int(max(box_w, box_h)),
                "aspect_ratio": round(aspect_ratio, 2),
                "density": round(dark_fraction, 2),
                "angle_groups": int(len(angle_groups)),
                "line_count": int(len(cluster)),
                "total_line_length": round(float(total_length), 1),
                "brightness_signed_delta": round(signed_delta, 1),
                "local_dark_delta": round(dark_p85, 1),
                "level": "medium",
                "source": "pc_fast_dark_x_stain",
                "black_radial_stain": True,
            }
        return best

    def build_center_defect_fallback_mask(self, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        if width <= 0 or height <= 0:
            return mask
        left = int(width * 0.30)
        right = int(width * 0.76)
        top = int(height * 0.24)
        bottom = int(height * 0.84)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
        return mask

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

    def fast_cv_find_center_radial_stain(self, gray):
        height, width = gray.shape[:2]
        if width < 120 or height < 90:
            return None

        x1 = int(width * 0.04)
        x2 = int(width * 0.76)
        y1 = int(height * 0.04)
        y2 = int(height * 0.82)
        if x2 <= x1 or y2 <= y1:
            return None

        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
        local_dark = np.maximum(local_background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
        roi_dark = local_dark[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]
        if roi_dark.size <= 0 or roi_gray.size <= 0:
            return None

        dark_threshold = max(7.0, float(np.percentile(roi_dark, 88.0)))
        absolute_threshold = min(
            float(np.percentile(roi_gray, 24.0)),
            float(np.mean(roi_gray) - max(7.0, np.std(roi_gray) * 0.38)),
        )
        radial_mask = np.where(
            (roi_dark >= dark_threshold) | (roi_gray <= absolute_threshold),
            255,
            0,
        ).astype(np.uint8)

        glare_limit = max(176.0, float(np.percentile(gray, 99.2)))
        glare_mask = np.where(roi_gray >= glare_limit, 255, 0).astype(np.uint8)
        glare_mask = cv2.dilate(glare_mask, np.ones((11, 11), dtype=np.uint8), iterations=1)
        radial_mask = cv2.bitwise_and(radial_mask, cv2.bitwise_not(glare_mask))
        radial_mask = cv2.morphologyEx(radial_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        radial_mask = cv2.dilate(radial_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _hierarchy = cv2.findContours(radial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(max(1, width * height))
        roi_mean = float(np.mean(roi_gray))
        best = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(120.0, frame_area * 0.0014) or area > frame_area * 0.075:
                continue
            lx, ly, box_w, box_h = cv2.boundingRect(contour)
            if box_w < 18 or box_h < 18:
                continue

            x = lx + x1
            y = ly + y1
            center_x = x + box_w / 2.0
            center_y = y + box_h / 2.0
            if center_x < width * 0.07 or center_x > width * 0.74:
                continue
            if center_y < height * 0.08 or center_y > height * 0.78:
                continue
            if (
                x <= width * 0.015
                or y <= height * 0.015
                or x + box_w >= width * 0.90
                or y + box_h >= height * 0.90
            ):
                continue

            aspect_ratio = float(max(box_w, box_h)) / float(max(1, min(box_w, box_h)))
            if aspect_ratio > 3.2:
                continue
            component_mask = np.zeros((box_h, box_w), dtype=np.uint8)
            shifted = contour - np.array([[[lx, ly]]], dtype=contour.dtype)
            cv2.drawContours(component_mask, [shifted], -1, 255, -1)
            fill_ratio = area / float(max(1, box_w * box_h))
            if fill_ratio < 0.06 or fill_ratio > 0.76:
                continue

            component_dark = local_dark[y:y + box_h, x:x + box_w][component_mask > 0]
            component_gray = gray[y:y + box_h, x:x + box_w][component_mask > 0]
            if component_dark.size <= 0 or component_gray.size <= 0:
                continue
            local_mean = float(np.mean(component_dark))
            signed_delta = float(np.mean(component_gray)) - roi_mean
            if signed_delta > FAST_REVIEW_CENTER_RADIAL_STAIN_MAX_SIGNED_DELTA:
                continue
            angle_groups = self.count_dark_cluster_angle_groups(gray, x, y, box_w, box_h, component_mask)
            if angle_groups < FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_ANGLE_GROUPS:
                continue
            if (
                local_mean < FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_LOCAL_DELTA
                and signed_delta > -FAST_REVIEW_CENTER_RADIAL_STAIN_MIN_SIGNED_DELTA
            ):
                continue
            if local_mean < 12.0 and signed_delta > -18.0:
                continue

            pad = max(4, int(min(width, height) * 0.018))
            left = max(0, x - pad)
            top = max(0, y - pad)
            right = min(width, x + box_w + pad)
            bottom = min(height, y + box_h + pad)
            score = (
                area
                + angle_groups * 120.0
                + local_mean * 90.0
                + abs(min(0.0, signed_delta)) * 45.0
                + fill_ratio * 220.0
            )
            if score > best_score:
                best_score = score
                out_w = max(1, right - left)
                out_h = max(1, bottom - top)
                best = {
                    "type": "stain",
                    "confidence": 0.91,
                    "x": int(left),
                    "y": int(top),
                    "w": int(out_w),
                    "h": int(out_h),
                    "area": int(area),
                    "length": int(max(out_w, out_h)),
                    "aspect_ratio": round(float(max(out_w, out_h)) / float(max(1, min(out_w, out_h))), 2),
                    "density": round(fill_ratio, 2),
                    "angle_groups": int(angle_groups),
                    "brightness_signed_delta": round(signed_delta, 1),
                    "local_dark_delta": round(local_mean, 1),
                    "level": "medium",
                    "source": "pc_fast_center_radial_stain",
                }

        return best

    def fast_cv_find_dark_star_stain_lines(self, gray):
        height, width = gray.shape[:2]
        if width < 120 or height < 90:
            return None

        x1 = int(width * 0.04)
        x2 = int(width * FAST_REVIEW_DARK_STAR_LINE_MAX_X_RATIO)
        y1 = int(height * 0.03)
        y2 = int(height * FAST_REVIEW_DARK_STAR_LINE_MAX_Y_RATIO)
        if x2 <= x1 or y2 <= y1:
            return None

        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size <= 0:
            return None

        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
        local_dark = np.maximum(local_background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)
        roi_dark = local_dark[y1:y2, x1:x2]
        dark_threshold = max(8.0, float(np.percentile(roi_dark, 92.0)))
        absolute_threshold = float(np.percentile(roi_gray, 18.0))
        candidate_mask = np.where(
            (roi_dark >= dark_threshold) | (roi_gray <= absolute_threshold),
            255,
            0,
        ).astype(np.uint8)

        glare_limit = max(176.0, float(np.percentile(gray, 99.2)))
        glare_mask = np.where(roi_gray >= glare_limit, 255, 0).astype(np.uint8)
        glare_mask = cv2.dilate(glare_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(glare_mask))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

        edges = cv2.Canny(roi_gray, 25, 75)
        edges = cv2.bitwise_and(edges, candidate_mask)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=12,
            minLineLength=18,
            maxLineGap=8,
        )
        if lines is None:
            return None

        segments = []
        angle_groups = set()
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(value) for value in line]
            length = float(((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5)
            if length < 18.0:
                continue
            center_x = (lx1 + lx2) / 2.0 + x1
            center_y = (ly1 + ly2) / 2.0 + y1
            if center_x < width * 0.04 or center_x > width * FAST_REVIEW_DARK_STAR_LINE_MAX_X_RATIO:
                continue
            if center_y < height * 0.03 or center_y > height * FAST_REVIEW_DARK_STAR_LINE_MAX_Y_RATIO:
                continue
            angle = (np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)) + 180.0) % 180.0
            angle_groups.add(int(angle // 22.5))
            segments.append((lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1, length))

        best = None
        best_score = 0.0
        for cluster in self.cluster_scratch_line_segments(segments):
            if len(cluster) < FAST_REVIEW_DARK_STAR_LINE_MIN_LINES:
                continue
            xs = []
            ys = []
            total_length = 0.0
            cluster_angle_groups = set()
            for sx1, sy1, sx2, sy2, line_length in cluster:
                xs.extend([sx1, sx2])
                ys.extend([sy1, sy2])
                total_length += line_length
                angle = (np.degrees(np.arctan2(sy2 - sy1, sx2 - sx1)) + 180.0) % 180.0
                cluster_angle_groups.add(int(angle // 22.5))
            if len(cluster_angle_groups) < FAST_REVIEW_DARK_STAR_LINE_MIN_ANGLE_GROUPS:
                continue
            if total_length < FAST_REVIEW_DARK_STAR_LINE_MIN_TOTAL_LENGTH:
                continue

            left = max(0, min(xs))
            top = max(0, min(ys))
            right = min(width, max(xs) + 1)
            bottom = min(height, max(ys) + 1)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            box_area = float(box_w * box_h)
            frame_area = float(max(1, width * height))
            box_area_ratio = box_area / frame_area
            if (
                box_area_ratio < FAST_REVIEW_DARK_STAR_LINE_MIN_BOX_AREA_RATIO
                or box_area_ratio > FAST_REVIEW_DARK_STAR_LINE_MAX_BOX_AREA_RATIO
            ):
                continue
            aspect_ratio = float(max(box_w, box_h)) / float(max(1, min(box_w, box_h)))
            if (
                aspect_ratio < FAST_REVIEW_DARK_STAR_LINE_MIN_ASPECT
                or aspect_ratio > FAST_REVIEW_DARK_STAR_LINE_MAX_ASPECT
            ):
                continue

            box_gray = gray[top:bottom, left:right]
            box_dark = local_dark[top:bottom, left:right]
            if box_gray.size <= 0 or box_dark.size <= 0:
                continue
            local_pad = max(10, int(min(width, height) * 0.035))
            local_left = max(0, left - local_pad)
            local_top = max(0, top - local_pad)
            local_right = min(width, right + local_pad)
            local_bottom = min(height, bottom + local_pad)
            local_gray = gray[local_top:local_bottom, local_left:local_right]
            if local_gray.size <= 0:
                continue

            signed_delta = float(np.mean(box_gray)) - float(np.mean(local_gray))
            dark_fraction = float(np.mean(box_dark >= dark_threshold))
            if (
                signed_delta > FAST_REVIEW_DARK_STAR_LINE_MIN_SIGNED_DELTA
                and dark_fraction < FAST_REVIEW_DARK_STAR_LINE_MIN_DARK_FRACTION
            ):
                continue
            bright_p95 = float(np.percentile(box_gray, 95.0))
            bright_fraction = float(np.mean(box_gray >= float(np.mean(local_gray)) + 20.0))
            bright_scratch_like = (
                bright_p95 >= FAST_REVIEW_DARK_STAR_LINE_BRIGHT_SCRATCH_MIN_P95
                and bright_p95 - float(np.mean(local_gray)) >= FAST_REVIEW_DARK_STAR_LINE_BRIGHT_SCRATCH_MIN_DELTA
                and bright_fraction >= FAST_REVIEW_DARK_STAR_LINE_BRIGHT_SCRATCH_MIN_FRACTION
            )
            if bright_scratch_like:
                continue
            score = total_length + len(cluster_angle_groups) * 90.0 + dark_fraction * 260.0 + abs(min(0.0, signed_delta)) * 28.0
            if score <= best_score:
                continue
            best_score = score
            best = {
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "box_area": int(box_area),
                "dark_fraction": dark_fraction,
                "signed_delta": signed_delta,
                "local_dark": float(np.mean(box_dark)),
                "line_count": int(len(cluster)),
                "angle_groups": int(len(cluster_angle_groups)),
                "total_length": total_length,
            }

        if best is None:
            return None

        pad = max(4, int(min(width, height) * 0.012))
        left = best["left"]
        top = best["top"]
        right = best["right"]
        bottom = best["bottom"]
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(width, right + pad)
        bottom = min(height, bottom + pad)
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        return {
            "type": "stain",
            "confidence": 0.91,
            "x": int(left),
            "y": int(top),
            "w": int(box_w),
            "h": int(box_h),
            "area": int(best["box_area"]),
            "length": int(max(box_w, box_h)),
            "aspect_ratio": round(float(max(box_w, box_h)) / float(max(1, min(box_w, box_h))), 2),
            "density": round(best["dark_fraction"], 2),
            "angle_groups": int(best["angle_groups"]),
            "line_count": int(best["line_count"]),
            "total_line_length": round(float(best["total_length"]), 1),
            "brightness_signed_delta": round(float(best["signed_delta"]), 1),
            "local_dark_delta": round(float(best["local_dark"]), 1),
            "level": "medium",
            "source": "pc_fast_dark_star_stain_lines",
        }

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
        curvilinear_detection = self.fast_cv_find_curvilinear_scratch(gray, mask, edge_distance)
        if curvilinear_detection is not None:
            if (
                line_detection is None
                or curvilinear_detection.get("total_line_length", 0) > line_detection.get("total_line_length", 0) * 0.64
            ):
                line_detection = curvilinear_detection
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
        fill_ratio = float(total_area) / float(max(1, (right - left) * (bottom - top)))
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
            "fill_ratio": round(fill_ratio, 2),
            "component_count": int(len(selected)),
            "strong_internal_scratch": (
                length >= 48
                and short_side >= 16
                and aggregate_aspect >= 1.8
                and box_area_ratio <= 0.08
            ),
        }

    def lsd_scratch_segments(self, gray, candidate_mask, mask, edge_distance, min_edge_distance, min_line_length):
        if not hasattr(cv2, "createLineSegmentDetector"):
            return []
        try:
            detector = cv2.createLineSegmentDetector()
            lens_values = gray[mask > 0]
            if lens_values.size <= 0:
                return []
            lsd_gray = gray.copy()
            lsd_gray[mask == 0] = int(np.median(lens_values))
            lines = detector.detect(lsd_gray)[0]
        except Exception:
            return []
        if lines is None:
            return []

        height, width = gray.shape[:2]
        segments = []
        seen = set()
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(round(float(value))) for value in line]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < min_line_length:
                continue
            center_x = max(0, min(width - 1, int((x1 + x2) / 2)))
            center_y = max(0, min(height - 1, int((y1 + y2) / 2)))
            if mask[center_y, center_x] == 0:
                continue

            samples = max(3, int(length / 7.0))
            active_count = 0
            distances = []
            for sample_index in range(samples + 1):
                ratio = float(sample_index) / float(samples)
                px = max(0, min(width - 1, int(round(x1 + (x2 - x1) * ratio))))
                py = max(0, min(height - 1, int(round(y1 + (y2 - y1) * ratio))))
                if mask[py, px] > 0 and candidate_mask[py, px] > 0:
                    active_count += 1
                if edge_distance is not None:
                    distances.append(float(edge_distance[py, px]))
            active_fraction = float(active_count) / float(samples + 1)
            if active_fraction < 0.22:
                continue
            if distances and float(np.percentile(distances, 35)) < min_edge_distance * 0.70:
                continue

            key = (
                int(round(x1 / 4.0)),
                int(round(y1 / 4.0)),
                int(round(x2 / 4.0)),
                int(round(y2 / 4.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            segments.append((x1, y1, x2, y2, length))
        return segments

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

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        segments = []
        if lines is not None:
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

        segments.extend(self.lsd_scratch_segments(
            gray,
            candidate_mask,
            mask,
            edge_distance,
            min_edge_distance,
            min_line_length,
        ))

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

    def fast_cv_find_curvilinear_scratch(self, gray, mask, edge_distance=None):
        height, width = gray.shape[:2]
        if width < 120 or height < 90:
            return None

        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=13, sigmaY=13)
        signed_diff = gray.astype(np.int16) - local_background.astype(np.int16)
        line_response = np.maximum(np.abs(signed_diff), 0).astype(np.uint8)
        mask_values = line_response[mask > 0]
        if mask_values.size <= 0:
            return None

        threshold = max(
            FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_LOCAL_DELTA,
            float(np.percentile(mask_values, 91.5)),
        )
        candidate_mask = np.where(line_response >= threshold, 255, 0).astype(np.uint8)
        candidate_mask = cv2.bitwise_and(candidate_mask, mask)

        gray_values = gray[mask > 0]
        if gray_values.size > 0:
            glare_limit = max(
                FAST_REVIEW_GLARE_MASK_MIN_GRAY,
                float(np.percentile(gray_values, 98.7)),
            )
            glare_mask = np.where(gray >= glare_limit, 255, 0).astype(np.uint8)
            glare_mask = cv2.dilate(glare_mask, np.ones((13, 13), dtype=np.uint8), iterations=1)
            candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(glare_mask))

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        diagonal_kernel = np.eye(7, dtype=np.uint8)
        anti_diagonal_kernel = np.fliplr(diagonal_kernel)
        line_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, vertical_kernel)
        line_mask = cv2.bitwise_or(line_mask, cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, horizontal_kernel))
        line_mask = cv2.bitwise_or(line_mask, cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, diagonal_kernel))
        line_mask = cv2.bitwise_or(line_mask, cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, anti_diagonal_kernel))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(line_mask, 8)
        if component_count <= 1:
            return None

        min_edge_distance = max(
            FAST_REVIEW_SCRATCH_MIN_EDGE_DISTANCE,
            int(min(width, height) * FAST_REVIEW_SCRATCH_EDGE_MARGIN_RATIO),
        )
        mask_area = max(1, int(cv2.countNonZero(mask)))
        components = []
        for index in range(1, component_count):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area < 6 or w <= 0 or h <= 0:
                continue
            length = max(w, h)
            short_side = max(1, min(w, h))
            aspect_ratio = float(length) / float(short_side)
            if length < 12 or aspect_ratio < 1.45:
                continue
            center_x = max(0, min(width - 1, int(x + w / 2)))
            center_y = max(0, min(height - 1, int(y + h / 2)))
            if mask[center_y, center_x] == 0:
                continue
            if edge_distance is not None and edge_distance[center_y, center_x] < min_edge_distance * 0.86:
                continue
            components.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "length": length,
                "aspect_ratio": aspect_ratio,
            })

        if len(components) < FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_COMPONENTS:
            return None

        clusters = []
        for component in sorted(components, key=lambda item: item["length"], reverse=True):
            cx = component["x"] + component["w"] / 2.0
            cy = component["y"] + component["h"] / 2.0
            best_cluster = None
            best_distance = None
            for cluster in clusters:
                xs = []
                ys = []
                for item in cluster:
                    xs.extend([item["x"], item["x"] + item["w"]])
                    ys.extend([item["y"], item["y"] + item["h"]])
                ccx = (min(xs) + max(xs)) / 2.0
                ccy = (min(ys) + max(ys)) / 2.0
                distance_value = max(abs(cx - ccx), abs(cy - ccy))
                if distance_value <= 42 and (best_distance is None or distance_value < best_distance):
                    best_cluster = cluster
                    best_distance = distance_value
            if best_cluster is None:
                clusters.append([component])
            else:
                best_cluster.append(component)

        best = None
        best_score = 0.0
        for cluster in clusters:
            if len(cluster) < FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_COMPONENTS:
                continue
            left = min(item["x"] for item in cluster)
            top = min(item["y"] for item in cluster)
            right = max(item["x"] + item["w"] for item in cluster)
            bottom = max(item["y"] + item["h"] for item in cluster)
            left = max(0, left - 3)
            top = max(0, top - 3)
            right = min(width - 1, right + 4)
            bottom = min(height - 1, bottom + 4)
            box_w = max(1, right - left)
            box_h = max(1, bottom - top)
            box_area = box_w * box_h
            box_ratio = float(box_area) / float(mask_area)
            if box_ratio > FAST_REVIEW_CURVILINEAR_SCRATCH_MAX_BOX_RATIO:
                continue
            length = max(box_w, box_h)
            short_side = max(1, min(box_w, box_h))
            aspect_ratio = float(length) / float(short_side)
            if aspect_ratio < FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_ASPECT:
                continue

            roi_mask = mask[top:bottom, left:right]
            if roi_mask.size <= 0:
                continue
            mask_overlap = float(cv2.countNonZero(roi_mask)) / float(max(1, box_area))
            if mask_overlap < FAST_REVIEW_SCRATCH_MIN_MASK_OVERLAP:
                continue
            line_roi = line_mask[top:bottom, left:right]
            active_area = int(cv2.countNonZero(line_roi))
            if active_area <= 0:
                continue
            fill_ratio = float(active_area) / float(max(1, box_area))
            if fill_ratio > FAST_REVIEW_CURVILINEAR_SCRATCH_MAX_FILL_RATIO:
                continue
            total_length = float(sum(item["length"] for item in cluster))
            if total_length < FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_TOTAL_LENGTH:
                continue
            diff_roi = line_response[top:bottom, left:right]
            local_peak = float(np.percentile(diff_roi[line_roi > 0], 82.0))
            if local_peak < FAST_REVIEW_CURVILINEAR_SCRATCH_MIN_LOCAL_DELTA:
                continue
            if edge_distance is not None and not self.scratch_box_has_safe_edge_distance(
                edge_distance,
                left,
                top,
                right,
                bottom,
                min_edge_distance,
                0.88,
            ):
                continue

            signed_roi = signed_diff[top:bottom, left:right]
            active_signed = signed_roi[line_roi > 0]
            local_signed_delta = float(np.mean(active_signed)) if active_signed.size else 0.0
            local_bright_delta = float(np.percentile(np.maximum(active_signed, 0), 85.0)) if active_signed.size else 0.0
            local_dark_delta = float(np.percentile(np.maximum(-active_signed, 0), 85.0)) if active_signed.size else 0.0
            score = total_length + local_peak * 4.0 + len(cluster) * 22.0
            if score > best_score:
                best_score = score
                confidence = min(0.91, 0.78 + min(0.07, total_length / 700.0) + min(0.04, local_peak / 160.0))
                best = {
                    "type": "scratch",
                    "confidence": round(confidence, 2),
                    "x": int(left),
                    "y": int(top),
                    "w": int(box_w),
                    "h": int(box_h),
                    "area": int(active_area),
                    "length": int(length),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "level": "medium" if total_length >= 120 or length >= 70 else "light",
                    "source": "pc_fast_curvilinear_scratch",
                    "line_count": len(cluster),
                    "total_line_length": round(total_length, 1),
                    "fill_ratio": round(fill_ratio, 2),
                    "local_signed_delta": round(local_signed_delta, 1),
                    "local_bright_delta": round(local_bright_delta, 1),
                    "local_dark_delta": round(local_dark_delta, 1),
                    "strong_internal_scratch": bool(total_length >= 72.0 and fill_ratio <= 0.26),
                    "candidates": [
                        {
                            "x": int(item["x"]),
                            "y": int(item["y"]),
                            "w": int(item["w"]),
                            "h": int(item["h"]),
                            "area": int(item["area"]),
                            "length": int(item["length"]),
                            "aspect_ratio": round(float(item["aspect_ratio"]), 2),
                        }
                        for item in cluster
                    ],
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
        if width < 180 or height < 120:
            return None
        frame_area = max(1, int(width * height))
        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=9)
        bright_diff = np.maximum(gray.astype(np.int16) - local_background.astype(np.int16), 0).astype(np.uint8)
        dark_diff = np.maximum(local_background.astype(np.int16) - gray.astype(np.int16), 0).astype(np.uint8)

        mask_points = cv2.findNonZero(mask)
        if mask_points is None:
            return None
        mx, my, mw, mh = cv2.boundingRect(mask_points)
        left_limit = max(0, int(mx + mw * 0.08))
        right_limit = min(width, int(mx + mw * 0.92))
        top_limit = max(0, int(my + mh * 0.08))
        bottom_limit = min(height, int(my + mh * 0.92))
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
            length = max(box_w, box_h)
            if length < max(46, int(min(width, height) * 0.18)):
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
            gray_roi = gray[top:bottom, left:right]
            dark_fraction = 0.0
            if gray_roi.size > 0:
                lens_gray_values = gray[mask > 0]
                dark_limit = min(
                    float(np.percentile(gray_roi, 35.0)) + 20.0,
                    float(np.percentile(lens_gray_values, 45.0)) if lens_gray_values.size else 80.0,
                    80.0,
                )
                dark_fraction = float(np.mean(gray_roi <= dark_limit))
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

            aspect_ratio = float(length) / float(max(1, min(box_w, box_h)))
            if aspect_ratio < 1.35:
                continue
            slender_ratio = float(slender_count) / float(max(1, len(cluster)))
            weak_center_cross = local_peak <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_DELTA
            if (
                weak_center_cross
                and total_line_length < FAST_REVIEW_CENTER_CROSS_WEAK_MIN_TOTAL_LENGTH
                and (
                    slender_ratio < FAST_REVIEW_CENTER_CROSS_WEAK_MIN_SLENDER_RATIO
                    or fill_ratio < FAST_REVIEW_CENTER_CROSS_WEAK_MIN_FILL_RATIO
                )
            ):
                continue
            if (
                aspect_ratio <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_ASPECT
                and local_peak <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_DELTA
                and slender_ratio <= FAST_REVIEW_REFLECTION_CENTER_CROSS_MAX_SLENDER_RATIO
            ):
                continue
            dark_star_stain_like = (
                center_x >= width * 0.56
                and box_w >= width * 0.18
                and aspect_ratio <= 2.25
                and local_peak <= FAST_REVIEW_DARK_STAR_STAIN_MAX_BRIGHT_DELTA
                and fill_ratio <= 0.32
                and len(angle_groups) >= 4
                and dark_fraction >= FAST_REVIEW_DARK_STAR_STAIN_MIN_DARK_FRACTION
            )
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
                    "dark_fraction": round(dark_fraction, 2),
                    "dark_star_stain_like": bool(dark_star_stain_like),
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

    def stage2_defect_matches_locked_black_stain(self, black_stain, candidate):
        if not isinstance(black_stain, dict) or not isinstance(candidate, dict):
            return False
        if candidate.get("type") != "stain":
            return False
        if self.fast_review_iou(black_stain, candidate) >= 0.16:
            return True
        return self.detection_center_distance_ratio(black_stain, candidate) <= 0.34

    def stage2_score_peak_for_defect(self, score_map, defect):
        if score_map is None or not hasattr(score_map, "shape") or not isinstance(defect, dict):
            return 0.0
        x = self._positive_int(defect.get("x"))
        y = self._positive_int(defect.get("y"))
        w = self._positive_int(defect.get("w"))
        h = self._positive_int(defect.get("h"))
        if w <= 0 or h <= 0:
            return 0.0
        height, width = score_map.shape[:2]
        left = max(0, min(int(x), int(width)))
        top = max(0, min(int(y), int(height)))
        right = max(left, min(int(x + w), int(width)))
        bottom = max(top, min(int(y + h), int(height)))
        score_roi = score_map[top:bottom, left:right]
        if score_roi.size <= 0:
            return 0.0
        return float(np.max(score_roi))

    def preserve_locked_black_stains_after_stage2(self, rule_defects, merged_defects, stage2_result):
        kept = [dict(defect) for defect in (merged_defects or []) if isinstance(defect, dict)]
        protected = [
            defect for defect in (rule_defects or [])
            if self.detection_looks_like_locked_black_stain(defect)
        ]
        preserved_count = 0
        if not protected:
            if isinstance(stage2_result, dict):
                stage2_result["preserved_black_stain_count"] = 0
            return kept, stage2_result

        score_map = stage2_result.get("score_map") if isinstance(stage2_result, dict) else None
        for defect in protected:
            if any(self.stage2_defect_matches_locked_black_stain(defect, old) for old in kept):
                continue
            preserved = dict(defect)
            preserved["stage2_confirmed"] = False
            preserved["stage2_preserved_black_stain"] = True
            preserved["stage2_reason"] = "center_black_stain_guard"
            preserved["stage2_score"] = round(self.stage2_score_peak_for_defect(score_map, defect), 4)
            preserved["confidence"] = round(max(self.confidence_float(defect), 0.90), 2)
            kept.append(preserved)
            preserved_count += 1

        if preserved_count:
            kept.sort(
                key=lambda item: (
                    1 if item.get("stage2_preserved_black_stain") else 0,
                    self.confidence_float(item),
                    self._positive_int(item.get("area")),
                ),
                reverse=True,
            )
        if isinstance(stage2_result, dict):
            stage2_result["preserved_black_stain_count"] = preserved_count
        return kept, stage2_result

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
        rule_defects = self.latest_detection_result.get("defects") or []
        merged_defects, stage2_result = merge_with_rule_defects(
            rule_defects,
            stage2_result,
        )
        merged_defects, stage2_result = self.preserve_locked_black_stains_after_stage2(
            rule_defects,
            merged_defects,
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
            "unconfirmed_rule_count": int(stage2_result.get("unconfirmed_rule_count", 0)),
            "added_stage2_count": int(stage2_result.get("added_stage2_count", 0)),
            "preserved_black_stain_count": int(stage2_result.get("preserved_black_stain_count", 0)),
        }
        refined = self.normalize_detection_result(refined)
        refined = self.filter_edge_defects_for_image(refined, payload, image)
        refined = self.stabilize_detection_result(refined)
        self.update_result_ui(refined, render_live_image=False)

    def build_detection_mask(self, image, payload, result):
        if cv2 is None or np is None:
            return None

        roi, source_w, source_h = self.resolve_detection_roi_for_image(image, payload, result)
        if not isinstance(roi, dict):
            return None
        self._last_effective_roi = dict(roi)

        image_w, image_h = image.size
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

        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        center = (x + w // 2, y + h // 2)
        axes = (max(2, int(w * 0.44)), max(2, int(h * 0.40)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        edge_ignore = max(8, int(min(w, h) * FAST_REVIEW_INNER_MASK_MARGIN_RATIO))
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        inner_mask = np.where(distance > edge_ignore, 255, 0).astype(np.uint8)
        if cv2.countNonZero(inner_mask) <= 0:
            fallback_ignore = max(4, int(min(w, h) * 0.10))
            inner_mask = np.where(distance > fallback_ignore, 255, 0).astype(np.uint8)
        return inner_mask

    def resolve_detection_roi_for_image(self, image, payload, result):
        image_w, image_h = image.size
        result = result if isinstance(result, dict) else {}
        payload = payload if isinstance(payload, dict) else {}
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

        auto_roi = self.detect_lens_roi_from_image(image)
        if auto_roi is None:
            auto_roi = self.detect_conveyor_workpiece_roi_from_image(image)
        if auto_roi is not None:
            scaled_auto = {
                "x": int(auto_roi["x"] * source_w / float(max(1, image_w))),
                "y": int(auto_roi["y"] * source_h / float(max(1, image_h))),
                "w": int(auto_roi["w"] * source_w / float(max(1, image_w))),
                "h": int(auto_roi["h"] * source_h / float(max(1, image_h))),
                "source": "pc_auto_roi",
            }
            if roi is None or self.roi_is_full_frame(roi, source_w, source_h):
                roi = scaled_auto
            elif self.roi_should_prefer_auto(roi, scaled_auto, source_w, source_h):
                roi = scaled_auto

        if roi is None:
            roi = self.center_fallback_roi(source_w, source_h)
        return roi, source_w, source_h

    def roi_should_prefer_auto(self, roi, auto_roi, source_w, source_h):
        if not isinstance(roi, dict) or not isinstance(auto_roi, dict):
            return False
        if str(roi.get("source", "")).lower() == "pc_yolo_center_guard":
            return False
        if self.roi_is_full_frame(roi, source_w, source_h):
            return True
        current_area = self._positive_int(roi.get("w")) * self._positive_int(roi.get("h"))
        auto_area = self._positive_int(auto_roi.get("w")) * self._positive_int(auto_roi.get("h"))
        frame_area = max(1, source_w * source_h)
        if current_area <= 0 or auto_area <= 0:
            return False
        current_ratio = float(current_area) / float(frame_area)
        auto_ratio = float(auto_area) / float(frame_area)
        if current_ratio >= 0.64 and 0.08 <= auto_ratio <= 0.62:
            return True
        if auto_area < current_area * 0.55 and auto_ratio >= 0.08:
            return True
        return False

    def build_stage2_analysis_mask(self, image, payload, result):
        return self.build_detection_mask(image, payload, result)

    def detect_lens_roi_from_image(self, image):
        if cv2 is None or np is None:
            return None

        image_w, image_h = image.size
        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = self.enhance_gray_for_inspection(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 90)
        edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=1)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = float(max(1, image_w * image_h))
        min_area = frame_area * FAST_REVIEW_AUTO_ROI_MIN_AREA_RATIO
        min_box_area = frame_area * 0.045
        min_w = max(24, int(image_w * FAST_REVIEW_AUTO_ROI_MIN_WIDTH_RATIO))
        min_h = max(18, int(image_h * FAST_REVIEW_AUTO_ROI_MIN_HEIGHT_RATIO))
        best_box = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            box_area = float(max(1, w * h))
            if area < min_area and box_area < min_box_area:
                continue
            if w < min_w or h < min_h:
                continue
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            if aspect_ratio > 3.6:
                continue
            fill_ratio = area / box_area
            if fill_ratio < 0.035 and box_area < frame_area * 0.10:
                continue
            center_x = (x + w / 2.0) / float(max(1, image_w))
            center_y = (y + h / 2.0) / float(max(1, image_h))
            center_penalty = abs(center_x - 0.45) * 0.18 + abs(center_y - 0.58) * 0.18
            bottom_bonus = min(0.18, max(0.0, (y + h) / float(max(1, image_h)) - 0.55))
            size_score = min(0.45, box_area / frame_area)
            score = size_score + fill_ratio * 0.16 + bottom_bonus - center_penalty
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

    def detect_conveyor_workpiece_roi_from_image(self, image):
        if cv2 is None or np is None:
            return None

        image_w, image_h = image.size
        if image_w <= 0 or image_h <= 0:
            return None

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray = self.enhance_gray_for_inspection(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 24, 82)
        top_ignore = int(image_h * 0.20)
        if top_ignore > 0:
            edges[:top_ignore, :] = 0
        edges = cv2.dilate(edges, np.ones((4, 4), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((13, 13), dtype=np.uint8), iterations=1)

        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(edges, 8)
        if component_count <= 1:
            return None

        frame_area = float(max(1, image_w * image_h))
        min_box_area = frame_area * 0.035
        min_w = max(28, int(image_w * 0.16))
        min_h = max(22, int(image_h * 0.12))
        best_box = None
        best_score = 0.0
        partial_boxes = []
        for index in range(1, component_count):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area >= max(18, int(frame_area * 0.0012)) and w >= 10 and h >= 10:
                center_x = (x + w / 2.0) / float(max(1, image_w))
                center_y = (y + h / 2.0) / float(max(1, image_h))
                if 0.16 <= center_x <= 0.88 and 0.22 <= center_y <= 0.98:
                    partial_boxes.append((x, y, w, h, area))
            if w < min_w or h < min_h:
                continue
            box_area = float(max(1, w * h))
            if box_area < min_box_area:
                continue
            area_ratio = box_area / frame_area
            if area_ratio > 0.74:
                continue
            aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
            if aspect_ratio > 3.8:
                continue
            roi = {"x": x, "y": y, "w": w, "h": h}
            if not self.roi_is_inside_conveyor_gate(roi, image_w, image_h):
                continue
            edge_fill = float(area) / box_area
            center_x = (x + w / 2.0) / float(max(1, image_w))
            center_y = (y + h / 2.0) / float(max(1, image_h))
            center_bonus = 1.0 - min(1.0, abs(center_x - 0.52) + abs(center_y - 0.62))
            score = min(0.60, area_ratio) + min(0.25, edge_fill * 1.8) + center_bonus * 0.15
            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        union_box = self.union_conveyor_workpiece_components(partial_boxes, image_w, image_h)
        if union_box is not None:
            x, y, w, h = union_box
            box_area = float(max(1, w * h))
            area_ratio = box_area / frame_area
            edge_area = float(sum(item[4] for item in partial_boxes))
            edge_fill = edge_area / box_area
            center_x = (x + w / 2.0) / float(max(1, image_w))
            center_y = (y + h / 2.0) / float(max(1, image_h))
            center_bonus = 1.0 - min(1.0, abs(center_x - 0.52) + abs(center_y - 0.64))
            score = min(0.64, area_ratio) + min(0.22, edge_fill * 1.6) + center_bonus * 0.18
            if score > best_score:
                best_score = score
                best_box = union_box

        if best_box is None:
            return None

        x, y, w, h = best_box
        padding = max(6, int(min(w, h) * 0.05))
        x = max(0, x - padding)
        y = max(0, y - padding)
        right = min(image_w, x + w + padding * 2)
        bottom = min(image_h, y + h + padding * 2)
        return {
            "x": int(x),
            "y": int(y),
            "w": int(max(1, right - x)),
            "h": int(max(1, bottom - y)),
        }

    def union_conveyor_workpiece_components(self, boxes, image_w, image_h):
        if not boxes:
            return None

        boxes = sorted(boxes, key=lambda item: item[4], reverse=True)[:12]
        left = min(item[0] for item in boxes)
        top = min(item[1] for item in boxes)
        right = max(item[0] + item[2] for item in boxes)
        bottom = max(item[1] + item[3] for item in boxes)
        w = max(1, right - left)
        h = max(1, bottom - top)
        roi = {"x": left, "y": top, "w": w, "h": h}
        if not self.roi_is_inside_conveyor_gate(roi, image_w, image_h):
            return None

        frame_area = float(max(1, image_w * image_h))
        area_ratio = float(w * h) / frame_area
        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        if area_ratio < 0.045 or area_ratio > 0.74 or aspect_ratio > 3.8:
            return None

        component_area = sum(item[4] for item in boxes)
        if component_area < frame_area * 0.004 and len(boxes) < 3:
            return None
        return (left, top, w, h)

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

        draw = ImageDraw.Draw(image)
        self.draw_conveyor_gate(draw, image)
        defects = self.latest_detection_result.get("defects") or []
        if not defects:
            return

        frame = self.latest_detection_result.get("frame") or {}
        source_w = self._positive_int(frame.get("w")) or self._positive_int(payload.get("width")) or image.width
        source_h = self._positive_int(frame.get("h")) or self._positive_int(payload.get("height")) or image.height
        x_scale = float(image.width) / float(source_w)
        y_scale = float(image.height) / float(source_h)
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
            polygon = self.scaled_mask_polygon(defect, x_scale, y_scale, image.width, image.height)
            if polygon and len(polygon) >= 3:
                draw.line(polygon + [polygon[0]], fill=color, width=outline_width)
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

    def draw_conveyor_gate(self, draw, image):
        if not self.conveyor_center_gate_var.get():
            return
        gate = self.conveyor_gate_rect(image.width, image.height)
        left = gate["x"]
        top = gate["y"]
        right = gate["x"] + gate["w"]
        bottom = gate["y"] + gate["h"]
        width = max(1, int(min(image.width, image.height) / 240))
        color = (64, 180, 220)
        draw.rectangle((left, top, right, bottom), outline=color, width=width)
        draw.text((left + 4, max(0, top - 16)), "中心检测区", fill=color)

    def scaled_mask_polygon(self, defect, x_scale, y_scale, image_w, image_h):
        polygon = defect.get("mask_polygon")
        if not isinstance(polygon, list):
            return []
        scaled = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = max(0, min(image_w - 1, int(float(point[0]) * x_scale)))
                y = max(0, min(image_h - 1, int(float(point[1]) * y_scale)))
            except (TypeError, ValueError):
                continue
            scaled.append((x, y))
        return scaled

    def _positive_int(self, value):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def format_lens_track_text(self, lens):
        track_name = self.current_mode_config().track_status_name
        if not isinstance(lens, dict):
            return "%s：暂无数据" % track_name

        found = bool(lens.get("found", False))
        status = "已锁定" if found else "未找到"
        source = lens.get("source", "")
        confidence = lens.get("confidence", 0)
        try:
            confidence_text = "%.2f" % float(confidence)
        except (TypeError, ValueError):
            confidence_text = str(confidence)

        return (
            "%s：%s，中心 (%s, %s)，ROI %sx%s+%s+%s，置信度 %s，%s"
            % (
                track_name,
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
        self.lens_track_var.set("%s：暂无数据" % self.current_mode_config().track_status_name)
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

        prefix = "lens_defect_history" if self.active_mode_key == "lens" else "slide_defect_history"
        default_name = "%s_%s.csv" % (prefix, datetime.now().strftime("%Y%m%d_%H%M%S"))
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
        self.mcu_client.disconnect()
        self.root.destroy()


def main():
    if "--self-test-runtime" in sys.argv:
        raise SystemExit(runtime_self_test())
    try:
        if not acquire_single_instance_lock():
            return
        root = tk.Tk()
        LensDefectHostApp(root)
        root.mainloop()
    except Exception:
        try:
            log_path = PROJECT_ROOT / "outputs" / "host_startup_error.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise


def runtime_self_test():
    checks = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "app_dir": str(APP_DIR),
        "project_root": str(PROJECT_ROOT),
        "cv2": None,
        "numpy": None,
        "stage2": None,
        "yolo": None,
        "yolo_seg": None,
    }
    if cv2 is None:
        checks["cv2"] = {"ok": False, "error": str(STAGE2_IMPORT_ERROR)}
    else:
        checks["cv2"] = {"ok": True, "version": getattr(cv2, "__version__", "")}
    if np is None:
        checks["numpy"] = {"ok": False, "error": str(STAGE2_IMPORT_ERROR)}
    else:
        checks["numpy"] = {"ok": True, "version": getattr(np, "__version__", "")}

    stage2_ok = ensure_stage2_runtime_loaded()
    checks["stage2"] = {"ok": bool(stage2_ok), "error": "" if stage2_ok else str(STAGE2_IMPORT_ERROR)}
    try:
        detector = YoloOnnxDetector(DEFAULT_YOLO_MODEL, DEFAULT_YOLO_LABELS)
        checks["yolo"] = {
            "ok": True,
            "model": str(detector.model_path),
            "labels": detector.labels,
            "input_size": detector.input_size,
        }
    except Exception as exc:
        checks["yolo"] = {"ok": False, "error": str(exc), "model": str(DEFAULT_YOLO_MODEL)}

    if DEFAULT_YOLO_SEG_MODEL.exists():
        try:
            detector = YoloOnnxDetector(DEFAULT_YOLO_SEG_MODEL, DEFAULT_YOLO_LABELS)
            checks["yolo_seg"] = {
                "ok": True,
                "model": str(detector.model_path),
                "task": detector.task,
                "labels": detector.labels,
                "input_size": detector.input_size,
            }
        except Exception as exc:
            checks["yolo_seg"] = {"ok": False, "error": str(exc), "model": str(DEFAULT_YOLO_SEG_MODEL)}
    else:
        checks["yolo_seg"] = {"ok": True, "available": False, "model": str(DEFAULT_YOLO_SEG_MODEL)}

    ok = bool(checks["cv2"]["ok"] and checks["numpy"]["ok"] and checks["stage2"]["ok"] and checks["yolo"]["ok"])
    text = json.dumps(checks, ensure_ascii=False, indent=2)
    try:
        output_path = PROJECT_ROOT / "outputs" / "runtime_self_test.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except Exception:
        pass
    print(text)
    return 0 if ok else 2


if __name__ == "__main__":
    main()
