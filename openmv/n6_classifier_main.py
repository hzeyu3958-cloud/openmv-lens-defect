# OpenMV N6 lens defect classifier.
#
# Workflow:
# 1. Capture images with n6_usb_image_capture.py.
# 2. Train on the PC with training/train_lens_classifier.py.
# 3. Copy lens_defect_classifier_int8.tflite and lens_defect_labels.txt to
#    the N6 USB drive.
# 4. Run this script on the N6. It emits one JSON line per result over USB VCP
#    and, optionally, UART3 for a Bluetooth serial module.

import csi
import image
import sys
import time
import ujson
import pyb
import ml

try:
    from machine import UART
except Exception:
    UART = None


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

MODEL_CANDIDATES = (
    "/rom/lens_defect_classifier_int8.tflite",
    "/lens_defect_classifier_int8.tflite",
    "/flash/lens_defect_classifier_int8.tflite",
    "/sdcard/lens_defect_classifier_int8.tflite",
)

LABEL_CANDIDATES = (
    "/rom/lens_defect_labels.txt",
    "/lens_defect_labels.txt",
    "/flash/lens_defect_labels.txt",
    "/sdcard/lens_defect_labels.txt",
    "/rom/lens_defect_classifier_int8.txt",
    "/lens_defect_classifier_int8.txt",
    "/flash/lens_defect_classifier_int8.txt",
    "/sdcard/lens_defect_classifier_int8.txt",
)

PIXFORMAT = csi.RGB565
FRAMESIZE = csi.XGA
STARTUP_STABLE_MS = 2000

# Match the data-collection crop in n6_usb_image_capture.py.
# Ratios map to roughly (160, 60, 460, 300) on a 640 x 400 frame.
INFERENCE_ROI_RATIO = (0.25, 0.15, 0.72, 0.75)
MODEL_INPUT_SIZE_FALLBACK = 128

CONFIDENCE_THRESHOLD = 0.45
SEND_INTERVAL_MS = 300
DRAW_DEBUG = False
PRINT_JSON_TO_IDE = False

# Temporal stability: require repeated agreement before changing the result.
STABLE_DEFECT_CONFIRM_FRAMES = 3
STABLE_DEFECT_SWITCH_FRAMES = 4
STABLE_NORMAL_CONFIRM_FRAMES = 5
STABLE_DEFECT_HOLD_FRAMES = 6
DEFECT_SWITCH_MIN_SCORE = 0.52
NORMAL_SWITCH_MIN_SCORE = 0.55

ENABLE_USB_IMAGE_STREAM = True
USB_IMAGE_INTERVAL_MS = 300
USB_IMAGE_JPEG_QUALITY = 40
USB_IMAGE_JPEG_SUBSAMPLING = None
SEND_PREVIEW_ROI_ONLY = True

DISABLE_AUTO_GAIN_AFTER_START = True
DISABLE_AUTO_WHITEBAL_AFTER_START = True

# 预留给单片机的口：开启后会把同一条检测 JSON 从 UART 发出。
# 接线示例：OpenMV TX -> 单片机 RX，OpenMV GND -> 单片机 GND。
ENABLE_UART_OUTPUT = False
UART_ID = 3
UART_BAUDRATE = 115200

DEFAULT_LABELS = (
    "normal",
    "scratch",
    "stain",
)

SUPPORTED_LABELS = DEFAULT_LABELS

LEVEL_MAP = {
    "normal": "normal",
    "scratch": "medium",
    "stain": "light",
}

SUMMARY_TYPES = (
    "scratch",
    "stain",
)


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


def file_exists(path):
    try:
        f = open(path, "rb")
        f.close()
        return True
    except Exception:
        return False


def first_existing_path(paths):
    for path in paths:
        if file_exists(path):
            return path
    return None


def load_labels(paths, model):
    for path in paths:
        if not file_exists(path):
            continue
        labels = []
        try:
            with open(path, "r") as f:
                for line in f:
                    label = line.strip()
                    if label:
                        labels.append(label)
        except Exception:
            labels = []
        if labels:
            return labels

    try:
        if model.labels:
            return list(model.labels)
    except Exception:
        pass

    return list(DEFAULT_LABELS)


def safe_round(value, digits):
    scale = 1
    for _ in range(digits):
        scale *= 10
    return int(value * scale + 0.5) / scale


def clamp_float(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def flatten_scores(output):
    try:
        return [float(value) for value in output.flatten().tolist()]
    except Exception:
        pass

    scores = []

    def walk(value):
        try:
            for item in value:
                walk(item)
            return
        except Exception:
            pass
        try:
            scores.append(float(value))
        except Exception:
            pass

    walk(output)
    return scores


def get_input_size(model):
    try:
        shape = model.input_shape[0]
        if len(shape) >= 4:
            return int(shape[-2]), int(shape[-3])
        if len(shape) >= 3:
            return int(shape[-2]), int(shape[-3])
    except Exception:
        pass
    return MODEL_INPUT_SIZE_FALLBACK, MODEL_INPUT_SIZE_FALLBACK


def clipped_roi(img, roi):
    if roi is None:
        side = min(img.width(), img.height())
        return ((img.width() - side) // 2, (img.height() - side) // 2, side, side)

    x, y, w, h = roi
    x = max(0, min(x, img.width() - 1))
    y = max(0, min(y, img.height() - 1))
    w = max(1, min(w, img.width() - x))
    h = max(1, min(h, img.height() - y))
    return x, y, w, h


def scaled_roi(img, ratio):
    if ratio is None:
        return None
    x_ratio, y_ratio, w_ratio, h_ratio = ratio
    return (
        int(img.width() * x_ratio),
        int(img.height() * y_ratio),
        int(img.width() * w_ratio),
        int(img.height() * h_ratio),
    )


def prepare_model_image(img, input_w, input_h):
    roi = clipped_roi(img, scaled_roi(img, INFERENCE_ROI_RATIO))
    x_scale = float(input_w) / float(roi[2])
    y_scale = float(input_h) / float(roi[3])
    return img.copy(roi=roi, x_scale=x_scale, y_scale=y_scale, copy_to_fb=False), roi


def build_summary(defect_type):
    summary = {}
    for item in SUMMARY_TYPES:
        summary[item] = 0
    if defect_type in summary:
        summary[defect_type] = 1
    return summary


def build_result(label, score, roi, frame_w, frame_h):
    has_defect = label != "normal"
    level = LEVEL_MAP.get(label, "light") if has_defect else "normal"
    defects = []

    if has_defect:
        x, y, w, h = roi
        defects.append({
            "type": label,
            "confidence": safe_round(score, 2),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": w * h,
            "length": max(w, h),
            "aspect_ratio": safe_round(float(w) / float(max(1, h)), 2),
            "level": level,
        })

    return {
        "has_defect": has_defect,
        "defect_count": len(defects),
        "summary": build_summary(label),
        "overall_level": level,
        "defects": defects,
        "timestamp": time.ticks_ms(),
        "model": "lens_defect_classifier_int8",
        "frame": {"w": frame_w, "h": frame_h},
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


def send_json(result):
    send_line(ujson.dumps(result))


def send_usb_image(img, roi=None):
    if not ENABLE_USB_IMAGE_STREAM:
        return

    try:
        preview_roi = clipped_roi(img, roi) if (SEND_PREVIEW_ROI_ONLY and roi is not None) else None
        width = preview_roi[2] if preview_roi is not None else img.width()
        height = preview_roi[3] if preview_roi is not None else img.height()
        try:
            if USB_IMAGE_JPEG_SUBSAMPLING is None:
                raise TypeError("JPEG subsampling is not available")
            if preview_roi is not None:
                jpg = img.compress(
                    roi=preview_roi,
                    quality=USB_IMAGE_JPEG_QUALITY,
                    subsampling=USB_IMAGE_JPEG_SUBSAMPLING,
                )
            else:
                jpg = img.compress(quality=USB_IMAGE_JPEG_QUALITY, subsampling=USB_IMAGE_JPEG_SUBSAMPLING)
        except TypeError:
            if preview_roi is not None:
                jpg = img.compress(roi=preview_roi, quality=USB_IMAGE_JPEG_QUALITY)
            else:
                jpg = img.compress(quality=USB_IMAGE_JPEG_QUALITY)
        usb.write(("IMG_BEGIN %d %d %d\n" % (jpg.size(), width, height)).encode("utf-8"))
        usb.write(jpg)
        usb.write(b"IMG_END\n")
    except Exception as exc:
        try:
            usb.write(("ERR USB_IMAGE_STREAM_FAILED %s\n" % exc).encode("utf-8"))
        except Exception:
            pass
        if PRINT_JSON_TO_IDE:
            print("USB image stream failed:", exc)


def fatal_error_loop(message):
    print("N6 classifier error:", message)
    result = {
        "has_defect": False,
        "defect_count": 0,
        "summary": build_summary("normal"),
        "overall_level": "error",
        "defects": [],
        "timestamp": time.ticks_ms(),
        "error": message,
    }
    while True:
        result["timestamp"] = time.ticks_ms()
        send_json(result)
        time.sleep_ms(1000)


def best_label(scores, labels):
    if not scores:
        return "normal", 0.0

    best_index = 0
    best_score = float(scores[0])
    limit = min(len(scores), len(labels))
    for i in range(1, limit):
        score = float(scores[i])
        if score > best_score:
            best_score = score
            best_index = i

    best_score = clamp_float(best_score, 0.0, 1.0)
    label = labels[best_index] if best_index < len(labels) else "normal"
    if best_score < CONFIDENCE_THRESHOLD or label not in SUPPORTED_LABELS:
        label = "normal"
    return label, best_score


stable_label = "normal"
stable_score = 1.0
candidate_label = None
candidate_count = 0
candidate_score = 0.0
defect_hold_frames = 0


def stabilize_label(label, score):
    global stable_label, stable_score, candidate_label, candidate_count, candidate_score, defect_hold_frames

    if label == stable_label:
        stable_score = stable_score * 0.70 + score * 0.30
        candidate_label = None
        candidate_count = 0
        candidate_score = 0.0
        if label != "normal":
            defect_hold_frames = STABLE_DEFECT_HOLD_FRAMES
        return stable_label, stable_score

    if label != candidate_label:
        candidate_label = label
        candidate_count = 1
        candidate_score = score
    else:
        candidate_count += 1
        candidate_score = candidate_score * 0.55 + score * 0.45

    if label == "normal":
        if stable_label != "normal" and defect_hold_frames > 0:
            defect_hold_frames -= 1
        if candidate_count >= STABLE_NORMAL_CONFIRM_FRAMES and defect_hold_frames <= 0 and score >= NORMAL_SWITCH_MIN_SCORE:
            stable_label = "normal"
            stable_score = candidate_score
            candidate_label = None
            candidate_count = 0
            candidate_score = 0.0
        return stable_label, stable_score

    required = STABLE_DEFECT_CONFIRM_FRAMES if stable_label == "normal" else STABLE_DEFECT_SWITCH_FRAMES
    if candidate_count >= required and candidate_score >= DEFECT_SWITCH_MIN_SCORE:
        stable_label = label
        stable_score = candidate_score
        defect_hold_frames = STABLE_DEFECT_HOLD_FRAMES
        candidate_label = None
        candidate_count = 0
        candidate_score = 0.0
    return stable_label, stable_score


if ENABLE_UART_OUTPUT and UART is not None:
    try:
        uart = UART(UART_ID, baudrate=UART_BAUDRATE)
    except Exception as exc:
        print("UART init failed:", exc)
        uart = None

model_path = first_existing_path(MODEL_CANDIDATES)
if model_path is None:
    fatal_error_loop("model file not found; copy lens_defect_classifier_int8.tflite to the N6 drive")

try:
    model = ml.Model(model_path)
except Exception as exc:
    fatal_error_loop("failed to load model: %s" % exc)

labels = load_labels(LABEL_CANDIDATES, model)
input_w, input_h = get_input_size(model)
print("Loaded model:", model_path)
print("Input:", input_w, input_h, "Labels:", labels)

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
last_image_ms = time.ticks_ms()

while True:
    clock.tick()
    img = csi0.snapshot()
    net_img, source_roi = prepare_model_image(img, input_w, input_h)

    outputs = model.predict([net_img])
    scores = flatten_scores(outputs[0]) if outputs else []
    raw_label, raw_score = best_label(scores, labels)
    label, score = stabilize_label(raw_label, raw_score)
    result = build_result(label, score, source_roi, img.width(), img.height())
    result["raw_label"] = raw_label
    result["raw_score"] = safe_round(raw_score, 2)
    result["stability"] = {
        "stable_label": stable_label,
        "candidate_label": candidate_label or "",
        "candidate_count": candidate_count,
    }

    if DRAW_DEBUG:
        img.draw_rectangle(source_roi, color=(0, 255, 0))
        img.draw_string(4, 4, "%s %.2f" % (label, score), color=(255, 255, 255))
        if label != "normal":
            img.draw_rectangle(source_roi, color=(255, 0, 0))

    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
        send_json(result)
        last_send_ms = now_ms
    if time.ticks_diff(now_ms, last_image_ms) >= USB_IMAGE_INTERVAL_MS:
        send_usb_image(img, source_roi)
        last_image_ms = now_ms
