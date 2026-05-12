# OpenMV N6 分类模型运行脚本
# 用途：
# 1. 电脑训练得到 lens_defect_classifier_int8.tflite 和 lens_defect_labels.txt。
# 2. 用 Windows 上位机复制到 OpenMV N6 根目录。
# 3. 运行本脚本，N6 会拍图、模型分类、输出 JSON 给 Windows 上位机。

import csi
import image
import time
import ujson
import pyb
import ml


# ============================================================
# 可调参数
# ============================================================

MODEL_PATH = "/lens_defect_classifier_int8.tflite"
LABELS_PATH = "/lens_defect_labels.txt"

PIXFORMAT = csi.RGB565
FRAMESIZE = csi.QVGA
MODEL_INPUT_SIZE = 128
SEND_INTERVAL_MS = 500
CONFIDENCE_THRESHOLD = 0.45

DISABLE_AUTO_GAIN_AFTER_START = True
DISABLE_AUTO_WHITEBAL_AFTER_START = True
STARTUP_STABLE_MS = 2000

LEVEL_MAP = {
    "normal": "normal",
    "scratch": "medium",
    "dust": "light",
    "stain": "light",
    "coating_damage": "medium",
    "crack": "serious",
    "edge_damage": "medium",
    "unknown": "light",
}

DEFECT_TYPES = [
    "scratch",
    "dust",
    "stain",
    "coating_damage",
    "crack",
    "edge_damage",
    "unknown",
]


usb = pyb.USB_VCP()


def send_line(text):
    usb.write((text + "\n").encode("utf-8"))
    print(text)


def load_labels(path):
    labels = []
    try:
        with open(path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception:
        labels = ["normal", "scratch", "dust", "stain", "coating_damage", "crack", "edge_damage", "unknown"]
    return labels


def flatten_scores(output):
    scores = []
    try:
        for i in range(len(output)):
            value = output[i]
            try:
                for j in range(len(value)):
                    scores.append(float(value[j]))
            except Exception:
                scores.append(float(value))
    except Exception:
        pass
    return scores


def build_summary(defect_type):
    summary = {}
    for item in DEFECT_TYPES:
        summary[item] = 0
    if defect_type in summary:
        summary[defect_type] = 1
    elif defect_type != "normal":
        summary["unknown"] = 1
    return summary


def prepare_model_image(img):
    # 取中心正方形并缩放到模型输入尺寸，和训练脚本的 128x128 输入保持一致。
    side = min(img.width(), img.height())
    x = (img.width() - side) // 2
    y = (img.height() - side) // 2
    scale = float(MODEL_INPUT_SIZE) / float(side)
    return img.copy(roi=(x, y, side, side), x_scale=scale, y_scale=scale, copy_to_fb=True)


labels = load_labels(LABELS_PATH)
model = ml.Model(MODEL_PATH)

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
send_line("READY OpenMV_N6_CLASSIFIER")

while True:
    clock.tick()
    img = csi0.snapshot()
    net_img = prepare_model_image(img)

    outputs = model.predict([net_img])
    scores = flatten_scores(outputs[0]) if outputs else []

    best_index = 0
    best_score = 0.0
    for i, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_index = i

    if best_index < len(labels):
        defect_type = labels[best_index]
    else:
        defect_type = "unknown"

    if best_score < CONFIDENCE_THRESHOLD:
        defect_type = "unknown"

    has_defect = defect_type != "normal"
    level = LEVEL_MAP.get(defect_type, "light") if has_defect else "normal"

    defects = []
    if has_defect:
        defects.append({
            "type": defect_type,
            "confidence": round(best_score, 2),
            "x": 0,
            "y": 0,
            "w": img.width(),
            "h": img.height(),
            "area": img.width() * img.height(),
            "length": max(img.width(), img.height()),
            "aspect_ratio": round(float(img.width()) / float(img.height()), 2),
            "level": level,
        })
        img.draw_rectangle((0, 0, img.width() - 1, img.height() - 1), color=(255, 0, 0))

    img.draw_string(4, 4, "%s %.2f" % (defect_type, best_score), color=(255, 255, 255))

    result = {
        "has_defect": has_defect,
        "defect_count": len(defects),
        "summary": build_summary(defect_type),
        "overall_level": level,
        "defects": defects,
        "timestamp": time.ticks_ms(),
    }

    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
        send_line(ujson.dumps(result))
        last_send_ms = now_ms
