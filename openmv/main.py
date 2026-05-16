# OpenMV 眼镜片缺陷检测 MVP
# 功能：采集图像 -> 传统图像处理 -> 规则分类 -> JSON 串口发送
#
# 硬件串口示例：
# OpenMV TX -> 蓝牙模块 RX
# OpenMV RX -> 蓝牙模块 TX
# OpenMV GND -> 蓝牙模块 GND
# OpenMV VCC -> 蓝牙模块 VCC

import sensor
import image
import time
import ujson
from pyb import UART, USB_VCP


# ============================================================
# 1. 可调参数区
# ============================================================

# 摄像头参数
PIXFORMAT = sensor.GRAYSCALE
FRAMESIZE = sensor.QVGA
FRAME_SKIP_TIME_MS = 2000

# 只检测镜片所在区域：(x, y, w, h)
# QVGA 分辨率为 320x240，后期按你的夹具位置微调。
LENS_ROI = (40, 30, 240, 180)

# 图像预处理开关
ENABLE_HISTEQ = True          # 直方图均衡，增强对比度
ENABLE_DENOISE = True         # 中值滤波，降低小噪声
ENABLE_EDGE_CANDIDATES = True # Canny 边缘图辅助寻找线状缺陷

# 阈值与 Blob 过滤参数
STD_FACTOR = 1.35             # 阈值 = ROI 均值 +/- max(MIN_CONTRAST_DELTA, 标准差*STD_FACTOR)
MIN_CONTRAST_DELTA = 18       # 最小亮度差，越小越敏感，越大越保守
MIN_PIXELS = 8                # 异常区域最少像素数
MIN_AREA = 8                  # 异常区域最小外接矩形面积
BLOB_MERGE_MARGIN = 4         # 合并相邻异常区域
MAX_DEFECTS = 20              # 每帧最多上报缺陷数，避免蓝牙数据过大
IOU_DUPLICATE_THRESHOLD = 0.45

# Canny 边缘参数
CANNY_LOW = 50
CANNY_HIGH = 90
EDGE_BINARY_THRESHOLD = 180
EDGE_MIN_PIXELS = 12

# 缺陷分类参数
EDGE_MARGIN = 10
DUST_MAX_PIXELS = 45
DUST_MAX_ASPECT_RATIO = 2.0
SCRATCH_MIN_ASPECT_RATIO = 5.0
SCRATCH_MIN_LENGTH = 24
STAIN_MIN_PIXELS = 60
STAIN_MAX_ASPECT_RATIO = 4.5
SERIOUS_PIXELS = 900
MEDIUM_PIXELS = 220

# 串口参数。OpenMV 常用 UART(3)，实际引脚请按你的板子型号确认。
# 也可把这个 UART 预留给单片机：OpenMV TX -> 单片机 RX，GND 共地。
UART_PORT = 3
UART_BAUDRATE = 115200
UART_TIMEOUT_CHAR = 1000
SEND_INTERVAL_MS = 300
PRINT_JSON_TO_IDE = True

ENABLE_USB_IMAGE_STREAM = True
USB_IMAGE_INTERVAL_MS = 700
USB_IMAGE_JPEG_QUALITY = 70

# 调试绘图颜色。灰度图下 color 是 0~255。
COLOR_MAP = {
    "scratch": 255,
    "dust": 210,
    "stain": 170,
}

DEFECT_TYPES = [
    "scratch",
    "dust",
    "stain",
]

LEVEL_SCORE = {
    "normal": 0,
    "light": 1,
    "medium": 2,
    "serious": 3,
}


# ============================================================
# 2. 工具函数
# ============================================================

def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def safe_round(value, digits):
    scale = 1
    for _ in range(digits):
        scale *= 10
    return int(value * scale + 0.5) / scale


def blob_pixels(blob):
    # blob.pixels() 是异常像素数；部分固件可用 blob.area() 返回矩形面积。
    try:
        return blob.pixels()
    except Exception:
        return blob.w() * blob.h()


def blob_rect_area(blob):
    return blob.w() * blob.h()


def blob_density(blob):
    rect_area = blob_rect_area(blob)
    if rect_area <= 0:
        return 0.0
    return float(blob_pixels(blob)) / float(rect_area)


def is_near_roi_edge(x, y, w, h, roi):
    rx, ry, rw, rh = roi
    return (
        x <= rx + EDGE_MARGIN or
        y <= ry + EDGE_MARGIN or
        x + w >= rx + rw - EDGE_MARGIN or
        y + h >= ry + rh - EDGE_MARGIN
    )


def get_roi_brightness_delta(img, defect_rect, lens_roi):
    # 计算缺陷区域与整个镜片 ROI 的平均亮度差，辅助判断污点。
    try:
        defect_stats = img.get_statistics(roi=defect_rect)
        lens_stats = img.get_statistics(roi=lens_roi)
        return abs(defect_stats.mean() - lens_stats.mean())
    except Exception:
        return 0


def estimate_level(defect_type, area, length, brightness_delta):
    # 严重程度是演示用规则，可按真实样本继续调参。
    if defect_type == "scratch":
        if length >= 70 or area >= MEDIUM_PIXELS:
            return "medium"
        return "light"

    if defect_type == "dust":
        return "light"

    if defect_type == "stain":
        if area >= MEDIUM_PIXELS:
            return "medium"
        return "light"

    if area >= SERIOUS_PIXELS:
        return "serious"
    if area >= MEDIUM_PIXELS:
        return "medium"
    if area > 0:
        return "light"
    return "normal"


def rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter_w = x2 - x1
    inter_h = y2 - y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def add_blob_if_unique(blob_list, blob):
    rect = blob.rect()
    for old_blob in blob_list:
        if rect_iou(rect, old_blob.rect()) >= IOU_DUPLICATE_THRESHOLD:
            # 保留异常像素数更多的区域。
            if blob_pixels(blob) > blob_pixels(old_blob):
                blob_list.remove(old_blob)
                blob_list.append(blob)
            return
    blob_list.append(blob)


def sort_blobs_by_size(blobs):
    # OpenMV MicroPython 支持 list.sort(key=...) 的固件较新；
    # 这里用简单选择排序，兼容性更好。
    result = []
    temp = list(blobs)
    while temp:
        max_index = 0
        max_pixels = blob_pixels(temp[0])
        for i in range(1, len(temp)):
            pixels = blob_pixels(temp[i])
            if pixels > max_pixels:
                max_pixels = pixels
                max_index = i
        result.append(temp.pop(max_index))
    return result


# ============================================================
# 3. 图像预处理与异常查找
# ============================================================

def preprocess_image(img):
    # 在原图上处理，减少内存占用。若光照变化大，可以先关闭 HISTEQ 再重新调阈值。
    if ENABLE_HISTEQ:
        try:
            img.histeq(adaptive=True, clip_limit=3)
        except Exception:
            try:
                img.histeq()
            except Exception:
                pass

    if ENABLE_DENOISE:
        try:
            img.median(1)
        except Exception:
            pass

    return img


def build_anomaly_thresholds(img, roi):
    # 根据镜片 ROI 的平均亮度和标准差自动生成明暗异常阈值。
    stats = img.get_statistics(roi=roi)
    mean = stats.mean()
    stdev = stats.stdev()
    delta = int(max(MIN_CONTRAST_DELTA, stdev * STD_FACTOR))

    dark_high = clamp(int(mean - delta), 0, 255)
    bright_low = clamp(int(mean + delta), 0, 255)

    thresholds = []
    if dark_high > 3:
        thresholds.append((0, dark_high))
    if bright_low < 252:
        thresholds.append((bright_low, 255))
    return thresholds


def find_intensity_blobs(img, roi):
    thresholds = build_anomaly_thresholds(img, roi)
    if not thresholds:
        return []
    return img.find_blobs(
        thresholds,
        roi=roi,
        pixels_threshold=MIN_PIXELS,
        area_threshold=MIN_AREA,
        merge=True,
        margin=BLOB_MERGE_MARGIN
    )


def find_edge_blobs(img, roi):
    # 边缘检测主要用于补充划痕这类线状缺陷。
    if not ENABLE_EDGE_CANDIDATES:
        return []

    try:
        edge_img = img.copy()
        edge_img.find_edges(image.EDGE_CANNY, threshold=(CANNY_LOW, CANNY_HIGH))
        return edge_img.find_blobs(
            [(EDGE_BINARY_THRESHOLD, 255)],
            roi=roi,
            pixels_threshold=EDGE_MIN_PIXELS,
            area_threshold=MIN_AREA,
            merge=True,
            margin=BLOB_MERGE_MARGIN
        )
    except Exception:
        return []


def find_defect_candidates(img, roi):
    candidates = []

    for blob in find_intensity_blobs(img, roi):
        add_blob_if_unique(candidates, blob)

    for blob in find_edge_blobs(img, roi):
        add_blob_if_unique(candidates, blob)

    candidates = sort_blobs_by_size(candidates)
    if len(candidates) > MAX_DEFECTS:
        candidates = candidates[:MAX_DEFECTS]
    return candidates


# ============================================================
# 4. 缺陷分类
# ============================================================

def classify_defect(blob, roi, img):
    x = blob.x()
    y = blob.y()
    w = blob.w()
    h = blob.h()
    area = blob_pixels(blob)
    length = max(w, h)
    short_side = max(1, min(w, h))
    aspect_ratio = float(length) / float(short_side)
    density = blob_density(blob)
    brightness_delta = get_roi_brightness_delta(img, (x, y, w, h), roi)

    defect_type = None
    confidence = 0.0

    # 只保留三类：划痕、灰尘颗粒、污点。
    if aspect_ratio >= SCRATCH_MIN_ASPECT_RATIO and length >= SCRATCH_MIN_LENGTH:
        defect_type = "scratch"
        confidence = 0.76 + min(0.16, (aspect_ratio - SCRATCH_MIN_ASPECT_RATIO) / 25.0)

    elif area <= DUST_MAX_PIXELS and aspect_ratio <= DUST_MAX_ASPECT_RATIO:
        defect_type = "dust"
        confidence = 0.70 + min(0.18, (DUST_MAX_PIXELS - area) / float(DUST_MAX_PIXELS))

    elif area >= STAIN_MIN_PIXELS and aspect_ratio <= STAIN_MAX_ASPECT_RATIO:
        # density 较低时多为不规则斑块；较高时可能是较实的污点。
        defect_type = "stain"
        confidence = 0.66 + min(0.18, abs(0.55 - density) + brightness_delta / 255.0)

    if defect_type is None:
        return None

    confidence = clamp(confidence, 0.50, 0.98)
    level = estimate_level(defect_type, area, length, brightness_delta)

    return {
        "type": defect_type,
        "confidence": safe_round(confidence, 2),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "length": length,
        "aspect_ratio": safe_round(aspect_ratio, 2),
        "level": level,
    }


def build_result(defects):
    summary = {}
    for defect_type in DEFECT_TYPES:
        summary[defect_type] = 0

    overall_level = "normal"
    for defect in defects:
        defect_type = defect["type"]
        if defect_type in summary:
            summary[defect_type] += 1

        if LEVEL_SCORE[defect["level"]] > LEVEL_SCORE[overall_level]:
            overall_level = defect["level"]

    return {
        "has_defect": len(defects) > 0,
        "defect_count": len(defects),
        "summary": summary,
        "overall_level": overall_level,
        "defects": defects,
        "timestamp": time.ticks_ms(),
    }


def draw_debug(img, roi, defects):
    img.draw_rectangle(roi, color=120)
    for defect in defects:
        defect_type = defect["type"]
        color = COLOR_MAP.get(defect_type, 255)
        rect = (defect["x"], defect["y"], defect["w"], defect["h"])
        img.draw_rectangle(rect, color=color)
        label = "%s %s" % (defect_type, defect["level"])
        img.draw_string(defect["x"], max(0, defect["y"] - 10), label, color=color)


def send_json(uart, result):
    line = ujson.dumps(result)
    uart.write(line + "\n")
    try:
        usb.write((line + "\n").encode("utf-8"))
    except Exception:
        pass
    if PRINT_JSON_TO_IDE:
        print(line)


def send_usb_image(img):
    if not ENABLE_USB_IMAGE_STREAM:
        return

    try:
        jpg = img.compress(quality=USB_IMAGE_JPEG_QUALITY)
        usb.write(("IMG_BEGIN %d %d %d\n" % (jpg.size(), img.width(), img.height())).encode("utf-8"))
        usb.write(jpg)
        usb.write(b"IMG_END\n")
    except Exception as exc:
        if PRINT_JSON_TO_IDE:
            print("USB image stream failed:", exc)


# ============================================================
# 5. 主程序
# ============================================================

sensor.reset()
sensor.set_pixformat(PIXFORMAT)
sensor.set_framesize(FRAMESIZE)
sensor.skip_frames(time=FRAME_SKIP_TIME_MS)
sensor.set_auto_gain(True)
sensor.set_auto_whitebal(False)

usb = USB_VCP()
uart = UART(UART_PORT, UART_BAUDRATE, timeout_char=UART_TIMEOUT_CHAR)
clock = time.clock()
last_send_ms = time.ticks_ms()
last_image_ms = time.ticks_ms()

while True:
    clock.tick()

    img = sensor.snapshot()
    preprocess_image(img)

    blobs = find_defect_candidates(img, LENS_ROI)
    defects = []
    for blob in blobs:
        defect = classify_defect(blob, LENS_ROI, img)
        if defect is not None:
            defects.append(defect)

    result = build_result(defects)
    draw_debug(img, LENS_ROI, defects)

    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
        send_json(uart, result)
        last_send_ms = now_ms
    if time.ticks_diff(now_ms, last_image_ms) >= USB_IMAGE_INTERVAL_MS:
        send_usb_image(img)
        last_image_ms = now_ms
