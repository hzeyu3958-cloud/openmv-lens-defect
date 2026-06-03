# OpenMV N6 USB 载玻片采集脚本
# 用途：
# 1. 把本脚本保存到 OpenMV N6 并运行。
# 2. Windows 上位机通过 USB COM 口发送 CAPTURE。
# 3. N6 拍一张 JPEG 图片，通过 USB 串口发回电脑保存到 dataset_slide。

import csi
import image
import select
import sys
import time
import pyb


PIXFORMAT = csi.RGB565
FRAMESIZE = csi.XGA
FALLBACK_FRAMESIZE = csi.VGA
JPEG_QUALITY = 80
JPEG_SUBSAMPLING = None
USB_WRITE_CHUNK_SIZE = 1024
USB_WRITE_CHUNK_DELAY_MS = 3

# 载玻片通常是横向矩形，这里默认截取画面中部的宽矩形区域。
# 640x400 预览下大约对应 (89, 96, 460, 160)，可按夹具位置微调。
CAPTURE_ROI_RATIO = (0.14, 0.24, 0.72, 0.40)

DISABLE_AUTO_GAIN_AFTER_START = True
DISABLE_AUTO_WHITEBAL_AFTER_START = True
STARTUP_STABLE_MS = 2000


class StdIOPort:
    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif hasattr(data, "bytearray"):
            data = data.bytearray()
        sys.stdout.buffer.write(data)

    def any(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(readable)

    def readline(self):
        return sys.stdin.buffer.readline()


def open_command_port():
    try:
        return pyb.USB_VCP()
    except Exception:
        return StdIOPort()


usb = open_command_port()

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(PIXFORMAT)
try:
    csi0.framesize(FRAMESIZE)
except Exception:
    csi0.framesize(FALLBACK_FRAMESIZE)
csi0.snapshot(time=STARTUP_STABLE_MS)

if DISABLE_AUTO_GAIN_AFTER_START:
    csi0.auto_gain(False)
if DISABLE_AUTO_WHITEBAL_AFTER_START:
    csi0.auto_whitebal(False)


def send_line(text):
    usb.write((text + "\n").encode("utf-8"))


def read_command():
    if not usb.any():
        return None
    line = usb.readline()
    if not line:
        return None
    try:
        return line.decode("utf-8").strip().upper()
    except Exception:
        return None


def scaled_roi(img, ratio):
    x_ratio, y_ratio, w_ratio, h_ratio = ratio
    x = int(img.width() * x_ratio)
    y = int(img.height() * y_ratio)
    w = int(img.width() * w_ratio)
    h = int(img.height() * h_ratio)
    x = max(0, min(x, img.width() - 1))
    y = max(0, min(y, img.height() - 1))
    w = max(1, min(w, img.width() - x))
    h = max(1, min(h, img.height() - y))
    return x, y, w, h


def capture_and_send():
    img = csi0.snapshot()
    capture_roi = scaled_roi(img, CAPTURE_ROI_RATIO)

    try:
        if JPEG_SUBSAMPLING is None:
            raise TypeError("JPEG subsampling is not available")
        jpg = img.compress(roi=capture_roi, quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    except TypeError:
        jpg = img.compress(roi=capture_roi, quality=JPEG_QUALITY)
    width = capture_roi[2]
    height = capture_roi[3]

    send_line("IMG_BEGIN %d %d %d" % (jpg.size(), width, height))
    try:
        jpg_bytes = jpg.bytearray()
    except Exception:
        jpg_bytes = None
    if jpg_bytes is None:
        usb.write(jpg)
    else:
        offset = 0
        while offset < jpg.size():
            next_offset = min(jpg.size(), offset + USB_WRITE_CHUNK_SIZE)
            usb.write(jpg_bytes[offset:next_offset])
            offset = next_offset
            if USB_WRITE_CHUNK_DELAY_MS > 0:
                time.sleep_ms(USB_WRITE_CHUNK_DELAY_MS)
    send_line("IMG_END")


send_line("READY OpenMV_N6_USB_SLIDE_CAPTURE")

while True:
    command = read_command()
    if command == "CAPTURE":
        try:
            capture_and_send()
        except Exception as exc:
            send_line("ERR %s" % exc)
    elif command == "PING":
        send_line("PONG")
    time.sleep_ms(10)
