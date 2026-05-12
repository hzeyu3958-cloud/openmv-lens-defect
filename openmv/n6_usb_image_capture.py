# OpenMV N6 USB 采集训练图片脚本
# 用途：
# 1. 把本脚本保存到 OpenMV N6 并运行。
# 2. Windows 上位机通过 USB COM 口发送 CAPTURE。
# 3. N6 拍一张 JPEG 图片，通过 USB 串口发回电脑保存到数据集。

import csi
import image
import time
import pyb


# ============================================================
# 可调参数
# ============================================================

PIXFORMAT = csi.RGB565
FRAMESIZE = csi.QVGA
JPEG_QUALITY = 92

# 如果只想拍镜片区域，可以设置 ROI，例如 (40, 30, 240, 180)。
# None 表示保存整张图，后期训练时电脑端会自动缩放。
CAPTURE_ROI = None

DISABLE_AUTO_GAIN_AFTER_START = True
DISABLE_AUTO_WHITEBAL_AFTER_START = True
STARTUP_STABLE_MS = 2000


usb = pyb.USB_VCP()

csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(PIXFORMAT)
csi0.framesize(FRAMESIZE)
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


def capture_and_send():
    img = csi0.snapshot()

    if CAPTURE_ROI is not None:
        jpg = img.compress(roi=CAPTURE_ROI, quality=JPEG_QUALITY)
        width = CAPTURE_ROI[2]
        height = CAPTURE_ROI[3]
    else:
        jpg = img.compress(quality=JPEG_QUALITY)
        width = img.width()
        height = img.height()

    # Image.size() 返回压缩后 JPEG 字节数；Image 可直接作为字节流写入 USB。
    send_line("IMG_BEGIN %d %d %d" % (jpg.size(), width, height))
    usb.write(jpg)
    send_line("IMG_END")


send_line("READY OpenMV_N6_USB_CAPTURE")

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
