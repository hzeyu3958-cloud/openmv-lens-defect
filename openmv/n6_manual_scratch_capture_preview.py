# OpenMV N6 manual scratch dataset preview.
#
# Use in OpenMV IDE when you want to save images from the framebuffer manually.
# The preview is cropped to the current lens area so saved images match the
# dataset crop used by n6_usb_image_capture.py and n6_classifier_main.py.

import csi
import image
import time


PIXFORMAT = csi.RGB565
PREFERRED_FRAMESIZE = csi.XGA
FALLBACK_FRAMESIZE = csi.VGA
STARTUP_STABLE_MS = 1500

# Same crop ratio as n6_usb_image_capture.py.
CAPTURE_ROI_RATIO = (0.25, 0.15, 0.72, 0.75)


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


csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(PIXFORMAT)

try:
    csi0.framesize(PREFERRED_FRAMESIZE)
    print("Manual scratch preview framesize: XGA")
except Exception as exc:
    print("XGA failed, fallback to VGA:", exc)
    csi0.framesize(FALLBACK_FRAMESIZE)

csi0.snapshot(time=STARTUP_STABLE_MS)
clock = time.clock()

while True:
    clock.tick()
    img = csi0.snapshot()
    roi = scaled_roi(img, CAPTURE_ROI_RATIO)
    crop = img.copy(roi=roi)
    crop.flush()
    print("fps:", clock.fps(), "roi:", roi)
