# OpenMV N6 focus preview.
#
# Use this only in OpenMV IDE before collecting data:
# 1. Connect the N6.
# 2. Run this script.
# 3. Watch the IDE framebuffer.
# 4. Rotate the camera lens until the real lens edge/scratches are sharp.

import csi
import time


PIXFORMAT = csi.RGB565
PREFERRED_FRAMESIZE = csi.XGA
FALLBACK_FRAMESIZE = csi.VGA
STARTUP_STABLE_MS = 1500


csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(PIXFORMAT)

try:
    csi0.framesize(PREFERRED_FRAMESIZE)
    print("Focus preview framesize: XGA")
except Exception as exc:
    print("XGA failed, fallback to VGA:", exc)
    csi0.framesize(FALLBACK_FRAMESIZE)

csi0.snapshot(time=STARTUP_STABLE_MS)

clock = time.clock()

while True:
    clock.tick()
    img = csi0.snapshot()
    print("focus preview fps:", clock.fps())
