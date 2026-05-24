import argparse
import csv
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

try:
    import serial
except ImportError as exc:
    raise SystemExit("pyserial is required. Use .venv_windows_host\\Scripts\\python.exe") from exc

try:
    from PIL import Image, ImageStat
except ImportError:
    Image = None
    ImageStat = None


READY_TEXT = "READY OpenMV_N6_USB_CAPTURE"
METADATA_COLUMNS = [
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
]


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_filename_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def log(message):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), message), flush=True)


def read_available(ser, timeout):
    deadline = time.time() + timeout
    data = bytearray()
    while time.time() < deadline:
        waiting = ser.in_waiting
        if waiting:
            data.extend(ser.read(waiting))
        else:
            time.sleep(0.02)
    return bytes(data)


def wait_for_text(ser, expected, timeout, stop_markers=()):
    deadline = time.time() + timeout
    data = bytearray()
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        data.extend(chunk)
        text = data.decode("utf-8", errors="ignore")
        if expected in text:
            return text
        for marker in stop_markers:
            if marker in text:
                raise RuntimeError(text[-1500:])
    text = data.decode("utf-8", errors="ignore")
    raise TimeoutError("Timed out waiting for %r. Last output:\n%s" % (expected, text[-1500:]))


def read_line(ser, timeout):
    deadline = time.time() + timeout
    data = bytearray()
    while time.time() < deadline:
        ch = ser.read(1)
        if not ch:
            continue
        data.extend(ch)
        if ch == b"\n":
            return data.decode("utf-8", errors="ignore").strip()
    return ""


def enter_raw_repl(ser):
    ser.write(b"\x03\x03")
    ser.flush()
    time.sleep(0.3)
    read_available(ser, 0.2)

    ser.write(b"\x01")
    ser.flush()
    try:
        return wait_for_text(ser, ">", timeout=3)
    except TimeoutError:
        ser.write(b"\x03\x01")
        ser.flush()
        return wait_for_text(ser, ">", timeout=3)


def start_capture_script(ser, script_text):
    enter_raw_repl(ser)
    source = script_text.encode("utf-8")
    for offset in range(0, len(source), 256):
        ser.write(source[offset:offset + 256])
        ser.flush()
        time.sleep(0.01)
    ser.write(b"\x04")
    ser.flush()
    return wait_for_text(
        ser,
        READY_TEXT,
        timeout=20,
        stop_markers=("Traceback", "SyntaxError", "ImportError", "NameError"),
    )


def read_exact(ser, size, timeout):
    deadline = time.time() + timeout
    data = bytearray()
    while len(data) < size and time.time() < deadline:
        chunk = ser.read(size - len(data))
        if chunk:
            data.extend(chunk)
    if len(data) != size:
        raise TimeoutError("Image data incomplete: %d / %d bytes" % (len(data), size))
    return bytes(data)


def capture_one(ser, timeout):
    read_available(ser, 0.05)
    ser.write(b"CAPTURE\n")
    ser.flush()

    header = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = read_line(ser, timeout=0.5)
        if not line:
            continue
        if line.startswith("ERR "):
            raise RuntimeError(line)
        if line.startswith("IMG_BEGIN "):
            header = line
            break
    if not header:
        raise TimeoutError("Timed out waiting for IMG_BEGIN")

    parts = header.split()
    if len(parts) < 2:
        raise RuntimeError("Bad image header: %s" % header)
    size = int(parts[1])
    width = int(parts[2]) if len(parts) >= 4 else 0
    height = int(parts[3]) if len(parts) >= 4 else 0
    image_bytes = read_exact(ser, size, timeout=timeout)
    end_line = read_line(ser, timeout=1)
    if end_line and end_line != "IMG_END":
        log("Warning: unexpected frame end line: %s" % end_line)
    if not image_bytes.startswith(b"\xff\xd8"):
        raise RuntimeError("Captured bytes do not look like a JPEG")
    return image_bytes, width, height


def inspect_image(image_bytes, fallback_width, fallback_height):
    result = {
        "width": fallback_width,
        "height": fallback_height,
        "brightness": "",
        "contrast": "",
        "status": "unchecked",
        "warnings": [],
    }
    if Image is None or ImageStat is None:
        return result
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            result["width"], result["height"] = img.size
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            brightness = float(stat.mean[0])
            contrast = float(stat.stddev[0])
            result["brightness"] = "%.1f" % brightness
            result["contrast"] = "%.1f" % contrast
            if brightness < 35:
                result["warnings"].append("too dark")
            elif brightness > 220:
                result["warnings"].append("too bright")
            if contrast < 12:
                result["warnings"].append("low contrast")
            if min(img.size) < 128:
                result["warnings"].append("small image")
    except Exception as exc:
        result["warnings"].append("inspection failed: %s" % exc)
    result["status"] = "review" if result["warnings"] else "ok"
    return result


def append_metadata(dataset_dir, save_path, split, label, port, baudrate, image_bytes, quality):
    metadata_path = dataset_dir / "metadata.csv"
    is_new_file = not metadata_path.exists()
    relative_path = save_path.relative_to(dataset_dir)
    with metadata_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(METADATA_COLUMNS)
        writer.writerow([
            now_text(),
            str(relative_path),
            split,
            label,
            quality["width"],
            quality["height"],
            len(image_bytes),
            quality["brightness"],
            quality["contrast"],
            quality["status"],
            ";".join(quality["warnings"]),
            port,
            baudrate,
            "auto_capture_openmv_dataset.py",
        ])


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenMV USB capture script and save dataset images.")
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--script", default="openmv/n6_usb_image_capture.py")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--label", default="scratch")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--skip-upload", action="store_true", help="Assume the capture script is already running.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    script_path = (root / args.script).resolve()
    dataset_dir = (root / args.dataset).resolve()
    save_dir = dataset_dir / args.split / args.label
    save_dir.mkdir(parents=True, exist_ok=True)

    if not script_path.exists() and not args.skip_upload:
        raise SystemExit("Missing OpenMV script: %s" % script_path)

    with serial.Serial(args.port, args.baudrate, timeout=0.1, write_timeout=3) as ser:
        time.sleep(0.3)
        if args.skip_upload:
            ser.write(b"PING\n")
            ser.flush()
            wait_for_text(ser, "PONG", timeout=3)
        else:
            script_text = script_path.read_text(encoding="utf-8")
            log("Uploading and running %s on %s..." % (script_path.name, args.port))
            start_capture_script(ser, script_text)
        log("Capture script is ready.")

        saved = []
        for index in range(1, args.count + 1):
            image_bytes, width, height = capture_one(ser, timeout=args.timeout)
            quality = inspect_image(image_bytes, width, height)
            filename = "%s_%s_%s_%03d.jpg" % (args.label, args.split, safe_filename_time(), index)
            save_path = save_dir / filename
            save_path.write_bytes(image_bytes)
            append_metadata(dataset_dir, save_path, args.split, args.label, args.port, args.baudrate, image_bytes, quality)
            saved.append(save_path)
            log(
                "Saved %02d/%02d: %s (%dx%d, %d bytes, brightness=%s, contrast=%s, %s)"
                % (
                    index,
                    args.count,
                    save_path,
                    quality["width"],
                    quality["height"],
                    len(image_bytes),
                    quality["brightness"],
                    quality["contrast"],
                    quality["status"],
                )
            )
            time.sleep(args.delay)

    log("Done. Saved %d image(s) to %s" % (len(saved), save_dir))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
