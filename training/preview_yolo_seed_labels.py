import argparse
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    0: (255, 64, 64),
    1: (255, 176, 0),
}
NAMES = {
    0: "scratch",
    1: "stain",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLO seed boxes for quick visual review.")
    parser.add_argument("--dataset", default="dataset_yolo_seed")
    parser.add_argument("--output", default="outputs/yolo_seed_preview")
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--max-images", type=int, default=200)
    return parser.parse_args()


def parse_label_line(line):
    parts = line.split("#", 1)[0].strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, w, h = [float(value) for value in parts[1:5]]
    except ValueError:
        return None
    return class_id, cx, cy, w, h


def find_image(image_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / (stem + extension)
        if candidate.exists():
            return candidate
    return None


def draw_preview(image_path, label_path, output_path):
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    image_w, image_h = image.size
    lines = [
        parse_label_line(line)
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ]
    boxes = [item for item in lines if item is not None]
    if not boxes:
        return False
    width = max(2, min(image_w, image_h) // 160)
    for class_id, cx, cy, box_w, box_h in boxes:
        left = int((cx - box_w / 2.0) * image_w)
        top = int((cy - box_h / 2.0) * image_h)
        right = int((cx + box_w / 2.0) * image_w)
        bottom = int((cy + box_h / 2.0) * image_h)
        color = COLORS.get(class_id, (96, 180, 255))
        draw.rectangle(
            (
                max(0, left),
                max(0, top),
                min(image_w - 1, right),
                min(image_h - 1, bottom),
            ),
            outline=color,
            width=width,
        )
        label = NAMES.get(class_id, str(class_id))
        text_top = max(0, top - 16)
        draw.rectangle((max(0, left), text_top, max(52, left + len(label) * 8), text_top + 14), fill=color)
        draw.text((max(0, left) + 2, text_top), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return True


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    saved = 0
    scanned = 0
    for split in splits:
        label_dir = dataset / "labels" / split
        image_dir = dataset / "images" / split
        if not label_dir.exists() or not image_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            if args.max_images > 0 and saved >= args.max_images:
                break
            if label_path.stat().st_size <= 0:
                continue
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            scanned += 1
            output_path = output / split / image_path.name
            if draw_preview(image_path, label_path, output_path):
                saved += 1
        if args.max_images > 0 and saved >= args.max_images:
            break
    print("YOLO seed previews saved: %d / %d" % (saved, scanned))
    print("Output:", output.resolve())


if __name__ == "__main__":
    main()
