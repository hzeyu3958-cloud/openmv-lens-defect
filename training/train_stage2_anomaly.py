import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WINDOWS_HOST_DIR = ROOT_DIR / "windows_host"
if str(WINDOWS_HOST_DIR) not in sys.path:
    sys.path.insert(0, str(WINDOWS_HOST_DIR))

from pc_camera_rule_test import build_roi, create_lens_segmentation, parse_roi, read_image
from stage2_anomaly import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_THRESHOLD_PERCENTILE,
    extract_lens_patch,
    feature_stack,
    save_reference,
    train_reference,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight stage-2 lens anomaly model from normal images.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root. Default: dataset")
    parser.add_argument("--normal-dir", help="Normal image folder. Default: dataset/train/normal")
    parser.add_argument("--output", default="models/lens_stage2_anomaly.npz", help="Output .npz model path.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--threshold-percentile", type=float, default=DEFAULT_THRESHOLD_PERCENTILE)
    parser.add_argument("--roi", type=parse_roi, help="Optional fixed ROI x,y,w,h used while finding the lens.")
    parser.add_argument("--edge-ignore", type=int, default=0)
    parser.add_argument("--lens-mode", choices=("auto", "center"), default="auto")
    return parser.parse_args()


def image_files(folder):
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_sample(path, image_size, roi, edge_ignore, lens_mode):
    frame = read_image(path)
    if frame is None:
        return None, "read_failed"

    active_roi = build_roi(frame.shape, roi)
    segmentation = create_lens_segmentation(frame, active_roi, edge_ignore, lens_mode)
    patch, patch_mask, _bounds = extract_lens_patch(frame, segmentation["analysis_mask"], image_size)
    if patch is None:
        return None, "empty_lens_mask"
    if patch_mask.sum() == 0:
        return None, "empty_patch_mask"

    return (feature_stack(patch), patch_mask), None


def main():
    args = parse_args()
    dataset_dir = (ROOT_DIR / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    normal_dir = Path(args.normal_dir) if args.normal_dir else dataset_dir / "train" / "normal"
    if not normal_dir.is_absolute():
        normal_dir = (ROOT_DIR / normal_dir).resolve()

    paths = image_files(normal_dir)
    if not paths:
        raise RuntimeError("没有正常样本，先用上位机采集 normal 图片：%s" % normal_dir)

    samples = []
    skipped = []
    for path in paths:
        sample, reason = build_sample(path, args.image_size, args.roi, args.edge_ignore, args.lens_mode)
        if sample is None:
            skipped.append({"path": str(path), "reason": reason})
            continue
        samples.append(sample)

    if len(samples) < 5:
        raise RuntimeError("二级模型至少建议 5 张正常图，当前有效样本 %d 张。请先多采正常镜片。" % len(samples))

    reference = train_reference(
        samples,
        threshold_percentile=args.threshold_percentile,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    save_reference(output_path, reference)

    summary = {
        "output": str(output_path),
        "normal_dir": str(normal_dir),
        "input_images": len(paths),
        "used_samples": len(samples),
        "skipped": skipped,
        "image_size": int(args.image_size),
        "threshold": float(reference["threshold"]),
        "threshold_percentile": float(args.threshold_percentile),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
