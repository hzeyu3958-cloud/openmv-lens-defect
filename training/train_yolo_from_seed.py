import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model from dataset_yolo_seed and export ONNX.")
    parser.add_argument("--data", default="dataset_yolo_seed/data.yaml")
    parser.add_argument("--output", default="models/lens_yolo.onnx")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="lens_yolo")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = Path(args.data)
    output = Path(args.output)
    if not data_yaml.is_absolute():
        data_yaml = project_root / data_yaml
    if not output.is_absolute():
        output = project_root / output
    if not data_yaml.exists():
        raise SystemExit("Missing YOLO data.yaml: %s. Run export_yolo_seed_labels.py first." % data_yaml)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install it in the training environment first: "
            "%s -m pip install -r training/requirements-training.txt" % sys.executable
        ) from exc

    base_model = Path(args.base_model)
    if not base_model.is_absolute() and base_model.exists():
        base_model = base_model.resolve()
    model = YOLO(str(base_model))
    result = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.image_size),
        batch=int(args.batch),
        device="cpu",
        task=args.task,
        project=str((project_root / "outputs" / "yolo_training").resolve()),
        name=args.name,
        exist_ok=True,
    )
    best_pt = Path(result.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best_pt if best_pt.exists() else base_model))
    exported = Path(trained.export(format="onnx", imgsz=int(args.image_size), opset=12))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    labels = output.with_name(output.stem + "_labels.txt")
    if output.name in ("lens_yolo.onnx", "lens_yolo_seg.onnx"):
        labels = output.with_name("lens_yolo_labels.txt")
    labels.write_text("scratch\nstain\n", encoding="utf-8")
    metadata = {
        "input_size": int(args.image_size),
        "base_model": str(base_model),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "data": str(data_yaml),
        "task": args.task,
        "recommended_runtime": "detect uses boxes; segment exports YOLO-seg masks when polygon labels are available.",
    }
    meta = output.with_name(output.stem + "_meta.json")
    if output.name == "lens_yolo.onnx":
        meta = output.with_name("lens_yolo_meta.json")
    meta.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print("YOLO ONNX exported:", output.resolve())


if __name__ == "__main__":
    main()
