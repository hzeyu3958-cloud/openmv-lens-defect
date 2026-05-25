import argparse
import collections
import json
import sys
from pathlib import Path

from PIL import Image


CLASSES = ("normal", "scratch", "stain")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the host PC fast-review detector on the local dataset.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--examples", type=int, default=40)
    parser.add_argument("--examples-per-pair", type=int, default=12)
    parser.add_argument("--json-output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "windows_host"))
    import lens_defect_host as host

    dataset_dir = Path(args.dataset)
    app = host.LensDefectHostApp.__new__(host.LensDefectHostApp)
    app.latest_detection_result = None

    summary = collections.defaultdict(collections.Counter)
    examples = []
    pair_examples = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        for truth in CLASSES:
            folder = dataset_dir / split / truth
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in host.IMAGE_EXTENSIONS:
                    continue
                with Image.open(path) as raw_image:
                    image = raw_image.convert("RGB")

                base = {
                    "frame": {"w": image.width, "h": image.height},
                    "roi": {"x": 0, "y": 0, "w": image.width, "h": image.height},
                }
                app.latest_detection_result = base
                mask = host.LensDefectHostApp.build_detection_mask(
                    app,
                    image,
                    {"width": image.width, "height": image.height},
                    base,
                )
                detection = host.LensDefectHostApp.fast_cv_detect_defect(app, image, mask)
                if detection is not None:
                    result = dict(base)
                    result["defects"] = [detection]
                    result = host.LensDefectHostApp.normalize_detection_result(app, result)
                    result = host.LensDefectHostApp.filter_edge_defects_for_image(
                        app,
                        result,
                        {"width": image.width, "height": image.height},
                        image,
                    )
                    defects = result.get("defects") or []
                    detection = defects[0] if defects else None

                predicted = detection.get("type") if detection else "normal"
                summary[truth][predicted] += 1
                if predicted != truth:
                    example = {
                        "split": split,
                        "truth": truth,
                        "predicted": predicted,
                        "file": str(path),
                        "detection": detection,
                    }
                    if len(examples) < args.examples:
                        examples.append(example)
                    pair_key = "%s->%s" % (truth, predicted)
                    if len(pair_examples[pair_key]) < args.examples_per_pair:
                        pair_examples[pair_key].append(example)

    report = {
        "summary": {truth: dict(summary[truth]) for truth in CLASSES},
        "examples": examples,
        "pair_examples": dict(pair_examples),
    }

    for truth in CLASSES:
        counts = report["summary"][truth]
        total = sum(counts.values())
        correct = counts.get(truth, 0)
        accuracy = correct / float(total or 1)
        print("%s: %s accuracy=%.3f" % (truth, counts, accuracy))

    if examples:
        print("examples:")
        for item in examples:
            detection = item.get("detection") or {}
            compact = {
                "truth": item["truth"],
                "predicted": item["predicted"],
                "file": item["file"],
                "type": detection.get("type"),
                "source": detection.get("source"),
                "confidence": detection.get("confidence"),
                "box": [detection.get("x"), detection.get("y"), detection.get("w"), detection.get("h")],
                "area": detection.get("area"),
                "length": detection.get("length"),
                "aspect_ratio": detection.get("aspect_ratio"),
            }
            print(json.dumps(compact, ensure_ascii=False))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
