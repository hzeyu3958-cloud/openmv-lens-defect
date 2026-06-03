import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_CLASSES = [
    "normal",
    "scratch",
    "stain",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train lens defect image classifier and export TFLite.")
    parser.add_argument("--dataset", required=True, help="Dataset folder with train/val/test subfolders.")
    parser.add_argument("--output", required=True, help="Output folder for .tflite and labels.txt.")
    parser.add_argument("--artifact-prefix", default="lens_defect", help="Artifact filename prefix, for example lens_defect or slide_defect.")
    parser.add_argument("--summary-name", default="training_summary.json", help="Training summary filename written under the output folder.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def image_files(folder):
    if not folder.exists():
        return []
    return [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def count_images(split_dir, labels):
    counts = {}
    for label in labels:
        counts[label] = len(image_files(split_dir / label))
    return counts


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_unique_hashes(split_dir, labels):
    counts = {}
    for label in labels:
        hashes = {file_sha256(path) for path in image_files(split_dir / label)}
        counts[label] = len(hashes)
    return counts


def build_class_weight(train_counts, labels):
    non_zero_counts = [train_counts[label] for label in labels if train_counts[label] > 0]
    if not non_zero_counts:
        return None
    max_count = max(non_zero_counts)
    return {
        index: float(max_count) / float(max(1, train_counts[label]))
        for index, label in enumerate(labels)
    }


def split_has_images(split_dir, labels):
    if not split_dir.exists():
        return False
    return sum(count_images(split_dir, labels).values()) > 0


def ensure_class_dirs(split_dir, labels):
    split_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (split_dir / label).mkdir(parents=True, exist_ok=True)


def check_dataset(dataset_dir):
    train_dir = dataset_dir / "train"
    if not train_dir.exists():
        raise RuntimeError("Missing dataset/train folder: %s" % train_dir)

    class_dirs = [
        path.name
        for path in train_dir.iterdir()
        if path.is_dir() and path.name in DEFAULT_CLASSES and image_files(path)
    ]
    if not class_dirs:
        raise RuntimeError("No class folders found under %s" % train_dir)

    return sorted(class_dirs, key=lambda name: DEFAULT_CLASSES.index(name) if name in DEFAULT_CLASSES else 999)


def write_history_csv(history, path):
    metric_names = sorted(history.history.keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + metric_names)
        epoch_count = len(next(iter(history.history.values()))) if history.history else 0
        for index in range(epoch_count):
            writer.writerow([index + 1] + [history.history[name][index] for name in metric_names])


def evaluate_split(model, dataset, split_name):
    if dataset is None:
        return None
    values = model.evaluate(dataset, verbose=0, return_dict=True)
    return {"split": split_name, **{name: float(value) for name, value in values.items()}}


def save_confusion_matrix(model, dataset, labels, path):
    if dataset is None:
        return None

    import numpy as np

    matrix = [[0 for _ in labels] for _ in labels]
    for images, batch_labels in dataset:
        predictions = model.predict(images, verbose=0)
        true_indices = np.argmax(batch_labels.numpy(), axis=1)
        pred_indices = np.argmax(predictions, axis=1)
        for true_index, pred_index in zip(true_indices, pred_indices):
            matrix[int(true_index)][int(pred_index)] += 1

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row)
    return str(path)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = str(args.artifact_prefix or "lens_defect").strip() or "lens_defect"
    summary_name = str(args.summary_name or "training_summary.json").strip() or "training_summary.json"
    classifier_stem = "%s_classifier" % artifact_prefix
    history_path = output_dir / ("%s_training_history.csv" % artifact_prefix)

    labels = check_dataset(dataset_dir)

    # TensorFlow 很大，放到 main 里导入。没有安装时，语法检查仍能通过。
    import tensorflow as tf

    image_size = (args.image_size, args.image_size)
    class_counts = {
        "train": count_images(dataset_dir / "train", labels),
        "val": count_images(dataset_dir / "val", labels),
        "test": count_images(dataset_dir / "test", labels),
    }
    unique_counts = {
        "train": count_unique_hashes(dataset_dir / "train", labels),
        "val": count_unique_hashes(dataset_dir / "val", labels),
        "test": count_unique_hashes(dataset_dir / "test", labels),
    }
    if sum(class_counts["train"].values()) == 0:
        raise RuntimeError("No training images found under %s" % (dataset_dir / "train"))
    print("Class counts:", json.dumps(class_counts, ensure_ascii=False))
    print("Unique image hashes:", json.dumps(unique_counts, ensure_ascii=False))
    class_weight = build_class_weight(class_counts["train"], labels)
    print("Class weights:", json.dumps(class_weight, ensure_ascii=False))

    ensure_class_dirs(dataset_dir / "train", labels)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        labels="inferred",
        label_mode="categorical",
        class_names=labels,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_dir = dataset_dir / "val"
    val_ds = None
    if split_has_images(val_dir, labels):
        ensure_class_dirs(val_dir, labels)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=labels,
            image_size=image_size,
            batch_size=args.batch_size,
            shuffle=False,
        )

    test_dir = dataset_dir / "test"
    test_ds = None
    if split_has_images(test_dir, labels):
        ensure_class_dirs(test_dir, labels)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=labels,
            image_size=image_size,
            batch_size=args.batch_size,
            shuffle=False,
        )

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.06),
        tf.keras.layers.RandomContrast(0.12),
    ], name="train_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(args.image_size, args.image_size, 3)),
        tf.keras.layers.Rescaling(1.0 / 255.0),
        augment,
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(16, 16)),
        tf.keras.layers.Conv2D(len(labels), 1, padding="valid", activation=None),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Softmax(),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    monitor_accuracy = "val_accuracy" if val_ds is not None else "accuracy"
    monitor_loss = "val_loss" if val_ds is not None else "loss"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / ("%s.keras" % classifier_stem)),
            monitor=monitor_accuracy,
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_accuracy,
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_loss, patience=3, factor=0.5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    write_history_csv(history, history_path)

    keras_path = output_dir / ("%s.keras" % classifier_stem)
    if keras_path.exists():
        model = tf.keras.models.load_model(keras_path)

    metrics = {
        "train": evaluate_split(model, train_ds, "train"),
        "val": evaluate_split(model, val_ds, "val"),
        "test": evaluate_split(model, test_ds, "test"),
    }
    confusion_source = test_ds if test_ds is not None else val_ds
    confusion_split = "test" if test_ds is not None else "val"
    confusion_path = None
    if confusion_source is not None:
        confusion_path = save_confusion_matrix(
            model,
            confusion_source,
            labels,
            output_dir / ("%s_confusion_matrix_%s.csv" % (artifact_prefix, confusion_split)),
        )

    labels_path = output_dir / ("%s_labels.txt" % artifact_prefix)
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite = float_converter.convert()
    float_path = output_dir / ("%s_float.tflite" % classifier_stem)
    float_path.write_bytes(float_tflite)

    def representative_dataset():
        for images, _labels in train_ds.unbatch().batch(1).take(100):
            yield [tf.cast(images, tf.float32)]

    int8_path = output_dir / ("%s_int8.tflite" % classifier_stem)
    try:
        int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        int8_converter.representative_dataset = representative_dataset
        int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        int8_converter.inference_input_type = tf.uint8
        int8_converter.inference_output_type = tf.uint8
        int8_tflite = int8_converter.convert()
        int8_path.write_bytes(int8_tflite)
        print("Saved INT8 TFLite:", int8_path)
    except Exception as exc:
        print("INT8 export failed, float model is still available:", exc)

    print("Saved float TFLite:", float_path)
    print("Saved labels:", labels_path)
    print("Class order:", labels)

    summary = {
        "dataset": str(dataset_dir),
        "output": str(output_dir),
        "artifact_prefix": artifact_prefix,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "labels": labels,
        "class_counts": class_counts,
        "unique_hash_counts": unique_counts,
        "class_weight": class_weight,
        "metrics": metrics,
        "artifacts": {
            "keras": str(keras_path),
            "float_tflite": str(float_path),
            "int8_tflite": str(int8_path) if int8_path.exists() else "",
            "labels": str(labels_path),
            "training_history": str(history_path),
            "confusion_matrix": confusion_path or "",
        },
    }
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved training summary:", summary_path)


if __name__ == "__main__":
    main()
