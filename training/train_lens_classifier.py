import argparse
from pathlib import Path


DEFAULT_CLASSES = [
    "normal",
    "scratch",
    "dust",
    "stain",
    "coating_damage",
    "crack",
    "edge_damage",
    "unknown",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train lens defect image classifier and export TFLite.")
    parser.add_argument("--dataset", required=True, help="Dataset folder with train/val/test subfolders.")
    parser.add_argument("--output", required=True, help="Output folder for .tflite and labels.txt.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def check_dataset(dataset_dir):
    train_dir = dataset_dir / "train"
    if not train_dir.exists():
        raise RuntimeError("Missing dataset/train folder: %s" % train_dir)

    class_dirs = [path.name for path in train_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise RuntimeError("No class folders found under %s" % train_dir)

    return sorted(class_dirs, key=lambda name: DEFAULT_CLASSES.index(name) if name in DEFAULT_CLASSES else 999)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = check_dataset(dataset_dir)

    # TensorFlow 很大，放到 main 里导入。没有安装时，语法检查仍能通过。
    import tensorflow as tf

    image_size = (args.image_size, args.image_size)

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
    if val_dir.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=labels,
            image_size=image_size,
            batch_size=args.batch_size,
            shuffle=False,
        )

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.15),
    ])

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
        tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "lens_defect_classifier.keras"),
            monitor="val_accuracy" if val_ds is not None else "accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    keras_path = output_dir / "lens_defect_classifier.keras"
    if keras_path.exists():
        model = tf.keras.models.load_model(keras_path)

    labels_path = output_dir / "lens_defect_labels.txt"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite = float_converter.convert()
    float_path = output_dir / "lens_defect_classifier_float.tflite"
    float_path.write_bytes(float_tflite)

    def representative_dataset():
        for images, _labels in train_ds.unbatch().batch(1).take(100):
            yield [tf.cast(images, tf.float32)]

    int8_path = output_dir / "lens_defect_classifier_int8.tflite"
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


if __name__ == "__main__":
    main()
