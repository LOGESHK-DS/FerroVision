"""Train FerroVision casting-defect classifiers"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Setting up the project directories and image training parameters
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "casting_data"
OUT = ROOT / "models" / "vision"
METRICS = ROOT / "outputs" / "metrics"

IMAGE_SIZE = (160, 160)
BATCH = 32
SEED = 42

def image_files(split):
    # Collecting image paths and their corresponding class labels
    paths, labels = [], []

    # Assigning 0 to normal images and 1 to defective images
    for label, name in enumerate(("normal", "defective")):
        files = sorted(path for path in (DATA / split / name).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        paths.extend(map(str, files))
        labels.extend([label] * len(files))

    return paths, labels

def make_dataset(paths, labels, training=False):

    # Creating a TensorFlow dataset from image paths and labels
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffling the training images to avoid learning from a fixed order
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        
    def load(path, label):

        # Reading the image and converting it into RGB format
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)

        # Resizing all images to the same dimensions
        image = tf.image.resize(image, IMAGE_SIZE)

        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)
    
    # Loading images in parallel and preparing batches for training
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

def datasets():

    # Loading image paths and labels from the training folder
    paths, labels = image_files("train")

    # Splitting training data into train and validation sets
    # Stratification maintains the same class distribution in both sets
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(paths, labels, test_size=.2, stratify=labels, random_state=SEED)

    # Keeping the original test dataset separate for final evaluation
    test_paths, test_labels = image_files("test")

    return make_dataset(train_paths, train_labels, True), make_dataset(valid_paths, valid_labels), make_dataset(test_paths, test_labels), ["normal", "defective"]

def custom_cnn():

    # Applying data augmentation to improve model generalization
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(.08),
        tf.keras.layers.RandomZoom(.1)
    ])
    
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))

    # Applying augmentation and scaling pixel values to the 0-1 range
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1/255)(x)

    # Building convolution and pooling blocks to extract image features
    for filters in (32, 64, 128):
        x = tf.keras.layers.MaxPool2D()(
            tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x))

    # Reducing feature dimensions and applying dropout to reduce overfitting
    x = tf.keras.layers.Dropout(.35)(
        tf.keras.layers.GlobalAveragePooling2D()(x))

    # Sigmoid gives the probability of the image belonging to the defective class
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="custom_casting_cnn")


def transfer(name):

    backbone_cls = {
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "efficientnetb0": tf.keras.applications.EfficientNetB0
    }[name]

    # Load pretrained ImageNet backbone
    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Stage 1: Freeze entire backbone
    base.trainable = False

    inputs = tf.keras.Input((*IMAGE_SIZE, 3))

    x = tf.keras.layers.RandomFlip("horizontal")(inputs)

    # Backbone-specific preprocessing
    if name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    else:
        x = tf.keras.applications.efficientnet.preprocess_input(x)

    # Feature extraction
    x = base(x, training=False)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs, outputs,
        name=f"{name}_casting"
    )

    return model, base


def main():

    # Taking model selection and number of epochs from command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["custom", "mobilenetv2", "efficientnetb0"], default="custom")
    parser.add_argument("--epochs", type=int, default=8)

    args = parser.parse_args()

    # Creating output folders if they don't already exist
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    # Preparing training, validation and test datasets
    train, valid, test, classes = datasets()

    # Build the selected model
    if args.model == "custom":
        model = custom_cnn()
        base = None
    else:
        model, base = transfer(args.model)

    path = OUT / f"{args.model}_casting.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Common metrics
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    # Training classification head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=metrics
    )

    print("\nSTAGE 1: Feature extraction")

    model.fit(
        train,
        validation_data=valid,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Fine-tune pretrained backbone
    if base is not None:

        print("\nSTAGE 2: Fine-tuning")

        base.trainable = True

        # Freeze all but the last 30 layers
        for layer in base.layers[:-30]:
            layer.trainable = False

        # Keep Batch Normalization layers frozen
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # Recompile with a very low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),
            loss="binary_crossentropy",
            metrics=metrics
        )

        model.fit(
            train,
            validation_data=valid,
            epochs=5,
            callbacks=callbacks
        )

    # Final evaluation on untouched test data
    model.load_weights(path)

    values = model.evaluate(
        test,
        return_dict=True,
        verbose=0
    )

    # Calculating F1 score using the precision and recall values
    values["f1"] = (2*values["precision"]*values["recall"]/(values["precision"]+values["recall"]+1e-8))

    # Adding model name and class order to the evaluation results
    values["model"] = args.model
    values["class_order"] = "|".join(classes)

    # Saving model evaluation results in a CSV file
    with open(METRICS / f"{args.model}_casting_metrics.csv", "w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=values.keys())
        writer.writeheader()
        writer.writerow(values)

    # Displaying the final evaluation results    
    print(values)

if __name__ == "__main__":
    main()