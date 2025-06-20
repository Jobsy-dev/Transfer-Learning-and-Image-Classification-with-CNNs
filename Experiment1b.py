import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import csv

# Configuration
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4

#  Model Definition 
def build_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    residual = x
    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(residual)
        x = layers.add([x, residual])
        residual = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

# Preprocessing
def preprocess_train(image, label):
    resize_larger = (int(IMAGE_SIZE[0] * 1.2), int(IMAGE_SIZE[1] * 1.2))
    image = tf.image.resize(image, resize_larger)
    image = tf.image.random_crop(image, size=(*IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_val(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Load Dataset
dataset, info = tfds.load("stanford_dogs", split="train", as_supervised=True, with_info=True)
num_total = info.splits["train"].num_examples
num_train = int(0.8 * num_total)

train_ds = dataset.take(num_train).map(preprocess_train).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.skip(num_train).map(preprocess_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_classes = info.features["label"].num_classes

# Build & Compile Model 
model = build_model(IMAGE_SIZE + (3,), num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#  Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_stanford_dogs_model.h5",  
        save_best_only=True,
        monitor="val_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max")
]

# Train 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

#  Save Training Results
with open("stanford_dogs_results.txt", "w") as txt_file, open("stanford_dogs_results.csv", "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss"])
    txt_file.write("Stanford Dogs Training Results:\n")

    for epoch in range(len(history.history["accuracy"])):
        line = (
            f"Epoch {epoch+1:2d} | "
            f"Train Acc: {history.history['accuracy'][epoch]:.4f} | "
            f"Val Acc: {history.history['val_accuracy'][epoch]:.4f} | "
            f"Train Loss: {history.history['loss'][epoch]:.4f} | "
            f"Val Loss: {history.history['val_loss'][epoch]:.4f}"
        )
        print(line)
        txt_file.write(line + "\n")
        writer.writerow([
            epoch+1,
            history.history["accuracy"][epoch],
            history.history["val_accuracy"][epoch],
            history.history["loss"][epoch],
            history.history["val_loss"][epoch]
        ])
