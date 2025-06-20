import os
import csv
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ------------------- Argument Configuration -------------------
parser = argparse.ArgumentParser(description="Experiment 2 - Transfer Learning with Output Layer Replacement")
parser.add_argument("save_dir", type=str, help="Directory to save model and logs")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
args = parser.parse_args()

# Dataset Path 
data_dir = "/home/jojo24tu/example/ASSIGNMENT1/PetImages"

#  Clean Broken Images 
def clean_invalid_images(root_path):
    removed = 0
    for category in ["Cat", "Dog"]:
        category_path = os.path.join(root_path, category)
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            try:
                content = tf.io.read_file(img_path)
                _ = tf.image.decode_jpeg(content, channels=3)
            except:
                os.remove(img_path)
                removed += 1
    print(f"Removed {removed} unreadable images.")

clean_invalid_images(data_dir)

#Load Image Dataset
img_shape = (180, 180)
batch_size = 64

train_data = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_shape,
    batch_size=batch_size,
)

val_data = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_shape,
    batch_size=batch_size,
)

# Apply Data Augmentation 
augment_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def augment_batch(images):
    for aug in augment_layers:
        images = aug(images)
    return images

train_data = train_data.map(
    lambda x, y: (augment_batch(x), y),
    num_parallel_calls=AUTOTUNE,
)

# Normalize and prefetch
norm = layers.Rescaling(1. / 255)
train_data = train_data.map(lambda x, y: (norm(x), y)).prefetch(AUTOTUNE)
val_data = val_data.map(lambda x, y: (norm(x), y)).prefetch(AUTOTUNE)

#  Load Pretrained Model
print("Loading base model trained on Stanford Dogs...")
base_model = keras.models.load_model("stanford_dogs_model.h5", compile=False)

# Modify Output Layer 
penultimate = base_model.layers[-2].output
binary_output = layers.Dense(1, activation="sigmoid")(penultimate)
model = keras.Model(inputs=base_model.input, outputs=binary_output)

# Freeze feature extractor layers
for layer in model.layers[:-1]:
    layer.trainable = False

# Prepare Logging Directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Define Training Callbacks 
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.save_dir, "model_exp2.h5"),
        save_best_only=True,
        monitor="val_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        mode="max"
    )
]

#  Compile and Train 
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

print("Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=args.epochs,
    callbacks=callbacks
)

#  Export Training Results 
txt_output = os.path.join(args.save_dir, "exp2_results.txt")
csv_output = os.path.join(args.save_dir, "exp2_results.csv")

with open(txt_output, "w") as txt_file, open(csv_output, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss"])

    txt_file.write("Experiment 2 – Only Output Layer Replaced\n")

    for idx in range(len(history.history["accuracy"])):
        acc = history.history["accuracy"][idx]
        val_acc = history.history["val_accuracy"][idx]
        loss = history.history["loss"][idx]
        val_loss = history.history["val_loss"][idx]

        result_str = (
            f"Epoch {idx + 1:02d} | "
            f"Train Acc: {acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Loss: {loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        print(result_str)
        txt_file.write(result_str + "\n")
        writer.writerow([idx + 1, acc, val_acc, loss, val_loss])
