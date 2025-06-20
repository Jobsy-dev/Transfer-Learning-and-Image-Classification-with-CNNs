import os
import csv
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Argument Parsing 
parser = argparse.ArgumentParser(description="Experiment 1A: Baseline CNN Training")
parser.add_argument("log_dir", type=str, help="Directory to store checkpoints and results")
parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
args = parser.parse_args()

#  Data Path Setup 
dataset_dir = "/home/jojo24tu/example/ASSIGNMENT1/PetImages"

def clean_corrupt_images(path):
    removed = 0
    for category in ["Cat", "Dog"]:
        full_path = os.path.join(path, category)
        for img_file in os.listdir(full_path):
            try:
                img_data = tf.io.read_file(os.path.join(full_path, img_file))
                tf.image.decode_jpeg(img_data, channels=3)
            except:
                os.remove(os.path.join(full_path, img_file))
                removed += 1
    print(f"Removed {removed} corrupted images.")

clean_corrupt_images(dataset_dir)

# Dataset Load 
img_size = (180, 180)
batch = 64

train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch
)

# Only apply augmentation to training set
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

train_ds = train_ds.map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

#  Model Definition 
def build_cnn_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    prev = x
    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        res = layers.Conv2D(size, 1, strides=2, padding="same")(prev)
        x = layers.add([x, res])
        prev = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)

model = build_cnn_model(img_size + (3,))
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
os.makedirs(args.log_dir, exist_ok=True)
ckpt_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.log_dir, "cnn_epoch_{epoch:02d}.keras"),
    save_freq="epoch",
    save_best_only=False
)

# Model Training 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=[ckpt_callback]
)

#  Save Results 
results_txt = os.path.join(args.log_dir, "results.txt")
results_csv = os.path.join(args.log_dir, "results.csv")

with open(results_txt, "w") as txt_f, open(results_csv, "w", newline='') as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow(["Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss"])
    txt_f.write("Training Results:\n")

    for i in range(args.epochs):
        acc = history.history["accuracy"][i]
        val_acc = history.history["val_accuracy"][i]
        loss = history.history["loss"][i]
        val_loss = history.history["val_loss"][i]
        line = f"Epoch {i+1:02d} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}"
        print(line)
        txt_f.write(line + "\n")
        writer.writerow([i+1, acc, val_acc, loss, val_loss])
