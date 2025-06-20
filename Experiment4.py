import os
import csv
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Argument Parser 
parser = argparse.ArgumentParser(description="Experiment 4: Modify final + last two conv layers")
parser.add_argument("log_dir", type=str, help="Directory to save model logs and results")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
args = parser.parse_args()

# Dataset Cleanup 
def purge_invalid_images(base_path):
    removed = 0
    for category in ("Cat", "Dog"):
        folder = os.path.join(base_path, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img_data = tf.io.read_file(img_path)
                tf.image.decode_jpeg(img_data, channels=3)
            except:
                os.remove(img_path)
                removed += 1
    print(f"Removed {removed} corrupted images.")

#  Dataset Setup 
data_root = "/home/jojo24tu/example/ASSIGNMENT1/PetImages"
purge_invalid_images(data_root)

img_size = (180, 180)
batch = 64

train_data = image_dataset_from_directory(
    data_root,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch
)
valid_data = image_dataset_from_directory(
    data_root,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch
)

#  Augmentation & Normalization 
augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

train_data = train_data.map(lambda x, y: (augmentation(x), y)).prefetch(tf.data.AUTOTUNE)
valid_data = valid_data.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

#  Pretrained Model 
print("Importing trained Stanford Dogs model")
base_model = keras.models.load_model("stanford_dogs_model.h5", compile=False)

# Modify last layer
last_features = base_model.layers[-2].output
new_head = layers.Dense(1, activation="sigmoid")(last_features)
model = keras.Model(inputs=base_model.input, outputs=new_head)

# Unfreeze only the last two conv layers
conv_layers = [layer for layer in model.layers if isinstance(layer, (layers.Conv2D, layers.SeparableConv2D))]
for layer in conv_layers[-2:]:
    layer.trainable = True
for layer in model.layers:
    if layer not in conv_layers[-2:] and layer != model.layers[-1]:
        layer.trainable = False

#  Compile Model 
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#  Callbacks
os.makedirs(args.log_dir, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.log_dir, "exp4_model.h5"),
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

# Training 
print("Training Experiment 4...")
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=args.epochs,
    callbacks=callbacks
)

# Logging Results
txt_path = os.path.join(args.log_dir, "exp4_log.txt")
csv_path = os.path.join(args.log_dir, "exp4_results.csv")

with open(txt_path, "w") as txt_file, open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss"])

    txt_file.write("Experiment 4: Replace output + last 2 conv layers\n")
    print("\nExperiment 4 Results:")
    for i in range(len(history.history["accuracy"])):
        row = [
            i + 1,
            history.history["accuracy"][i],
            history.history["val_accuracy"][i],
            history.history["loss"][i],
            history.history["val_loss"][i],
        ]
        line = f"Epoch {row[0]:02d} | Train Acc: {row[1]:.4f} | Val Acc: {row[2]:.4f} | Train Loss: {row[3]:.4f} | Val Loss: {row[4]:.4f}"
        print(line)
        txt_file.write(line + "\n")
        writer.writerow(row)
