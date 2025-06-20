import os
import csv
import argparse
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing import image_dataset_from_directory

#  Argument Handling
parser = argparse.ArgumentParser(description="Experiment 3: Output + First 2 Conv Layers Replaced")
parser.add_argument("output_path", type=str, help="Directory to save model and logs")
parser.add_argument("--epochs", type=int, default=50, help="Epoch count")
args = parser.parse_args()

# Dataset Cleanup 
image_root = "/home/jojo24tu/example/ASSIGNMENT1/PetImages"

def delete_corrupt_files(path):
    count = 0
    for label in ["Cat", "Dog"]:
        label_path = os.path.join(path, label)
        for img_file in os.listdir(label_path):
            full_path = os.path.join(label_path, img_file)
            try:
                content = tf.io.read_file(full_path)
                _ = tf.image.decode_jpeg(content, channels=3)
            except:
                os.remove(full_path)
                count += 1
    print(f"Removed {count} broken images.")

delete_corrupt_files(image_root)

#Dataset Loading
img_dims = (180, 180)
batch = 64

train_set = image_dataset_from_directory(
    image_root,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_dims,
    batch_size=batch,
)

val_set = image_dataset_from_directory(
    image_root,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_dims,
    batch_size=batch,
)

# Augmentation + Normalization 
augmentation = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
]

def apply_aug(images):
    for transform in augmentation:
        images = transform(images)
    return images

rescale = layers.Rescaling(1. / 255)

train_set = train_set.map(lambda x, y: (apply_aug(x), y), num_parallel_calls=AUTOTUNE)
train_set = train_set.map(lambda x, y: (rescale(x), y)).prefetch(AUTOTUNE)
val_set = val_set.map(lambda x, y: (rescale(x), y)).prefetch(AUTOTUNE)

# Load Base Model 
print("Loading pretrained Stanford Dogs model...")
base_model = keras.models.load_model("stanford_dogs_model.h5", compile=False)

#  Modify Layers 
penultimate = base_model.layers[-2].output
final_output = layers.Dense(1, activation="sigmoid")(penultimate)
updated_model = keras.Model(base_model.input, final_output)

# Unfreeze first 2 conv layers
conv_blocks = [l for l in updated_model.layers if isinstance(l, (layers.Conv2D, layers.SeparableConv2D))]
for i, layer in enumerate(updated_model.layers):
    if layer in conv_blocks[:2] or layer == updated_model.layers[-1]:
        layer.trainable = True
    else:
        layer.trainable = False

#  Training Setup
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(args.output_path, "model_exp3.h5"),
        save_best_only=True,
        monitor="val_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max"
    )
]

# Compile and Train 
updated_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

print("Beginning training process...")
history = updated_model.fit(
    train_set,
    validation_data=val_set,
    epochs=args.epochs,
    callbacks=callbacks
)

# Save Metrics 
text_file = os.path.join(args.output_path, "experiment3_summary.txt")
csv_file = os.path.join(args.output_path, "experiment3_metrics.csv")

with open(text_file, "w") as txt, open(csv_file, "w", newline="") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["Epoch", "Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss"])

    txt.write("Experiment 3 – Replaced Output Layer + First 2 Conv Layers\n")

    for i in range(len(history.history["accuracy"])):
        acc = history.history["accuracy"][i]
        val_acc = history.history["val_accuracy"][i]
        loss = history.history["loss"][i]
        val_loss = history.history["val_loss"][i]

        summary = (
            f"Epoch {i+1:02d} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        print(summary)
        txt.write(summary + "\n")
        writer.writerow([i+1, acc, val_acc, loss, val_loss])
