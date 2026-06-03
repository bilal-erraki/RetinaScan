import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models


# =========================
# 1. Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection")

train_csv_path = os.path.join(DATA_DIR, "train_split.csv")
val_csv_path = os.path.join(DATA_DIR, "val_split.csv")
train_images_path = os.path.join(DATA_DIR, "train_images")

print("Loading data...")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))


# =========================
# 2. Image paths
# =========================

train_df["image_path"] = train_df["id_code"].apply(
    lambda x: os.path.join(train_images_path, x + ".png")
)

val_df["image_path"] = val_df["id_code"].apply(
    lambda x: os.path.join(train_images_path, x + ".png")
)


# =========================
# 3. Preprocessing
# =========================

IMG_SIZE = 224
BATCH_SIZE = 32

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)

    return image, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_df["diagnosis"].astype("int32").values
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["diagnosis"].astype("int32").values
    )
)

train_ds = train_ds.map(load_image).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =========================
# 4. Class weights
# =========================

print("Class distribution:")
print(train_df["diagnosis"].value_counts().sort_index())

classes = np.unique(train_df["diagnosis"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["diagnosis"]
)

class_weights = {
    int(c): float(np.sqrt(w))
    for c, w in zip(classes, weights)
}

print("Class weights:")
print(class_weights)


# =========================
# 5. Data augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# =========================
# 6. MobileNetV2 baseline model
# =========================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

checkpoint_head = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(BASE_DIR, "best_head_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        mode="max"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    ),
    checkpoint_head
]


# =========================
# 7. Train model
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================
# 8. Fine-tuning MobileNetV2
# =========================

print("Starting fine-tuning...")

base_model.trainable = True

# Freeze all layers except the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compile again with a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_finetune = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(BASE_DIR, "best_finetuned_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

fine_tune_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        mode="max"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    ),
    checkpoint_finetune

]

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks
)


# =========================
# 9. Evaluate model
# =========================

val_loss, val_acc = model.evaluate(val_ds)

print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)


# =========================
# 10. Classification report
# =========================

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# =========================
# 11. Save model
# =========================

model_path = os.path.join(BASE_DIR, "retinascan_mobilenetv2_baseline.keras")
model.save(model_path)

print("Model saved at:", model_path)


# =========================
# 12. Plot training curves
# =========================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

