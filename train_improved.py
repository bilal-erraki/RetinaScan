import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB3


# =========================
# 1. Reproducibility / paths
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection")
HISTORY_DIR = os.path.join(BASE_DIR, "history")

train_csv_path = os.path.join(DATA_DIR, "train_split.csv")
val_csv_path = os.path.join(DATA_DIR, "val_split.csv")
train_images_path = os.path.join(DATA_DIR, "train_images")

os.makedirs(HISTORY_DIR, exist_ok=True)

IMG_SIZE = int(os.environ.get("RETINASCAN_IMG_SIZE", "300"))
BATCH_SIZE = int(os.environ.get("RETINASCAN_BATCH_SIZE", "16"))
HEAD_EPOCHS = int(os.environ.get("RETINASCAN_HEAD_EPOCHS", "12"))
FINE_TUNE_EPOCHS = int(os.environ.get("RETINASCAN_FINE_TUNE_EPOCHS", "25"))
MAX_TRAIN_SAMPLES = int(os.environ.get("RETINASCAN_MAX_TRAIN_SAMPLES", "0"))
MAX_VAL_SAMPLES = int(os.environ.get("RETINASCAN_MAX_VAL_SAMPLES", "0"))
NUM_CLASSES = 5

MODEL_WEIGHTS = os.environ.get("RETINASCAN_WEIGHTS", "imagenet")
if MODEL_WEIGHTS.strip().lower() in {"", "none", "random"}:
    MODEL_WEIGHTS = None

if os.environ.get("RETINASCAN_MIXED_PRECISION") == "1":
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


# =========================
# 2. Data loading
# =========================

print("Loading data...")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)


def limit_samples(df, max_samples, name):
    if max_samples <= 0 or max_samples >= len(df):
        return df

    per_class = max(1, max_samples // NUM_CLASSES)
    limited_parts = [
        group.sample(min(len(group), per_class), random_state=SEED)
        for _, group in df.groupby("diagnosis")
    ]
    limited = pd.concat(limited_parts)
    limited = limited.sample(frac=1, random_state=SEED).head(max_samples)
    limited = limited.reset_index(drop=True)

    print(f"Using {len(limited)} {name} samples because debug sample limit is set.")
    return limited


train_df = limit_samples(train_df, MAX_TRAIN_SAMPLES, "training")
val_df = limit_samples(val_df, MAX_VAL_SAMPLES, "validation")

train_df["image_path"] = train_df["id_code"].apply(
    lambda x: os.path.join(train_images_path, x + ".png")
)
val_df["image_path"] = val_df["id_code"].apply(
    lambda x: os.path.join(train_images_path, x + ".png")
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Train distribution:")
print(train_df["diagnosis"].value_counts().sort_index())
print("Validation distribution:")
print(val_df["diagnosis"].value_counts().sort_index())


def crop_black_borders(image, threshold=8.0):
    """Remove black camera borders before resizing the retinal field."""
    image = tf.cast(image, tf.float32)
    gray = tf.reduce_max(image, axis=-1)
    mask = gray > threshold
    coords = tf.cast(tf.where(mask), tf.int32)

    def cropped():
        y_min = tf.reduce_min(coords[:, 0])
        x_min = tf.reduce_min(coords[:, 1])
        y_max = tf.reduce_max(coords[:, 0])
        x_max = tf.reduce_max(coords[:, 1])

        height = y_max - y_min + 1
        width = x_max - x_min + 1
        return tf.image.crop_to_bounding_box(image, y_min, x_min, height, width)

    return tf.cond(tf.size(coords) > 0, cropped, lambda: image)


def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])

    image = crop_black_borders(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bicubic")
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, tf.cast(label, tf.int32)


def make_dataset(df, training):
    ds = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path"].values,
            df["diagnosis"].astype("int32").values,
        )
    )
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)


# =========================
# 3. Imbalance handling
# =========================

classes = np.unique(train_df["diagnosis"])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["diagnosis"],
)

# Square-root weights are gentler than fully balanced weights on noisy classes.
class_weights = {int(c): float(np.sqrt(w)) for c, w in zip(classes, weights)}

print("Class weights:")
print(class_weights)


# =========================
# 4. Model
# =========================

augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(height_factor=(-0.05, 0.12), width_factor=(-0.05, 0.12)),
    layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
    layers.RandomContrast(0.15),
]

if hasattr(layers, "RandomBrightness"):
    try:
        augmentation_layers.append(layers.RandomBrightness(0.08, value_range=(0, 255)))
    except TypeError:
        augmentation_layers.append(layers.RandomBrightness(0.08))

data_augmentation = tf.keras.Sequential(augmentation_layers, name="retina_augmentation")

base_model = EfficientNetB3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=MODEL_WEIGHTS,
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.30)(x)

# Keep the output float32 even when mixed precision is enabled.
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs, name="retinascan_efficientnetb3")


def make_optimizer(learning_rate, weight_decay=1e-5):
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def compile_model(learning_rate):
    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


class QuadraticKappaCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true = []
        y_pred = []

        for images, labels in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(np.argmax(predictions, axis=1).tolist())

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        logs["val_qwk"] = qwk
        print(f"\nval_qwk: {qwk:.4f}")


class BestQWKCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, initial_best=-np.inf):
        super().__init__()
        self.filepath = filepath
        self.best = initial_best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get("val_qwk")
        if current is None:
            return

        if current > self.best:
            self.best = current
            self.model.save(self.filepath)
            print(f"Saved best QWK model to {self.filepath}")


class BestMetricCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor, mode="max", initial_best=None):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = initial_best
        self.compare = np.greater if mode == "max" else np.less

        if self.best is None:
            self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.compare(current, self.best):
            self.best = current
            self.model.save(self.filepath)
            print(f"Saved best {self.monitor} model to {self.filepath}")


best_model_path = os.environ.get(
    "RETINASCAN_BEST_MODEL_PATH",
    os.path.join(BASE_DIR, "best_improved_model.keras"),
)
best_accuracy_model_path = os.environ.get(
    "RETINASCAN_BEST_ACCURACY_MODEL_PATH",
    os.path.join(BASE_DIR, "best_improved_accuracy_model.keras"),
)
csv_log_path = os.environ.get(
    "RETINASCAN_CSV_LOG_PATH",
    os.path.join(HISTORY_DIR, "improved_training_log.csv"),
)

for output_path in (best_model_path, best_accuracy_model_path, csv_log_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def make_callbacks(initial_best=-np.inf, append_log=False):
    return [
        QuadraticKappaCallback(val_ds),
        BestQWKCheckpoint(best_model_path, initial_best=initial_best),
        BestMetricCheckpoint(best_accuracy_model_path, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_qwk",
            patience=6,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path, append=append_log),
    ]


# =========================
# 5. Train classifier head
# =========================

compile_model(learning_rate=1e-3)
model.summary()

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

best_head_qwk = max(history_head.history.get("val_qwk", [-np.inf]))


# =========================
# 6. Fine-tune EfficientNetB3
# =========================

print("Starting fine-tuning...")

base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

# BatchNorm statistics are fragile on small medical datasets.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

compile_model(learning_rate=2e-5)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(initial_best=best_head_qwk, append_log=True),
)


# =========================
# 7. Evaluate best checkpoint
# =========================

if os.path.exists(best_model_path):
    print("Loading best QWK checkpoint:", best_model_path)
    model = tf.keras.models.load_model(best_model_path, compile=False)
    compile_model(learning_rate=2e-5)

val_loss, val_acc = model.evaluate(val_ds, verbose=1)

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(predictions, axis=1).tolist())

val_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)
print("Validation quadratic weighted kappa:", val_qwk)

print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

final_model_path = os.environ.get(
    "RETINASCAN_FINAL_MODEL_PATH",
    os.path.join(BASE_DIR, "retinascan_efficientnetb3_improved.keras"),
)
final_model_dir = os.path.dirname(final_model_path)
if final_model_dir:
    os.makedirs(final_model_dir, exist_ok=True)

model.save(final_model_path)

print("Best model saved at:", best_model_path)
print("Best accuracy model saved at:", best_accuracy_model_path)
print("Final model saved at:", final_model_path)
print("Training log saved at:", csv_log_path)
