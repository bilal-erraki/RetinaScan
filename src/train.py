"""Train RetinaScan's EfficientNetB3 model and log the run with MLflow.

Training starts only when this file is run. The DVC train stage is its primary entry
point; importing this module never starts training.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_class_weight

from src.config import MODEL_PATH, PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_directories
from src.data_loader import make_tf_dataset


def load_params(path: str | Path = PROJECT_ROOT / "params.yaml") -> dict:
    """Load pipeline settings from YAML."""

    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DVC prepare-stage outputs or explain how to create them."""

    train_path = PROCESSED_DATA_DIR / "train_split.csv"
    validation_path = PROCESSED_DATA_DIR / "val_split.csv"
    missing = [path for path in (train_path, validation_path) if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing processed split(s): {names}. Run `dvc repro prepare` first."
        )
    return pd.read_csv(train_path), pd.read_csv(validation_path)


def build_model(tf, train_params: dict, image_size: int, num_classes: int):
    """Build the configured EfficientNetB3 classifier."""

    backbone_name = str(train_params["backbone"])
    if backbone_name != "EfficientNetB3":
        raise ValueError(f"Unsupported backbone: {backbone_name}. Expected EfficientNetB3.")

    weights = train_params.get("weights", "imagenet")
    if str(weights).lower() in {"none", "null", "random"}:
        weights = None

    layers = tf.keras.layers
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(height_factor=(-0.05, 0.12), width_factor=(-0.05, 0.12)),
            layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
            layers.RandomContrast(0.15),
        ],
        name="retina_augmentation",
    )
    backbone = tf.keras.applications.EfficientNetB3(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights
    )
    backbone.trainable = False

    inputs = layers.Input((image_size, image_size, 3))
    features = augmentation(inputs)
    features = backbone(features, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(0.35)(features)
    features = layers.Dense(256, activation="relu")(features)
    features = layers.Dropout(0.30)(features)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(features)
    return tf.keras.Model(inputs, outputs, name="retinascan_efficientnetb3"), backbone


def make_optimizer(tf, learning_rate: float, strategy: str):
    """Create the optimizer selected in params.yaml."""

    normalized = strategy.strip().lower()
    if normalized == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if normalized in {"adamw", "adamw_or_adam"}:
        try:
            return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        except AttributeError:
            if normalized == "adamw":
                raise RuntimeError("This TensorFlow version does not provide AdamW.")
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer strategy: {strategy}")


def compile_model(tf, model, learning_rate: float, optimizer_strategy: str, loss_name: str) -> None:
    model.compile(
        optimizer=make_optimizer(tf, learning_rate, optimizer_strategy),
        loss=loss_name,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def plot_training_history(histories: list, output_path: Path) -> str:
    """Save combined head-training and fine-tuning curves."""

    import matplotlib.pyplot as plt

    merged: dict[str, list] = {}
    for history in histories:
        for name, values in history.history.items():
            merged.setdefault(name, []).extend(values)

    phase_boundary = len(histories[0].history.get("loss", []))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(merged.get("accuracy", []), label="train")
    axes[0].plot(merged.get("val_accuracy", []), label="validation")
    axes[0].axvline(phase_boundary - 0.5, color="gray", linestyle="--", label="fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(merged.get("loss", []), label="train")
    axes[1].plot(merged.get("val_loss", []), label="validation")
    axes[1].axvline(phase_boundary - 0.5, color="gray", linestyle="--", label="fine-tuning")
    axes[1].set_title("Loss")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def main() -> None:
    """Run two-phase training, save the best model, and record an MLflow run."""

    try:
        import mlflow
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before starting training.") from exc

    ensure_directories()
    params = load_params()
    data_params = params["data"]
    train_params = params["train"]
    random_state = int(data_params["random_state"])
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    train_frame, validation_frame = load_processed_splits()
    image_size = int(data_params["image_size"])
    batch_size = int(train_params["batch_size"])
    train_data = make_tf_dataset(
        tf, train_frame, image_size, batch_size, training=True, random_state=random_state
    )
    validation_data = make_tf_dataset(
        tf, validation_frame, image_size, batch_size, training=False, random_state=random_state
    )

    classes = np.unique(train_frame["diagnosis"])
    balanced = compute_class_weight("balanced", classes=classes, y=train_frame["diagnosis"])
    class_weights = {int(label): float(np.sqrt(weight)) for label, weight in zip(classes, balanced)}
    model, backbone = build_model(tf, train_params, image_size, int(data_params["num_classes"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    class BestValidationCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.best = -np.inf

        def on_epoch_end(self, epoch, logs=None):
            current = (logs or {}).get("val_accuracy")
            if current is not None and current > self.best:
                self.best = float(current)
                self.model.save(self.path)
                print(f"Saved best validation model to {self.path}")

    checkpoint = BestValidationCheckpoint(MODEL_PATH)

    def phase_callbacks():
        return [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(train_params.get("early_stopping_patience", 6)),
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7
            ),
        ]

    mlflow.set_experiment("retinascan_efficientnetb3")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "backbone": train_params["backbone"],
                "weights": train_params.get("weights", "imagenet"),
                "image_size": image_size,
                "batch_size": batch_size,
                "random_state": random_state,
                "learning_rate_phase_1": train_params["lr_head"],
                "learning_rate_fine_tuning": train_params["lr_finetune"],
                "epochs_phase_1": train_params["epochs_head"],
                "epochs_fine_tuning": train_params["epochs_finetune"],
                "optimizer": train_params["optimizer"],
                "loss": train_params["loss"],
                "class_weighting_strategy": train_params["class_weighting"],
            }
        )

        compile_model(
            tf,
            model,
            float(train_params["lr_head"]),
            str(train_params["optimizer"]),
            str(train_params["loss"]),
        )
        history_head = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=int(train_params["epochs_head"]),
            class_weight=class_weights,
            callbacks=phase_callbacks(),
        )

        backbone.trainable = True
        fine_tune_layers = int(train_params.get("fine_tune_layers", 80))
        for layer in backbone.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        compile_model(
            tf,
            model,
            float(train_params["lr_finetune"]),
            str(train_params["optimizer"]),
            str(train_params["loss"]),
        )
        history_finetune = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=int(train_params["epochs_finetune"]),
            class_weight=class_weights,
            callbacks=phase_callbacks(),
        )

        if not MODEL_PATH.is_file():
            model.save(MODEL_PATH)
        try:
            best_model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        except TypeError:
            best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        compile_model(
            tf,
            best_model,
            float(train_params["lr_finetune"]),
            str(train_params["optimizer"]),
            str(train_params["loss"]),
        )
        validation_loss, validation_accuracy = best_model.evaluate(validation_data, verbose=1)

        history_path = REPORTS_DIR / "training_history.png"
        plot_training_history([history_head, history_finetune], history_path)
        mlflow.log_metrics(
            {
                "best_validation_accuracy": float(validation_accuracy),
                "best_validation_loss": float(validation_loss),
            }
        )
        mlflow.log_artifact(str(history_path), artifact_path="figures")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

    print(f"Training complete. Best model: {MODEL_PATH}")
    print(f"Training curves: {REPORTS_DIR / 'training_history.png'}")


if __name__ == "__main__":
    main()
