"""Model loading and reusable prediction helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH, MODEL_VERSION, PROJECT_ROOT
from src.preprocessing import preprocess_image_from_bytes, preprocess_image_from_path


def resolve_model_path() -> Path:
    """Return the configured model path, including legacy fallbacks."""

    configured_path = os.getenv("RETINASCAN_MODEL_PATH")
    candidates = [
        Path(configured_path).expanduser() if configured_path else None,
        MODEL_PATH,
        PROJECT_ROOT / "best_improved_model.keras",
        PROJECT_ROOT / "retinascan_efficientnetb3_improved.keras",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()

    checked = "\n - ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "No RetinaScan model was found. Put the trained model at "
        f"'{MODEL_PATH}' or set RETINASCAN_MODEL_PATH. Checked:\n - {checked}"
    )


@lru_cache(maxsize=1)
def load_model_once() -> Any:
    """Load the Keras model once and cache it for subsequent predictions."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on the local runtime
        raise RuntimeError("TensorFlow is required to load the RetinaScan model.") from exc

    model_path = resolve_model_path()
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:  # TensorFlow/Keras versions without the safe_mode argument
        return tf.keras.models.load_model(model_path, compile=False)


def _prediction_result(predictions: Any) -> dict:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim == 2:
        values = values[0]
    values = values.reshape(-1)

    if values.size != len(CLASS_NAMES) or not np.isfinite(values).all():
        raise ValueError(f"Model output must contain {len(CLASS_NAMES)} finite class scores.")

    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
        shifted = values - values.max()
        values = np.exp(shifted) / np.exp(shifted).sum()

    predicted_class = int(np.argmax(values))
    percentages = values * 100.0
    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": round(float(percentages[predicted_class]), 2),
        "probabilities": {
            label: round(float(percentages[index]), 2) for index, label in enumerate(CLASS_NAMES)
        },
        "model_version": MODEL_VERSION,
    }


def _predict_batch(batch: np.ndarray) -> dict:
    predictions = load_model_once().predict(batch, verbose=0)
    return _prediction_result(predictions)


def predict_image(image_path: str | Path) -> dict:
    """Predict diabetic retinopathy class from a local image path."""

    return _predict_batch(preprocess_image_from_path(image_path, IMAGE_SIZE))


def predict_image_bytes(image_bytes: bytes) -> dict:
    """Predict diabetic retinopathy class from uploaded image bytes."""

    return _predict_batch(preprocess_image_from_bytes(image_bytes, IMAGE_SIZE))
