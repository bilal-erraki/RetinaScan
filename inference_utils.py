"""Backward-compatible aliases for the modular inference implementation."""

from pathlib import Path

from src.config import CLASS_NAMES as CLASS_NAME_LIST
from src.predict import resolve_model_path
from src.preprocessing import crop_black_borders as crop_black_borders_pil  # noqa: F401
from src.preprocessing import (
    preprocess_image_from_bytes,
    preprocess_image_from_path,
)

CLASS_NAMES = dict(enumerate(CLASS_NAME_LIST))


def get_default_model_path():
    return str(resolve_model_path())


def load_retinascan_model(model_path):
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


def get_model_image_size(model, fallback=300):
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    return int(shape[1]) if shape[1] and shape[1] == shape[2] else fallback


def should_crop_retina(model_path):
    return True


def preprocess_image(image_source, image_size, crop_retina=True):
    del crop_retina
    if isinstance(image_source, bytes):
        return preprocess_image_from_bytes(image_source, image_size)
    return preprocess_image_from_path(Path(image_source), image_size)
