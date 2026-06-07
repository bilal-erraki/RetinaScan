import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def get_default_model_path():
    env_path = os.environ.get("RETINASCAN_MODEL_PATH")
    if env_path:
        return env_path

    improved_path = os.path.join(BASE_DIR, "best_improved_model.keras")
    if os.path.exists(improved_path):
        return improved_path

    return os.path.join(BASE_DIR, "best_finetuned_model.keras")


def load_retinascan_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input},
    )


def get_model_image_size(model, fallback=224):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1]
    width = input_shape[2]

    if height is None or width is None or height != width:
        return fallback

    return int(height)


def should_crop_retina(model_path):
    override = os.environ.get("RETINASCAN_CROP_RETINA")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}

    return "improved" in os.path.basename(model_path).lower()


def crop_black_borders_pil(image, threshold=8):
    array = np.asarray(image)
    mask = array.max(axis=2) > threshold

    if not mask.any():
        return image

    ys, xs = np.where(mask)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1

    return image.crop((left, top, right, bottom))


def preprocess_image(image_source, image_size, crop_retina=False):
    if isinstance(image_source, bytes):
        image = Image.open(BytesIO(image_source)).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    if crop_retina:
        image = crop_black_borders_pil(image)

    bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    image = image.resize((image_size, image_size), bicubic)
    image_array = np.array(image).astype("float32")
    return np.expand_dims(image_array, axis=0)
