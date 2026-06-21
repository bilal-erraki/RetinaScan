"""Image preprocessing shared by training, CLI, and API inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import IMAGE_SIZE

ImageInput = Union[Image.Image, np.ndarray]


def crop_black_borders(image: ImageInput, threshold: int = 8) -> Image.Image:
    """Crop black borders around a retina image when possible."""

    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    pil_image = pil_image.convert("RGB")
    array = np.asarray(pil_image)
    mask = array.max(axis=2) > threshold
    if not mask.any():
        return pil_image

    rows, columns = np.where(mask)
    return pil_image.crop(
        (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
    )


def _prepare_image(image: Image.Image, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer")

    image = crop_black_borders(image.convert("RGB"))
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((image_size, image_size), resampling.BICUBIC)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def preprocess_image_from_path(image_path: str | Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load an image from disk and return a preprocessed prediction batch."""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file was not found: {path}")

    try:
        with Image.open(path) as image:
            image.load()
            return _prepare_image(image, image_size)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unsupported image file: {path}") from exc


def preprocess_image_from_bytes(image_bytes: bytes, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load uploaded image bytes and return a preprocessed prediction batch."""

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return _prepare_image(image, image_size)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid PNG or JPEG image.") from exc
