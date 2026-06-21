import numpy as np
import pytest

from src.preprocessing import preprocess_image_from_bytes, preprocess_image_from_path


def test_preprocess_valid_image_path(image_path):
    batch = preprocess_image_from_path(image_path)
    assert batch.shape == (1, 300, 300, 3)
    assert np.issubdtype(batch.dtype, np.number)


def test_preprocess_valid_image_bytes(image_bytes):
    batch = preprocess_image_from_bytes(image_bytes)
    assert batch.shape == (1, 300, 300, 3)
    assert batch.dtype == np.float32


def test_preprocess_invalid_bytes_raise_clear_error():
    with pytest.raises(ValueError, match="valid PNG or JPEG"):
        preprocess_image_from_bytes(b"this is not an image")
