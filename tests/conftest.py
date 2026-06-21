from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image


@pytest.fixture
def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (80, 60), color=(120, 45, 20)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def image_path(tmp_path, image_bytes):
    path = tmp_path / "retina.png"
    path.write_bytes(image_bytes)
    return path
