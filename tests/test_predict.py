from unittest.mock import Mock

import numpy as np

from src import predict


def test_model_path_resolution_finds_preserved_model():
    model_path = predict.resolve_model_path()
    assert model_path.is_file()
    assert model_path.suffix == ".keras"


def test_prediction_output_with_mocked_model(monkeypatch, image_bytes):
    model = Mock()
    model.predict.return_value = np.array([[0.05, 0.64, 0.20, 0.07, 0.04]])
    monkeypatch.setattr(predict, "load_model_once", lambda: model)

    result = predict.predict_image_bytes(image_bytes)

    assert {
        "predicted_class",
        "predicted_label",
        "confidence",
        "probabilities",
        "model_version",
    }.issubset(result)
    assert result["predicted_class"] == 1
    assert isinstance(result["probabilities"], dict)
    assert 0 <= result["confidence"] <= 100
