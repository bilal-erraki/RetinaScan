from fastapi.testclient import TestClient

import api.app as api_module

client = TestClient(api_module.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert len(response.json()["classes"]) == 5
    assert "medical_disclaimer" in response.json()


def test_predict_rejects_invalid_extension():
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_predict_rejects_invalid_image_content():
    response = client.post("/predict", files={"file": ("retina.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_predict_uses_inference_result(monkeypatch, image_bytes):
    expected = {
        "predicted_class": 0,
        "predicted_label": "No DR",
        "confidence": 90.0,
        "probabilities": {
            "No DR": 90.0,
            "Mild": 2.0,
            "Moderate": 4.0,
            "Severe": 2.0,
            "Proliferative DR": 2.0,
        },
        "model_version": "1.0.0",
    }
    monkeypatch.setattr(api_module, "predict_image_bytes", lambda _: expected)
    response = client.post("/predict", files={"file": ("retina.png", image_bytes, "image/png")})
    assert response.status_code == 200
    assert response.json() == expected
