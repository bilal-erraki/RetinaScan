"""FastAPI service for RetinaScan inference."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MACRO_F1,
    MEDICAL_DISCLAIMER,
    MODEL_NAME,
    MODEL_VERSION,
    MODELS_DIR,
    QUADRATIC_WEIGHTED_KAPPA,
    VALIDATION_ACCURACY,
    WEIGHTED_F1,
)
from src.predict import predict_image_bytes

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="RetinaScan API",
    version=MODEL_VERSION,
    description="Academic diabetic retinopathy classification demo.",
)


def _default_model_info() -> dict:
    return {
        "model_name": MODEL_NAME,
        "task": "Diabetic retinopathy classification",
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "validation_accuracy": VALIDATION_ACCURACY,
        "quadratic_weighted_kappa": QUADRATIC_WEIGHTED_KAPPA,
        "macro_f1": MACRO_F1,
        "weighted_f1": WEIGHTED_F1,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


def _read_model_info() -> dict:
    metadata_path = MODELS_DIR / "model_info.json"
    if not metadata_path.is_file():
        return _default_model_info()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_model_info()
    return {
        "model_name": metadata.get("model_name", MODEL_NAME),
        "task": "Diabetic retinopathy classification",
        "classes": metadata.get("classes", CLASS_NAMES),
        "image_size": metadata.get("image_size", IMAGE_SIZE),
        "validation_accuracy": metadata.get("validation_accuracy", VALIDATION_ACCURACY),
        "quadratic_weighted_kappa": metadata.get(
            "quadratic_weighted_kappa", QUADRATIC_WEIGHTED_KAPPA
        ),
        "macro_f1": metadata.get("macro_f1", MACRO_F1),
        "weighted_f1": metadata.get("weighted_f1", WEIGHTED_F1),
        "medical_disclaimer": metadata.get("disclaimer", MEDICAL_DISCLAIMER),
    }


@app.get("/")
def home() -> dict:
    return {"message": "Welcome to the RetinaScan API"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "RetinaScan API",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


@app.get("/model-info")
def model_info() -> dict:
    return _read_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    extension = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG files are accepted.")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image content type.")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")
    try:
        return predict_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
