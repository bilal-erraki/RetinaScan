import numpy as np

from fastapi import FastAPI, UploadFile, File

from inference_utils import (
    CLASS_NAMES,
    get_default_model_path,
    get_model_image_size,
    load_retinascan_model,
    preprocess_image,
    should_crop_retina,
)

MODEL_PATH = get_default_model_path()

app = FastAPI(title="RetinaScan API")

print("Loading model:", MODEL_PATH)
model = load_retinascan_model(MODEL_PATH)
IMG_SIZE = get_model_image_size(model)
CROP_RETINA = should_crop_retina(MODEL_PATH)
print("Model loaded successfully")
print("Image size:", IMG_SIZE)
print("Crop retina:", CROP_RETINA)


@app.get("/")
def home():
    return {"message": "RetinaScan API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_array = preprocess_image(
        image_bytes,
        image_size=IMG_SIZE,
        crop_retina=CROP_RETINA,
    )

    predictions = model.predict(image_array)

    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities
    }
