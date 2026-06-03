import os
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_finetuned_model.keras")

IMG_SIZE = 224

class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

app = FastAPI(title="RetinaScan API")

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False,
    custom_objects={"preprocess_input": preprocess_input}
)

print("Model loaded successfully")


def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.get("/")
def home():
    return {"message": "RetinaScan API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_array = preprocess_image(image_bytes)

    predictions = model.predict(image_array)

    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    probabilities = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities
    }