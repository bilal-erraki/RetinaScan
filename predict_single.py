import os
import numpy as np
import pandas as pd

from inference_utils import (
    CLASS_NAMES,
    get_default_model_path,
    get_model_image_size,
    load_retinascan_model,
    preprocess_image,
    should_crop_retina,
)

# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection")

MODEL_PATH = get_default_model_path()
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")

# =========================
# Load model
# =========================

print("Loading model:", MODEL_PATH)
model = load_retinascan_model(MODEL_PATH)
IMG_SIZE = get_model_image_size(model)
CROP_RETINA = should_crop_retina(MODEL_PATH)
print("Model loaded successfully")
print("Image size:", IMG_SIZE)
print("Crop retina:", CROP_RETINA)

# =========================
# Choose one image from validation set
# =========================

val_df = pd.read_csv(VAL_CSV)

sample = val_df.iloc[0]
image_id = sample["id_code"]
true_label = int(sample["diagnosis"])

image_path = os.path.join(IMAGE_DIR, image_id + ".png")

print("Image:", image_path)
print("True label:", true_label, "-", CLASS_NAMES[true_label])

# =========================
# Preprocess image
# =========================

image_array = preprocess_image(
    image_path,
    image_size=IMG_SIZE,
    crop_retina=CROP_RETINA,
)

# =========================
# Prediction
# =========================

predictions = model.predict(image_array)

predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print("Predicted class:", predicted_class, "-", CLASS_NAMES[predicted_class])
print("Confidence:", round(confidence * 100, 2), "%")

print("\nAll probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"{i} - {CLASS_NAMES[i]}: {prob * 100:.2f}%")
