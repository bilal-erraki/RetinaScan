import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "aptos2019-blindness-detection")

MODEL_PATH = os.path.join(BASE_DIR, "best_finetuned_model.keras")
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")

IMG_SIZE = 224

class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# =========================
# Load model
# =========================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    safe_mode=False,
    custom_objects={"preprocess_input": preprocess_input}
)

print("Model loaded successfully")

# =========================
# Choose one image from validation set
# =========================

val_df = pd.read_csv(VAL_CSV)

sample = val_df.iloc[0]
image_id = sample["id_code"]
true_label = int(sample["diagnosis"])

image_path = os.path.join(IMAGE_DIR, image_id + ".png")

print("Image:", image_path)
print("True label:", true_label, "-", class_names[true_label])

# =========================
# Preprocess image
# =========================

image = Image.open(image_path).convert("RGB")
image = image.resize((IMG_SIZE, IMG_SIZE))

image_array = np.array(image).astype("float32")
image_array = np.expand_dims(image_array, axis=0)

# Important:
# Do NOT apply preprocess_input here
# because it is already inside your saved model.

# =========================
# Prediction
# =========================

predictions = model.predict(image_array)

predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print("Predicted class:", predicted_class, "-", class_names[predicted_class])
print("Confidence:", round(confidence * 100, 2), "%")

print("\nAll probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"{i} - {class_names[i]}: {prob * 100:.2f}%")