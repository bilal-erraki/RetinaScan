"""Simple Streamlit client for the RetinaScan API."""

from __future__ import annotations

import os

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, MEDICAL_DISCLAIMER

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
CLASS_DESCRIPTIONS = {
    "No DR": "No diabetic retinopathy detected.",
    "Mild": "Mild signs of diabetic retinopathy.",
    "Moderate": "Moderate diabetic retinopathy detected.",
    "Severe": "Severe diabetic retinopathy detected.",
    "Proliferative DR": "Advanced diabetic retinopathy detected.",
}


def main() -> None:
    st.set_page_config(page_title="RetinaScan", page_icon="👁️", layout="centered")
    st.title("👁️ RetinaScan")
    st.subheader("Diabetic Retinopathy Classification")
    st.write("Upload a retina fundus image to classify its diabetic retinopathy stage.")

    uploaded_file = st.file_uploader("Choose a retina image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("This file is not a valid PNG or JPEG image.")
        else:
            st.image(image, caption="Uploaded retina image", use_container_width=True)
            if st.button("Predict", type="primary"):
                _request_prediction(uploaded_file)

    st.divider()
    st.warning(MEDICAL_DISCLAIMER)


def _request_prediction(uploaded_file) -> None:
    api_url = os.getenv("RETINASCAN_API_URL", DEFAULT_API_URL)
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    try:
        with st.spinner("Analyzing image..."):
            response = requests.post(api_url, files=files, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not reach the RetinaScan API: {exc}")
        return

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Prediction failed: {detail}")
        return

    result = response.json()
    predicted_label = result["predicted_label"]
    st.success("Prediction completed")
    st.metric("Predicted class", predicted_label)
    st.metric("Confidence", f"{result['confidence']:.2f}%")
    st.write(CLASS_DESCRIPTIONS.get(predicted_label, ""))
    st.subheader("Class probabilities")
    for label in CLASS_NAMES:
        probability = float(result["probabilities"].get(label, 0.0))
        st.write(f"{label}: {probability:.2f}%")
        st.progress(min(max(probability / 100.0, 0.0), 1.0))


if __name__ == "__main__":
    main()
