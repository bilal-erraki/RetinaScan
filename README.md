# RetinaScan

RetinaScan is a deep learning project for diabetic retinopathy classification from fundus images using the APTOS 2019 dataset.

## Final Model

- Model: EfficientNetB3
- Input size: 300x300
- Validation accuracy: 78.58%
- Quadratic Weighted Kappa: 0.8603
- Weighted F1-score: 0.7773

## Classes

0. No DR
1. Mild
2. Moderate
3. Severe
4. Proliferative DR

## Technologies

- Python
- TensorFlow / Keras
- EfficientNetB3
- FastAPI
- Streamlit

## Run FastAPI

python -m uvicorn api:app --reload

Then open:

http://127.0.0.1:8000/docs

## Run Streamlit

python -m streamlit run app.py

## Note

This project is for educational purposes and should not replace professional medical diagnosis.
