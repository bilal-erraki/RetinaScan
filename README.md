# RetinaScan MLOps

RetinaScan is an academic deep-learning project that classifies diabetic retinopathy from retinal fundus images. It uses an EfficientNetB3 model trained on the APTOS 2019 dataset and packages the workflow as reproducible data preparation, tracked training, evaluation, API inference, and a small web interface.

> **Medical disclaimer:** RetinaScan is an academic demo only. It is not a medical diagnostic device and must not replace assessment by a qualified clinician.

## Pipeline

```text
APTOS images and labels
         |
         v
DVC prepare -> processed train/validation splits
                              |
                              v
               DVC train + MLflow tracking
                              |
                              v
                 models/best_model.keras
                       |             |
                       v             v
               DVC evaluate      FastAPI -> Streamlit
                       |
                       v
        fresh metrics + report + confusion matrix

DVC runs and tracks the complete prepare -> train -> evaluate dependency graph.
Pytest and pre-commit provide automated quality checks.
```

## Project structure

```text
RetinaScan/
|-- api/                    FastAPI service
|-- data/raw/               local APTOS data (not committed)
|-- data/processed/         generated train/validation splits
|-- models/                 model metadata and local model location
|-- notebooks/              exploratory analysis
|-- reports/                metrics and generated figures
|-- src/                    preprocessing, training, evaluation, prediction
|-- tests/                  pytest suite
|-- webapp/                 Streamlit client
|-- dvc.yaml                reproducible pipeline stages
|-- params.yaml             data and training parameters
`-- requirements.txt        project dependencies
```

The original `best_improved_model.keras` location remains supported. You may instead place the model at `models/best_model.keras` or set `RETINASCAN_MODEL_PATH` to another local path.

## Installation

Python 3.10 or 3.11 is recommended. TensorFlow 2.21.0 is pinned because that version was verified with the saved Keras model. In Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The configured dataset location is `aptos2019-blindness-detection/` and must contain `train.csv` and `train_images/`. To use `data/raw/aptos2019-blindness-detection/` instead, update `data.dataset_dir` in `params.yaml`; DVC uses that same parameter for its source dependencies.

## DVC workflow

Parameters live in `params.yaml`; stages live in `dvc.yaml`.

```powershell
dvc repro
dvc dag
dvc metrics show
```

`dvc repro` runs a real reproducible pipeline:

1. `prepare` reads APTOS `train.csv` and creates portable, stratified train/validation CSVs.
2. `train` trains and fine-tunes EfficientNetB3, logs the run with MLflow, and saves `models/best_model.keras` plus training curves.
3. `evaluate` loads that saved model, predicts the validation split, and writes fresh metrics, a classification report, and a confusion matrix.

Training is computationally expensive. Run the full pipeline only when the dataset and suitable compute are available. Individual stages can be run with `dvc repro prepare`, `dvc repro train`, and `dvc repro evaluate`.

Large datasets, model binaries, MLflow runs, and generated figures stay outside normal Git tracking. Commit `dvc.yaml`, `dvc.lock`, `params.yaml`, `.dvc/config`, and any standalone `.dvc` pointer files.

## Training and MLflow

Training is never triggered by tests or imports. Start it explicitly only when the dataset and suitable compute are available:

```powershell
python src/train.py
```

The script runs frozen-head training followed by fine-tuning, logs parameters and validation results to the `retinascan_efficientnetb3` experiment, and records the best model and training curves. The separate evaluation stage creates the classification report, confusion matrix, and DVC metrics. Inspect MLflow runs with:

```powershell
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## Run the API

```powershell
python -m uvicorn api.app:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. Endpoints:

- `GET /` — welcome message
- `GET /health` — service and model version
- `GET /model-info` — classes, input size, reported results, and disclaimer
- `POST /predict` — PNG or JPEG multipart upload

Prediction probabilities and confidence are returned as percentages from 0 to 100.

## Run Streamlit

Start the API first, then in another PowerShell window run:

```powershell
python -m streamlit run webapp/streamlit_app.py
```

Set `RETINASCAN_API_URL` if the prediction endpoint is hosted somewhere other than `http://127.0.0.1:8000/predict`.

## Tests and code quality

Tests mock model inference, so they do not repeatedly load the large Keras model:

```powershell
pytest -v
pytest --cov=src
pre-commit run --all-files
```

## Historical reference experiment

| Metric | Value |
|---|---:|
| Validation accuracy | 0.7858 |
| Validation loss | 0.6081 |
| Quadratic weighted kappa | 0.8603 |
| Macro F1 | 0.6021 |
| Weighted F1 | 0.7773 |

These values are the historical reference recorded in `models/model_info.json`. A successful `dvc repro` replaces `reports/metrics.json` with metrics freshly computed from the newly trained model.

## Technologies

Python, TensorFlow/Keras, EfficientNetB3, pandas, scikit-learn, Pillow, FastAPI, Streamlit, MLflow, DVC, pytest, Black, isort, flake8, and pre-commit.

## Git and GitHub notes

- Commit source, tests, configuration, documentation, metrics JSON, model metadata, and DVC pointer files.
- Keep local data, generated figures, experiment runs, and Keras/HDF5 binaries out of normal Git history.
- Run tests and pre-commit before opening a pull request.
- Use clear, scoped commit messages such as `feat: add FastAPI model metadata endpoint`.
