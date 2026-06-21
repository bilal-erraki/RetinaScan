"""Evaluate the trained RetinaScan model on the prepared validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

from src.config import (  # noqa: E402
    CLASS_NAMES,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_directories,
)
from src.data_loader import make_tf_dataset  # noqa: E402


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_proba=None
) -> dict:
    """Compute accuracy, macro F1, weighted F1, and quadratic weighted kappa."""

    del y_proba  # Reserved for future probability-based metrics.
    return {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_metrics(metrics: dict, output_path: str | Path = "reports/metrics.json") -> None:
    """Save freshly computed metrics to a DVC-readable JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> str:
    """Save a labeled confusion matrix figure and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=list(class_names),
        cmap="Blues",
        xticks_rotation=30,
    )
    display.ax_.set_title("RetinaScan validation confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)
    return str(path)


def save_classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], output_path: str | Path
) -> str:
    """Save the per-class validation report as JSON."""

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return str(path)


def load_trained_model(tf, model_path: Path):
    """Load the DVC train-stage model without restoring optimizer state."""

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run `dvc repro train` first."
        )
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DVC-trained RetinaScan model.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--validation-csv", type=Path, default=PROCESSED_DATA_DIR / "val_split.csv")
    parser.add_argument("--metrics", type=Path, default=REPORTS_DIR / "metrics.json")
    parser.add_argument(
        "--confusion-matrix", type=Path, default=REPORTS_DIR / "confusion_matrix.png"
    )
    parser.add_argument(
        "--classification-report",
        type=Path,
        default=REPORTS_DIR / "classification_report.json",
    )
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to evaluate the trained model.") from exc

    ensure_directories()
    if not args.validation_csv.is_file():
        raise FileNotFoundError(
            f"Validation split not found: {args.validation_csv}. Run `dvc repro prepare` first."
        )
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
    image_size = int(params["data"]["image_size"])
    batch_size = int(params["train"]["batch_size"])
    validation_frame = pd.read_csv(args.validation_csv)
    validation_data = make_tf_dataset(
        tf,
        validation_frame,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        random_state=int(params["data"]["random_state"]),
    )

    model = load_trained_model(tf, args.model)
    probabilities = model.predict(validation_data, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    truth = validation_frame["diagnosis"].astype(int).to_numpy()
    if len(probabilities) != len(truth):
        raise ValueError(
            f"Model returned {len(probabilities)} predictions for {len(truth)} validation rows."
        )
    loss_function = tf.keras.losses.get(str(params["train"]["loss"]))
    loss_values = loss_function(truth, probabilities)
    validation_loss = float(np.mean(np.asarray(loss_values)))

    metrics = compute_classification_metrics(truth, predictions, probabilities)
    metrics["validation_loss"] = validation_loss
    save_metrics(metrics, args.metrics)
    plot_confusion_matrix(truth, predictions, args.confusion_matrix)
    save_classification_report(truth, predictions, args.classification_report)

    main_metric = str(params["evaluate"]["main_metric"])
    if main_metric not in metrics:
        raise ValueError(
            f"Configured main metric '{main_metric}' is not one of: {', '.join(metrics)}"
        )
    minimum_accuracy = float(params["evaluate"]["min_validation_accuracy"])
    if metrics["validation_accuracy"] < minimum_accuracy:
        print(
            "Warning: validation accuracy "
            f"{metrics['validation_accuracy']:.4f} is below the configured minimum "
            f"{minimum_accuracy:.4f}."
        )
    print(f"Saved fresh validation metrics to {args.metrics}")
    print(f"Main metric ({main_metric}): {metrics[main_metric]:.4f}")
    print(f"Saved confusion matrix to {args.confusion_matrix}")
    print(f"Saved classification report to {args.classification_report}")


if __name__ == "__main__":
    main()
