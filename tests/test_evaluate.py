import json

from src.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    save_classification_report,
    save_metrics,
)


def test_evaluation_artifacts_are_computed_and_saved(tmp_path):
    truth = [0, 1, 2, 3, 4]
    predictions = [0, 1, 2, 2, 4]
    metrics = compute_classification_metrics(truth, predictions)
    metrics["validation_loss"] = 0.5

    metrics_path = tmp_path / "metrics.json"
    confusion_path = tmp_path / "confusion_matrix.png"
    report_path = tmp_path / "classification_report.json"
    save_metrics(metrics, metrics_path)
    plot_confusion_matrix(truth, predictions, confusion_path)
    save_classification_report(truth, predictions, report_path)

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["validation_accuracy"] == 0.8
    assert saved_metrics["validation_loss"] == 0.5
    assert confusion_path.stat().st_size > 0
    assert "Moderate" in json.loads(report_path.read_text(encoding="utf-8"))
