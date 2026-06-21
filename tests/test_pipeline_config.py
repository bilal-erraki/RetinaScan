from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dvc_pipeline_is_prepare_train_evaluate():
    pipeline = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]

    assert list(stages) == ["prepare", "train", "evaluate"]
    assert stages["prepare"]["cmd"] == "python src/data_loader.py"
    assert stages["train"]["cmd"] == "python src/train.py"
    assert "models/best_model.keras" in stages["train"]["outs"]
    assert "requirements.txt" in stages["train"]["deps"]
    assert stages["evaluate"]["cmd"] == "python src/evaluate.py"
    assert "models/best_model.keras" in stages["evaluate"]["deps"]
    assert "reports/metrics.json" in stages["evaluate"]["metrics"][0]
