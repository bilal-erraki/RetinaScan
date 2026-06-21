import pandas as pd
import pytest

from src import data_loader
from src.data_loader import create_stratified_split, load_labels


def _labels_frame(samples_per_class=20):
    return pd.DataFrame(
        {
            "id_code": [
                f"image_{label}_{index}" for label in range(5) for index in range(samples_per_class)
            ],
            "diagnosis": [label for label in range(5) for _ in range(samples_per_class)],
        }
    )


def test_load_labels_accepts_expected_classes(tmp_path):
    csv_path = tmp_path / "train.csv"
    _labels_frame().to_csv(csv_path, index=False)
    loaded = load_labels(csv_path)
    assert set(loaded["diagnosis"]) == {0, 1, 2, 3, 4}


def test_load_labels_rejects_unexpected_class(tmp_path):
    csv_path = tmp_path / "train.csv"
    frame = _labels_frame()
    frame.loc[0, "diagnosis"] = 5
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Unexpected diagnosis"):
        load_labels(csv_path)


def test_create_stratified_split_preserves_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    csv_path = dataset_dir / "train.csv"
    _labels_frame().to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"

    train_path, validation_path = create_stratified_split(
        csv_path=csv_path,
        image_dir=dataset_dir / "train_images",
        output_dir=output_dir,
    )
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    assert train_path.is_file() and validation_path.is_file()
    assert len(train) == 80 and len(validation) == 20
    assert set(train["diagnosis"]) == set(validation["diagnosis"]) == {0, 1, 2, 3, 4}
    assert "image_path" in train.columns
    assert not train["image_path"].str.contains(str(tmp_path), regex=False).any()
