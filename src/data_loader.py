"""Load APTOS labels and create reproducible stratified data splits."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_DIR,
    LEGACY_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    ensure_directories,
)
from src.preprocessing import preprocess_image_from_path

EXPECTED_LABELS = {0, 1, 2, 3, 4}


def _find_dataset_dir() -> Path:
    for path in (DATASET_DIR, LEGACY_DATA_DIR):
        if (path / "train.csv").is_file():
            return path
    raise FileNotFoundError(
        "APTOS train.csv was not found. Place the dataset under "
        f"'{DATASET_DIR}' or the legacy path '{LEGACY_DATA_DIR}'."
    )


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """Load APTOS train.csv labels and validate the required columns/classes."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels CSV was not found: {path}")

    labels = pd.read_csv(path)
    required_columns = {"id_code", "diagnosis"}
    missing = required_columns.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing columns: {', '.join(sorted(missing))}")
    if labels.empty:
        raise ValueError("Labels CSV contains no rows.")
    if labels[["id_code", "diagnosis"]].isnull().any().any():
        raise ValueError("Labels CSV contains missing image IDs or diagnoses.")

    try:
        labels["diagnosis"] = labels["diagnosis"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Diagnosis labels must be integers from 0 through 4.") from exc

    found_labels = set(labels["diagnosis"].unique())
    if not found_labels.issubset(EXPECTED_LABELS):
        raise ValueError(f"Unexpected diagnosis labels: {sorted(found_labels - EXPECTED_LABELS)}")
    return labels


def create_stratified_split(
    csv_path: str | Path | None = None,
    image_dir: str | Path | None = None,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[Path, Path]:
    """Create and save an 80/20 stratified train/validation split."""

    dataset_dir = Path(csv_path).parent if csv_path else _find_dataset_dir()
    labels = load_labels(csv_path or dataset_dir / "train.csv")
    images = Path(image_dir) if image_dir else dataset_dir / "train_images"

    labels = labels.copy()

    def portable_image_path(image_id: str) -> str:
        image_path = (images / f"{image_id}.png").resolve()
        try:
            return image_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            # Custom datasets outside the project remain supported, although a path
            # inside the repository is preferable for cross-machine reproducibility.
            return str(image_path)

    labels["image_path"] = labels["id_code"].map(portable_image_path)
    train_data, validation_data = train_test_split(
        labels,
        test_size=val_size,
        random_state=random_state,
        stratify=labels["diagnosis"],
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train_split.csv"
    validation_path = destination / "val_split.csv"
    train_data.reset_index(drop=True).to_csv(train_path, index=False)
    validation_data.reset_index(drop=True).to_csv(validation_path, index=False)
    return train_path, validation_path


def resolve_image_paths(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve portable split paths against the project root."""

    if "image_path" not in frame.columns:
        raise ValueError("Split CSV is missing the image_path column. Run the prepare stage again.")
    resolved = frame.copy()
    resolved["image_path"] = resolved["image_path"].map(
        lambda value: str(
            (PROJECT_ROOT / Path(str(value))).resolve()
            if not Path(str(value)).is_absolute()
            else Path(str(value)).resolve()
        )
    )
    return resolved


def make_tf_dataset(
    tf,
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool = False,
    random_state: int = RANDOM_STATE,
):
    """Build a deterministic TensorFlow dataset from a processed split."""

    frame = resolve_image_paths(frame)

    def load_with_pillow(path):
        decoded_path = path.numpy().decode("utf-8")
        return preprocess_image_from_path(decoded_path, image_size)[0]

    def load_sample(path, label):
        image = tf.py_function(load_with_pillow, [path], Tout=tf.float32)
        image.set_shape((image_size, image_size, 3))
        return image, tf.cast(label, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["image_path"].astype(str).values, frame["diagnosis"].astype("int32").values)
    )
    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            min(len(frame), 2048), seed=random_state, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    ensure_directories()
    params_path = PROJECT_ROOT / "params.yaml"
    data_params = yaml.safe_load(params_path.read_text(encoding="utf-8"))["data"]
    dataset_dir = PROJECT_ROOT / str(data_params["dataset_dir"])
    train_path, validation_path = create_stratified_split(
        csv_path=dataset_dir / "train.csv",
        image_dir=dataset_dir / "train_images",
        val_size=float(data_params["val_size"]),
        random_state=int(data_params["random_state"]),
    )
    print(f"Created training split: {train_path}")
    print(f"Created validation split: {validation_path}")


if __name__ == "__main__":
    main()
