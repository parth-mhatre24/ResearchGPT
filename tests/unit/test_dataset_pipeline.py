"""Unit tests for dataset acquisition utility and DatasetService."""

import json
from pathlib import Path
import pandas as pd
import pytest
from datasets import Dataset

from backend.app.services.dataset_service import DatasetService
from scripts.download_datasets import DATASET_CONFIGS, split_single_dataset


def test_dataset_configs_contain_all_five_benchmarks():
    """Verify all 5 datasets from PRD are configured."""
    expected_keys = {"imdb", "sms_spam", "sst2", "conll2003", "stsb"}
    assert expected_keys.issubset(set(DATASET_CONFIGS.keys()))

    assert DATASET_CONFIGS["imdb"]["task"] == "Binary sentiment classification"
    assert DATASET_CONFIGS["sms_spam"]["task"] == "Binary spam classification"
    assert DATASET_CONFIGS["sst2"]["task"] == "Fine-grained/binary sentiment"
    assert DATASET_CONFIGS["conll2003"]["task"] == "Named Entity Recognition"
    assert DATASET_CONFIGS["stsb"]["task"] == "Semantic textual similarity"


def test_split_single_dataset():
    """Test splitting single-split datasets with stratification."""
    data = {
        "text": [f"sample {i}" for i in range(100)],
        "label": [0] * 50 + [1] * 50,
    }
    ds = Dataset.from_dict(data)
    splits = split_single_dataset(
        ds,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        stratify_by_column="label",
    )

    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 10
    assert len(splits["test"]) == 10

    # Ensure label balance is maintained in test split
    test_labels = splits["test"]["label"]
    assert test_labels.count(0) == 5
    assert test_labels.count(1) == 5


def test_dataset_service_with_temp_dir(tmp_path: Path):
    """Test DatasetService loading and queries against mock datasets."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Create mock IMDB dataset
    imdb_dir = raw_dir / "imdb"
    imdb_dir.mkdir()

    train_df = pd.DataFrame({
        "text": ["Great paper!", "Poor methodology."],
        "label": [1, 0],
    })
    train_df.to_parquet(imdb_dir / "train.parquet")

    meta = {
        "dataset_key": "imdb",
        "name": "IMDB Movie Reviews",
        "task": "Binary sentiment classification",
        "splits": {
            "train": {"num_rows": 2, "columns": ["text", "label"]},
        },
    }
    with open(imdb_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    manifest = {
        "datasets": {
            "imdb": meta,
        }
    }
    with open(raw_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    service = DatasetService(data_dir=raw_dir)

    # Test available datasets
    available = service.get_available_datasets()
    assert available == ["imdb"]

    # Test metadata
    read_meta = service.get_dataset_metadata("imdb")
    assert read_meta["name"] == "IMDB Movie Reviews"

    # Test load split
    loaded_df = service.load_split("imdb", "train")
    assert len(loaded_df) == 2
    assert list(loaded_df["label"]) == [1, 0]

    # Test load classification data
    texts, labels = service.load_classification_data("imdb", "train")
    assert texts == ["Great paper!", "Poor methodology."]
    assert labels == [1, 0]


def test_dataset_service_error_handling(tmp_path: Path):
    """Test error conditions in DatasetService."""
    service = DatasetService(data_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        service.get_dataset_metadata("non_existent")

    with pytest.raises(FileNotFoundError):
        service.load_split("non_existent", "train")
