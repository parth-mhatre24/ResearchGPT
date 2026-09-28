"""ResearchGPT Dataset Service.

Provides a standardized programmatic interface for loading, validating,
and accessing locally downloaded benchmark datasets for classification,
NER, and semantic similarity tasks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class DatasetService:
    """Service to load and manage local benchmark datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to repo root / data / raw
            self.data_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
        else:
            self.data_dir = Path(data_dir)

    def get_manifest(self) -> Dict[str, Any]:
        """Read and return the dataset manifest."""
        manifest_file = self.data_dir / "manifest.json"
        if not manifest_file.exists():
            return {}
        with open(manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_available_datasets(self) -> List[str]:
        """Return a list of dataset keys available locally in data_dir."""
        if not self.data_dir.exists():
            return []
        available = []
        for child in self.data_dir.iterdir():
            if child.is_dir() and (child / "meta.json").exists():
                available.append(child.name)
        return sorted(available)

    def get_dataset_metadata(self, dataset_key: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific dataset."""
        meta_file = self.data_dir / dataset_key / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No metadata found for dataset '{dataset_key}' at {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_split(self, dataset_key: str, split: str) -> pd.DataFrame:
        """Load a specific split as a pandas DataFrame."""
        dataset_folder = self.data_dir / dataset_key
        if not dataset_folder.exists():
            raise FileNotFoundError(f"Dataset directory '{dataset_key}' does not exist in {self.data_dir}")

        parquet_path = dataset_folder / f"{split}.parquet"
        csv_path = dataset_folder / f"{split}.csv"
        json_path = dataset_folder / f"{split}.json"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        elif json_path.exists():
            return pd.read_json(json_path)
        else:
            raise FileNotFoundError(f"Split '{split}' not found for dataset '{dataset_key}' in {dataset_folder}")

    def load_classification_data(
        self,
        dataset_key: str,
        split: str,
    ) -> Tuple[List[str], List[Any]]:
        """Extract texts and labels for classification tasks."""
        df = self.load_split(dataset_key, split)

        # Mapping for text and label columns across supported datasets
        text_cols = {
            "imdb": "text",
            "sms_spam": "sms",
            "sst2": "sentence",
        }
        label_cols = {
            "imdb": "label",
            "sms_spam": "label",
            "sst2": "label",
        }

        if dataset_key not in text_cols:
            raise ValueError(f"Dataset '{dataset_key}' is not registered as a standard text classification dataset")

        text_col = text_cols[dataset_key]
        label_col = label_cols[dataset_key]

        if text_col not in df.columns or label_col not in df.columns:
            raise KeyError(f"Expected columns '{text_col}' and '{label_col}' in dataset '{dataset_key}' (found: {list(df.columns)})")

        # Drop any nulls
        clean_df = df.dropna(subset=[text_col, label_col])
        texts = clean_df[text_col].astype(str).tolist()
        labels = clean_df[label_col].tolist()

        return texts, labels
