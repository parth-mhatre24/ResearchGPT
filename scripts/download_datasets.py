"""ResearchGPT Dataset Downloader and Acquisition Utility.

Downloads, standardizes, verifies, and caches benchmark datasets:
1. IMDB Movie Reviews (Binary sentiment classification)
2. SMS Spam Collection (Binary spam classification)
3. SST-2 (Fine-grained/binary sentiment classification)
4. CoNLL-2003 (Named Entity Recognition)
5. STS Benchmark / STS-B (Semantic textual similarity)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
try:
    from datasets import Dataset, DatasetDict, load_dataset
    _DATASETS_AVAILABLE = True
except Exception as _e:
    _DATASETS_AVAILABLE = False
    _DATASETS_IMPORT_ERROR = str(_e)
    Dataset = Any  # type: ignore[misc]
    DatasetDict = Any  # type: ignore[misc]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("download_datasets")

DATASET_CONFIGS = {
    "imdb": {
        "name": "IMDB Movie Reviews",
        "task": "Binary sentiment classification",
        "hub_id": "stanfordnlp/imdb",
        "config": None,
        "splits": ["train", "test"],
        "text_col": "text",
        "label_col": "label",
        "needs_split": False,
    },
    "sms_spam": {
        "name": "SMS Spam Collection",
        "task": "Binary spam classification",
        "hub_id": "ucirvine/sms_spam",
        "config": None,
        "splits": ["train"],
        "text_col": "sms",
        "label_col": "label",
        "needs_split": True,  # ucirvine/sms_spam comes as a single train split
    },
    "sst2": {
        "name": "SST-2 (Stanford Sentiment Treebank)",
        "task": "Fine-grained/binary sentiment",
        "hub_id": "stanfordnlp/sst2",
        "config": None,
        "splits": ["train", "validation", "test"],
        "text_col": "sentence",
        "label_col": "label",
        "needs_split": False,
    },
    "conll2003": {
        "name": "CoNLL-2003",
        "task": "Named Entity Recognition",
        "hub_id": "lhoestq/conll2003",
        "config": None,
        "splits": ["train", "validation", "test"],
        "text_col": "tokens",
        "label_col": "ner_tags",
        "needs_split": False,
    },
    "stsb": {
        "name": "STS Benchmark (STS-B)",
        "task": "Semantic textual similarity",
        "hub_id": "nyu-mll/glue",
        "config": "stsb",
        "splits": ["train", "validation", "test"],
        "text_col": "sentence1",
        "label_col": "label",
        "needs_split": False,
    },
}


def split_single_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_column: Optional[str] = None,
) -> DatasetDict:
    """Split a single-split dataset into train, validation, and test splits."""
    from datasets import ClassLabel

    if stratify_by_column and stratify_by_column in dataset.column_names:
        if not isinstance(dataset.features[stratify_by_column], ClassLabel):
            try:
                dataset = dataset.class_encode_column(stratify_by_column)
            except Exception as e:
                logger.warning("Could not class_encode_column '%s': %s. Falling back to non-stratified split.", stratify_by_column, e)
                stratify_by_column = None

    if stratify_by_column and stratify_by_column in dataset.column_names:
        # First split into train + temp (val+test)
        split_1 = dataset.train_test_split(
            test_size=(1.0 - train_ratio),
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
        train_data = split_1["train"]
        temp_data = split_1["test"]

        # Next split temp into validation and test equally
        val_share = val_ratio / (1.0 - train_ratio)
        split_2 = temp_data.train_test_split(
            test_size=(1.0 - val_share),
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
        val_data = split_2["train"]
        test_data = split_2["test"]
    else:
        split_1 = dataset.train_test_split(test_size=(1.0 - train_ratio), seed=seed)
        train_data = split_1["train"]
        temp_data = split_1["test"]
        val_share = val_ratio / (1.0 - train_ratio)
        split_2 = temp_data.train_test_split(test_size=(1.0 - val_share), seed=seed)
        val_data = split_2["train"]
        test_data = split_2["test"]

    return DatasetDict({
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    })


def download_single_dataset(
    dataset_key: str,
    output_dir: Path,
    sample_size: Optional[int] = None,
    save_format: str = "parquet",
    seed: int = 42,
) -> Dict[str, Any]:
    """Download, process, verify, and save a single dataset."""
    cfg = DATASET_CONFIGS[dataset_key]
    logger.info("=" * 60)
    logger.info("Downloading [%s] (%s)", cfg["name"], cfg["task"])
    logger.info("Hub ID: %s | Config: %s", cfg["hub_id"], cfg["config"])

    load_kwargs = {}
    if cfg["config"]:
        raw_data = load_dataset(cfg["hub_id"], cfg["config"], **load_kwargs)
    else:
        raw_data = load_dataset(cfg["hub_id"], **load_kwargs)

    dataset_dict: DatasetDict
    if cfg["needs_split"]:
        logger.info("Splitting single '%s' split into train (80%%), validation (10%%), test (10%%)...", cfg["splits"][0])
        single_split = raw_data[cfg["splits"][0]]
        dataset_dict = split_single_dataset(
            single_split,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed,
            stratify_by_column=cfg.get("label_col"),
        )
    else:
        dataset_dict = DatasetDict({split: raw_data[split] for split in cfg["splits"] if split in raw_data})

    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "name": cfg["name"],
        "task": cfg["task"],
        "hub_id": cfg["hub_id"],
        "config": cfg["config"],
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    for split_name, split_data in dataset_dict.items():
        if sample_size and len(split_data) > sample_size:
            split_data = split_data.select(range(sample_size))
            logger.info("Sampled [%s] to %d items", split_name, len(split_data))

        # Save to file
        file_path = dataset_dir / f"{split_name}.{save_format}"
        if save_format == "parquet":
            split_data.to_parquet(str(file_path))
        elif save_format == "csv":
            split_data.to_csv(str(file_path))
        elif save_format == "json":
            split_data.to_json(str(file_path))
        else:
            raise ValueError(f"Unsupported format: {save_format}")

        metadata["splits"][split_name] = {
            "num_rows": len(split_data),
            "columns": split_data.column_names,
            "file": str(file_path.name),
        }
        logger.info("Saved %s [%s]: %d rows -> %s", dataset_key, split_name, len(split_data), file_path)

    # Save dataset-level metadata
    with open(dataset_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def download_all(
    dataset_keys: list,
    output_dir: Path,
    sample_size: Optional[int] = None,
    save_format: str = "parquet",
    seed: int = 42,
) -> Dict[str, Any]:
    """Download multiple datasets and generate a central manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "format": save_format,
        "sample_size": sample_size,
        "datasets": {},
    }

    for key in dataset_keys:
        meta = download_single_dataset(
            dataset_key=key,
            output_dir=output_dir,
            sample_size=sample_size,
            save_format=save_format,
            seed=seed,
        )
        manifest["datasets"][key] = meta

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("Manifest created at: %s", manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ResearchGPT benchmark datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "imdb", "sms_spam", "sst2", "conll2003", "stsb"],
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Target directory for downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "csv", "json"],
        help="Storage file format (default: parquet)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional max sample size per split for testing/quick verification",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    target_keys = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    logger.info("Starting acquisition of datasets: %s", target_keys)
    manifest = download_all(
        dataset_keys=target_keys,
        output_dir=output_dir,
        sample_size=args.sample_size,
        save_format=args.format,
        seed=args.seed,
    )

    print("\n" + "=" * 65)
    print("DATASET ACQUISITION SUMMARY")
    print("=" * 65)
    for key, data in manifest["datasets"].items():
        print(f"\n* {data['name']} ({key})")
        print(f"  Task: {data['task']}")
        print(f"  Source: {data['hub_id']}")
        for split, info in data["splits"].items():
            print(f"    - {split}: {info['num_rows']:,} rows | Columns: {info['columns']}")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
