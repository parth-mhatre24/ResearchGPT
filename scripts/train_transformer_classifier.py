#!/usr/bin/env python
"""Task 08 – Transformer Classification Training Script.

Fine-tunes DistilBERT on IMDB, SMS Spam, and SST-2 datasets and
writes persisted checkpoints + benchmark metrics to disk.

Usage:
    python scripts/train_transformer_classifier.py
    python scripts/train_transformer_classifier.py --datasets imdb sst2
    python scripts/train_transformer_classifier.py --model distilbert-base-uncased --epochs 3
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_transformer_classifier")

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.dataset_service import DatasetService
from backend.app.services.transformer_classifier_service import TransformerClassifierService


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "imdb": {
        "train_split": "train",
        "val_split": None,       # No validation split; use a 10% slice of test
        "test_split": "test",
        "description": "IMDB Movie Reviews (Binary Sentiment)",
    },
    "sms_spam": {
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "description": "SMS Spam Collection (Binary Spam)",
    },
    "sst2": {
        "train_split": "train",
        "val_split": "validation",
        "test_split": None,      # Test labels not available in SST-2 (GLUE)
        "description": "SST-2 Stanford Sentiment Treebank (Binary Sentiment)",
    },
}


def run_dataset(
    dataset_key: str,
    cfg: dict,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    max_length: int,
    results_dir: Path,
    models_dir: Path,
) -> dict:
    """Fine-tune and evaluate one dataset. Returns a metrics dict."""
    ds = DatasetService()
    logger.info("=" * 60)
    logger.info("Dataset: %s — %s", dataset_key, cfg["description"])
    logger.info("=" * 60)

    # Load training data
    train_texts, train_labels = ds.load_classification_data(dataset_key, cfg["train_split"])
    logger.info("Train samples: %d", len(train_texts))

    # Load validation data if available
    val_texts = val_labels = None
    if cfg.get("val_split"):
        val_texts, val_labels = ds.load_classification_data(dataset_key, cfg["val_split"])
        logger.info("Validation samples: %d", len(val_texts))

    # If no validation split, carve 5% off training data
    if val_texts is None:
        cutoff = max(1, len(train_texts) * 95 // 100)
        val_texts = train_texts[cutoff:]
        val_labels = train_labels[cutoff:]
        train_texts = train_texts[:cutoff]
        train_labels = train_labels[:cutoff]
        logger.info("No val split — using last %d samples as validation.", len(val_texts))

    # Train
    service = TransformerClassifierService(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        num_epochs=num_epochs,
        warmup_ratio=0.06,
        output_dir=str(models_dir / f"{dataset_key}_ckpts"),
    )

    t0 = time.perf_counter()
    service.train(train_texts, train_labels, val_texts, val_labels)
    train_time = time.perf_counter() - t0
    logger.info("Training time: %.1fs", train_time)

    # Evaluate on test split (or val as fallback)
    if cfg.get("test_split"):
        eval_texts, eval_labels = ds.load_classification_data(dataset_key, cfg["test_split"])
        eval_split_name = cfg["test_split"]
    else:
        eval_texts, eval_labels = val_texts, val_labels
        eval_split_name = "validation"
    logger.info("Evaluating on '%s' split (%d samples) ...", eval_split_name, len(eval_texts))

    metrics = service.evaluate(eval_texts, eval_labels)

    result = {
        "dataset": dataset_key,
        "description": cfg["description"],
        "model_name": model_name,
        "eval_split": eval_split_name,
        "eval_samples": len(eval_texts),
        "accuracy": round(metrics.accuracy, 4),
        "precision_macro": round(metrics.precision_macro, 4),
        "recall_macro": round(metrics.recall_macro, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "precision_weighted": round(metrics.precision_weighted, 4),
        "recall_weighted": round(metrics.recall_weighted, 4),
        "f1_weighted": round(metrics.f1_weighted, 4),
        "training_time_s": round(train_time, 2),
    }

    logger.info(
        "Results: Acc=%.4f  F1_macro=%.4f  F1_weighted=%.4f",
        result["accuracy"], result["f1_macro"], result["f1_weighted"],
    )

    # Persist model checkpoint
    save_path = service.save(
        str(models_dir),
        model_name=f"{dataset_key}_{model_name.replace('/', '_')}",
        extra_metadata=result,
    )
    logger.info("Model saved to %s", save_path)

    # Persist per-dataset JSON result
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{dataset_key}_transformer_results.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Result written to %s", out_file)

    return result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer classifiers on benchmark datasets.")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        help="Which datasets to train on (default: all three).",
    )
    parser.add_argument(
        "--model", default="distilbert-base-uncased",
        help="HuggingFace model hub ID (default: distilbert-base-uncased).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length.")
    args = parser.parse_args()

    results_dir = Path("experiments/results/transformer/classification")
    models_dir = Path("models/saved/transformer")
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds_key in args.datasets:
        cfg = DATASET_CONFIGS[ds_key]
        result = run_dataset(
            dataset_key=ds_key,
            cfg=cfg,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            results_dir=results_dir,
            models_dir=models_dir,
        )
        all_results.append(result)

    # Write consolidated summary
    summary_file = results_dir / "all_results_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("\nAll results written to %s", summary_file)

    # Print markdown table
    print("\n## Transformer Classification Benchmark Results\n")
    header = "| Dataset | Model | Split | Accuracy | F1 Macro | F1 Weighted | Train Time |"
    sep    = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in all_results:
        print(
            f"| **{r['dataset']}** | {r['model_name']} | {r['eval_split']} "
            f"({r['eval_samples']}) | {r['accuracy']:.4f} | {r['f1_macro']:.4f} "
            f"| {r['f1_weighted']:.4f} | {r['training_time_s']:.1f}s |"
        )


if __name__ == "__main__":
    main()
