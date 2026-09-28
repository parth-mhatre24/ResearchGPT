"""ResearchGPT Classical Classification Training & Evaluation Script.

Trains TF-IDF baselines (MultinomialNB, LogisticRegression, LinearSVC)
on downloaded benchmark datasets (SMS Spam, IMDB, SST-2) and logs
reproducible experiment metrics and serialized model artifacts.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.classical_classifier_service import ClassicalClassifierService
from backend.app.services.dataset_service import DatasetService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_classification")

SUPPORTED_DATASETS = ["sms_spam", "imdb", "sst2"]
SUPPORTED_MODELS = ["naive_bayes", "logistic_regression", "svm"]


def train_and_eval(
    dataset_key: str,
    model_type: str,
    data_dir: Path,
    models_dir: Path,
    results_dir: Path,
    ngram_range: tuple = (1, 2),
    max_features: int = 20000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a single model on a dataset and log metrics."""
    logger.info("=" * 60)
    logger.info("Starting Experiment: [%s] on Dataset [%s]", model_type, dataset_key)

    service_ds = DatasetService(data_dir=data_dir)
    train_texts, train_labels = service_ds.load_classification_data(dataset_key, "train")
    test_texts, test_labels = service_ds.load_classification_data(dataset_key, "test")

    # For datasets with hidden GLUE test labels (e.g. SST-2 test labels are all -1),
    # use the standard labeled validation split for evaluation
    eval_split_name = "test"
    if all(l == -1 for l in test_labels):
        logger.info("Test split is unlabeled (-1, GLUE benchmark). Using 'validation' split for evaluation.")
        test_texts, test_labels = service_ds.load_classification_data(dataset_key, "validation")
        eval_split_name = "validation"

    logger.info(
        "Loaded %d train samples and %d eval samples (%s split)",
        len(train_texts),
        len(test_texts),
        eval_split_name,
    )

    # Initialize classifier
    clf_service = ClassicalClassifierService(
        model_type=model_type,
        ngram_range=ngram_range,
        max_features=max_features,
        random_state=seed,
    )

    # Train
    start_time = datetime.now(timezone.utc)
    clf_service.train(train_texts, train_labels)
    train_duration_sec = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Evaluate
    metrics = clf_service.evaluate(test_texts, test_labels)

    logger.info(
        "Results: Accuracy=%.4f | F1 (macro)=%.4f | F1 (weighted)=%.4f",
        metrics.accuracy,
        metrics.f1_macro,
        metrics.f1_weighted,
    )

    # Save model artifact
    models_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{dataset_key}_{model_type}"
    extra_meta = {
        "dataset": dataset_key,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "train_duration_sec": train_duration_sec,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    saved_model_path = clf_service.save(
        output_dir=models_dir,
        model_name=model_name,
        extra_metadata=extra_meta,
    )

    # Save experiment result
    results_dir.mkdir(parents=True, exist_ok=True)
    result_record = {
        "experiment_id": f"{dataset_key}_{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "dataset": dataset_key,
        "model_type": model_type,
        "ngram_range": list(ngram_range),
        "max_features": max_features,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "train_duration_sec": train_duration_sec,
        "model_path": str(saved_model_path),
        "metrics": metrics.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    result_file = results_dir / f"{model_name}_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)

    logger.info("Experiment results saved to: %s", result_file)
    return result_record


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF classical classification baselines.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_DATASETS,
        help="Dataset to train on (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_MODELS,
        help="Classifier architecture (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory (default: data/raw)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/saved/classical",
        help="Directory to save trained model artifacts (default: models/saved/classical)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/classification",
        help="Directory to save evaluation reports (default: experiments/results/classification)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Max TF-IDF features (default: 20000)",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Min n-gram size (default: 1)",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Max n-gram size (default: 2)",
    )

    args = parser.parse_args()

    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    models = SUPPORTED_MODELS if args.model == "all" else [args.model]

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    ngram_range = (args.ngram_min, args.ngram_max)

    all_results: List[Dict[str, Any]] = []

    for ds in datasets:
        for m in models:
            res = train_and_eval(
                dataset_key=ds,
                model_type=m,
                data_dir=data_dir,
                models_dir=models_dir,
                results_dir=results_dir,
                ngram_range=ngram_range,
                max_features=args.max_features,
            )
            all_results.append(res)

    print("\n" + "=" * 80)
    print(f"{'DATASET':<12} | {'MODEL':<20} | {'ACCURACY':<10} | {'F1 MACRO':<10} | {'F1 WEIGHTED':<12}")
    print("-" * 80)
    for r in all_results:
        m = r["metrics"]
        print(
            f"{r['dataset']:<12} | {r['model_type']:<20} | {m['accuracy']:<10.4f} | "
            f"{m['f1_macro']:<10.4f} | {m['f1_weighted']:<12.4f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
