#!/usr/bin/env python
"""Task 09 – CRF NER Training Script.

Trains a feature-rich sklearn-crfsuite CRF on the CoNLL-2003 dataset,
evaluates on the test split, persists the model, and writes benchmark metrics.

Usage:
    python scripts/train_crf_ner.py
    python scripts/train_crf_ner.py --max-iter 300 --c1 0.05 --c2 0.05
    python scripts/train_crf_ner.py --limit 2000   # quick smoke-test on 2000 sentences
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
logger = logging.getLogger("train_crf_ner")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.crf_ner_service import CRFNERService
from backend.app.services.dataset_service import DatasetService


def main():
    parser = argparse.ArgumentParser(description="Train a CRF NER baseline on CoNLL-2003.")
    parser.add_argument("--dataset", default="conll2003", help="Dataset key (default: conll2003)")
    parser.add_argument("--max-iter", type=int, default=200, help="CRF max iterations (default: 200)")
    parser.add_argument("--c1", type=float, default=0.1, help="L1 regularisation (default: 0.1)")
    parser.add_argument("--c2", type=float, default=0.1, help="L2 regularisation (default: 0.1)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of training sentences for quick smoke-tests.",
    )
    args = parser.parse_args()

    results_dir = Path("experiments/results/ner")
    models_dir = Path("models/saved/ner")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    ds = DatasetService()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading CoNLL-2003 train split ...")
    train_tokens, train_labels = ds.load_ner_data(args.dataset, "train")
    if args.limit:
        train_tokens = train_tokens[: args.limit]
        train_labels = train_labels[: args.limit]
    logger.info("Training sentences: %d", len(train_tokens))

    logger.info("Loading CoNLL-2003 validation split ...")
    val_tokens, val_labels = ds.load_ner_data(args.dataset, "validation")
    logger.info("Validation sentences: %d", len(val_tokens))

    logger.info("Loading CoNLL-2003 test split ...")
    test_tokens, test_labels = ds.load_ner_data(args.dataset, "test")
    logger.info("Test sentences: %d", len(test_tokens))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    service = CRFNERService(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter,
        all_possible_transitions=True,
    )

    logger.info(
        "Training CRF [c1=%.3f, c2=%.3f, max_iter=%d] ...",
        args.c1, args.c2, args.max_iter,
    )
    t0 = time.perf_counter()
    service.train(train_tokens, train_labels)
    train_time = time.perf_counter() - t0
    logger.info("CRF trained in %.1fs", train_time)

    # ------------------------------------------------------------------
    # Evaluate on validation and test splits
    # ------------------------------------------------------------------
    logger.info("Evaluating on validation split (%d sentences) ...", len(val_tokens))
    val_metrics = service.evaluate(val_tokens, val_labels)
    logger.info(
        "Validation — F1_micro=%.4f  F1_macro=%.4f  TokenAcc=%.4f",
        val_metrics.f1_micro, val_metrics.f1_macro, val_metrics.token_accuracy,
    )

    logger.info("Evaluating on test split (%d sentences) ...", len(test_tokens))
    test_metrics = service.evaluate(test_tokens, test_labels)
    logger.info(
        "Test — F1_micro=%.4f  F1_macro=%.4f  TokenAcc=%.4f",
        test_metrics.f1_micro, test_metrics.f1_macro, test_metrics.token_accuracy,
    )

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_file = service.save(
        models_dir,
        model_name="crf_conll2003",
        extra_metadata={
            "c1": args.c1,
            "c2": args.c2,
            "max_iterations": args.max_iter,
            "training_sentences": len(train_tokens),
            "training_time_s": round(train_time, 2),
        },
    )
    logger.info("Model saved to %s", model_file)

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    result = {
        "dataset": args.dataset,
        "model": "CRF (sklearn-crfsuite, lbfgs)",
        "hyperparameters": {"c1": args.c1, "c2": args.c2, "max_iterations": args.max_iter},
        "training_sentences": len(train_tokens),
        "training_time_s": round(train_time, 2),
        "validation": {
            "sentences": val_metrics.sample_count,
            "precision_micro": val_metrics.precision_micro,
            "recall_micro": val_metrics.recall_micro,
            "f1_micro": val_metrics.f1_micro,
            "f1_macro": val_metrics.f1_macro,
            "token_accuracy": val_metrics.token_accuracy,
            "per_entity": [e.model_dump() for e in val_metrics.per_entity_metrics],
        },
        "test": {
            "sentences": test_metrics.sample_count,
            "precision_micro": test_metrics.precision_micro,
            "recall_micro": test_metrics.recall_micro,
            "f1_micro": test_metrics.f1_micro,
            "f1_macro": test_metrics.f1_macro,
            "token_accuracy": test_metrics.token_accuracy,
            "per_entity": [e.model_dump() for e in test_metrics.per_entity_metrics],
        },
    }

    result_file = results_dir / "crf_conll2003_results.json"
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Results written to %s", result_file)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n## CRF NER Benchmark Results (CoNLL-2003)\n")
    print("| Split | Sentences | P (micro) | R (micro) | F1 (micro) | F1 (macro) | Token Acc |")
    print("|---|---|---|---|---|---|---|")
    for split_name, m in [("Validation", val_metrics), ("Test", test_metrics)]:
        print(
            f"| **{split_name}** | {m.sample_count} "
            f"| {m.precision_micro:.4f} | {m.recall_micro:.4f} "
            f"| {m.f1_micro:.4f} | {m.f1_macro:.4f} | {m.token_accuracy:.4f} |"
        )

    print("\n### Per-entity F1 (Test split)\n")
    print("| Entity Type | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for e in test_metrics.per_entity_metrics:
        print(f"| {e.entity_type} | {e.precision:.4f} | {e.recall:.4f} | {e.f1:.4f} | {e.support} |")


if __name__ == "__main__":
    main()
