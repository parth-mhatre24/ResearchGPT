"""CLI Training & Evaluation script for BiLSTM-CRF NER (Task 10).

Usage:
    python scripts/train_bilstm_crf_ner.py [--epochs 5] [--lr 0.001]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.bilstm_crf_ner_service import BiLSTMCRFNERService
from backend.app.services.dataset_service import DatasetService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_bilstm_crf_ner")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BiLSTM-CRF NER model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    logger.info("Loading CoNLL-2003 dataset via DatasetService...")
    ds_service = DatasetService()
    train_tokens, train_tags = ds_service.load_ner_data("conll2003", split="train")
    val_tokens, val_tags = ds_service.load_ner_data("conll2003", split="validation")
    test_tokens, test_tags = ds_service.load_ner_data("conll2003", split="test")

    logger.info(f"Loaded CoNLL-2003: train={len(train_tokens)}, val={len(val_tokens)}, test={len(test_tokens)} sentences.")

    service = BiLSTMCRFNERService(embedding_dim=100, hidden_dim=128)

    logger.info(f"Starting BiLSTM-CRF training for {args.epochs} epochs...")
    train_result = service.train(
        train_sentences=train_tokens,
        train_tags=train_tags,
        val_sentences=val_tokens,
        val_tags=val_tags,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    logger.info("Evaluating BiLSTM-CRF on CoNLL-2003 Test split...")
    test_metrics = service.evaluate(test_tokens, test_tags)

    print("\n" + "=" * 60)
    print("      BiLSTM-CRF NER EXPERIMENT RESULTS (CoNLL-2003 Test)")
    print("=" * 60)
    print(f"Micro F1 Score:  {test_metrics.f1_micro * 100:.2f}%")
    print(f"Micro Precision: {test_metrics.precision_micro * 100:.2f}%")
    print(f"Micro Recall:    {test_metrics.recall_micro * 100:.2f}%")
    print(f"Macro F1 Score:  {test_metrics.f1_macro * 100:.2f}%")
    print(f"Token Accuracy:  {test_metrics.token_accuracy * 100:.2f}%")
    print("-" * 60)

    # Save artifacts
    save_dir = PROJECT_ROOT / "models" / "saved" / "ner" / "bilstm_crf"
    service.save(save_dir)

    results_dir = PROJECT_ROOT / "experiments" / "results" / "ner"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "bilstm_crf_results.json"

    result_payload = {
        "model_type": "bilstm_crf",
        "dataset": "conll2003",
        "epochs": args.epochs,
        "lr": args.lr,
        "training_result": train_result,
        "test_metrics": test_metrics.model_dump(),
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    logger.info(f"Saved experiment results to {results_file}")


if __name__ == "__main__":
    main()
