"""CLI Benchmark & Evaluation script for Semantic Textual Similarity (Task 13 / FR-07).

Evaluates TF-IDF Baseline vs. Sentence-BERT on the STS Benchmark (STS-B).

Usage:
    python scripts/evaluate_similarity.py [--split validation] [--model-name sentence-transformers/all-MiniLM-L6-v2]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dataset_service import DatasetService
from backend.app.services.similarity_service import SemanticSimilarityService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_similarity")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Textual Similarity on STS-B.")
    parser.add_argument("--split", type=str, default="validation", help="STS-B dataset split (validation / test)")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence embedding model")
    args = parser.parse_args()

    logger.info(f"Loading STS-B ({args.split} split) via DatasetService...")
    ds_service = DatasetService()
    sentences_a, sentences_b, gold_scores = ds_service.load_similarity_data("stsb", split=args.split)
    logger.info(f"Loaded {len(sentences_a)} sentence pairs from STS-B {args.split} split.")

    sim_service = SemanticSimilarityService(model_name=args.model_name)

    logger.info("Evaluating TF-IDF Cosine Similarity baseline...")
    tfidf_metrics = sim_service.evaluate_stsb(sentences_a, sentences_b, gold_scores, method="tfidf")

    logger.info(f"Evaluating Sentence-BERT ({args.model_name})...")
    sbert_metrics = sim_service.evaluate_stsb(sentences_a, sentences_b, gold_scores, method="sentence_bert")

    print("\n" + "=" * 70)
    print(f"        STS-B BENCHMARK EVALUATION RESULTS ({args.split.upper()} SPLIT)")
    print("=" * 70)
    print(f"Sample Count:            {len(gold_scores)} sentence pairs")
    print("-" * 70)
    print("1. TF-IDF Cosine Baseline:")
    print(f"   - Pearson Correlation (r):   {tfidf_metrics.pearson_correlation:.4f}")
    print(f"   - Spearman Correlation (rho): {tfidf_metrics.spearman_correlation:.4f}")
    print(f"   - Mean Squared Error (MSE):  {tfidf_metrics.mse:.4f}")
    print("-" * 70)
    print(f"2. Sentence-BERT ({args.model_name}):")
    print(f"   - Pearson Correlation (r):   {sbert_metrics.pearson_correlation:.4f}")
    print(f"   - Spearman Correlation (rho): {sbert_metrics.spearman_correlation:.4f}")
    print(f"   - Mean Squared Error (MSE):  {sbert_metrics.mse:.4f}")
    print("=" * 70)

    # Save results
    output_dir = Path("experiments/results/similarity")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "stsb_results.json"

    result_payload = {
        "dataset": "stsb",
        "split": args.split,
        "sample_count": len(gold_scores),
        "tfidf_baseline": tfidf_metrics.model_dump(),
        "sentence_bert": sbert_metrics.model_dump(),
    }

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, indent=2)

    logger.info(f"Saved evaluation results to {out_file.resolve()}")


if __name__ == "__main__":
    main()
