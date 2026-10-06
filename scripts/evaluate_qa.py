"""CLI Evaluation script for Extractive Question Answering (Task 16 / FR-09).

Evaluates Transformer QA models on landmark scientific paper question-answering benchmarks.

Usage:
    python scripts/evaluate_qa.py [--model-name distilbert-base-cased-distilled-squad]
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

from backend.app.services.qa_service import QuestionAnsweringService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_qa")

SCIENTIFIC_QA_BENCHMARK = [
    {
        "id": "qa_01",
        "question": "What is the primary architecture of the Transformer model based on?",
        "context": (
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show "
            "these models to be superior in quality while being more parallelizable."
        ),
        "answers": ["attention mechanisms", "solely on attention mechanisms"],
    },
    {
        "id": "qa_02",
        "question": "What does BERT stand for?",
        "context": (
            "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations "
            "from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations."
        ),
        "answers": ["Bidirectional Encoder Representations from Transformers"],
    },
    {
        "id": "qa_03",
        "question": "What function is applied to obtain the weights on the values in Scaled Dot-Product Attention?",
        "context": (
            "We compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function "
            "to obtain the weights on the values."
        ),
        "answers": ["softmax", "a softmax function"],
    },
    {
        "id": "qa_04",
        "question": "What procedure does BERT use to train a deep bidirectional representation?",
        "context": (
            "In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, "
            "and then predict those masked tokens. We refer to this procedure as a Masked Language Model (MLM)."
        ),
        "answers": ["Masked Language Model (MLM)", "Masked Language Model", "mask some percentage of the input tokens at random"],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Question Answering model on scientific benchmarks.")
    parser.add_argument("--model-name", type=str, default="distilbert-base-cased-distilled-squad", help="QA transformer model")
    args = parser.parse_args()

    logger.info(f"Initializing QA Service with [{args.model_name}]...")
    qa_service = QuestionAnsweringService(model_name=args.model_name)

    logger.info(f"Evaluating on {len(SCIENTIFIC_QA_BENCHMARK)} scientific paper QA pairs...")
    metrics = qa_service.evaluate_qa(SCIENTIFIC_QA_BENCHMARK)

    print("\n" + "=" * 70)
    print(f"        QUESTION ANSWERING EVALUATION RESULTS ({args.model_name})")
    print("=" * 70)
    print(f"Sample Count:            {metrics.sample_count} QA pairs")
    print(f"Exact Match (EM):        {metrics.exact_match:.2f}%")
    print(f"Token F1 Score:          {metrics.token_f1:.2f}%")
    print(f"Token Precision:         {metrics.precision:.2f}%")
    print(f"Token Recall:            {metrics.recall:.2f}%")
    print("=" * 70)

    # Print sample predictions
    print("\nSample Predictions:")
    for item in SCIENTIFIC_QA_BENCHMARK:
        resp = qa_service.answer_question(item["question"], item["context"])
        print(f"\n[Q]: {item['question']}")
        print(f"[PREDICTED ANSWER]: \"{resp.answer}\" (Confidence: {resp.confidence_score:.4f})")
        print(f"[GOLD]: {item['answers']}")

    # Save results
    output_dir = Path("experiments/results/qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "qa_results.json"

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(metrics.model_dump(), fh, indent=2)

    logger.info(f"Saved QA evaluation results to {out_file.resolve()}")


if __name__ == "__main__":
    main()
