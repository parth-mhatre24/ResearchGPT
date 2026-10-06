"""CLI Evaluation script for Abstractive Summarization (Task 12).

Usage:
    python scripts/evaluate_summarization.py [--model-name t5-small] [--num-beams 4]
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

from backend.app.services.summarization_service import SummarizationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_summarization")

# Benchmark sample research paper texts and reference abstracts for evaluation
SAMPLE_RESEARCH_DOCS = [
    {
        "title": "Attention Is All You Need",
        "text": (
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
            "that include an encoder and a decoder. The best performing models also connect the encoder and decoder "
            "through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely "
            "on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation "
            "tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
        ),
        "reference": (
            "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions to achieve superior translation quality and faster training."
        ),
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "text": (
            "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations "
            "from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional "
            "representations from unlabeled text by jointly conditioning on both left and right context in all layers. "
            "As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create "
            "state-of-the-art models for a wide range of tasks, such as question answering and language inference."
        ),
        "reference": (
            "BERT introduces deep bidirectional transformer pre-training from unlabeled text, enabling state-of-the-art "
            "fine-tuning across diverse natural language processing tasks."
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate abstractive document summarization model.")
    parser.add_argument("--model-name", type=str, default="t5-small", help="Transformer checkpoint (e.g. t5-small, facebook/bart-base)")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam search width")
    args = parser.parse_args()

    logger.info(f"Initializing SummarizationService with model: {args.model_name}")
    service = SummarizationService(model_name=args.model_name)

    references = [doc["reference"] for doc in SAMPLE_RESEARCH_DOCS]
    generated_summaries = []

    print("\n" + "=" * 70)
    print(f"        SUMMARIZATION MODEL EVALUATION ({args.model_name})")
    print("=" * 70)

    for doc in SAMPLE_RESEARCH_DOCS:
        response = service.summarize(doc["text"], num_beams=args.num_beams)
        generated_summaries.append(response.summary)
        print(f"\nDocument Title: {doc['title']}")
        print(f"Input Words:    {response.input_length_words}")
        print(f"Summary Words:  {response.summary_length_words} (Compression: {response.compression_ratio:.2%})")
        print(f"Generated Summary:\n  {response.summary}")
        print("-" * 70)

    logger.info("Computing ROUGE evaluation metrics...")
    metrics = service.evaluate(references, generated_summaries)

    print("\n" + "=" * 70)
    print("                  ROUGE EVALUATION RESULTS")
    print("=" * 70)
    print(f"ROUGE-1 F1: {metrics.rouge1 * 100:.2f}%")
    print(f"ROUGE-2 F1: {metrics.rouge2 * 100:.2f}%")
    print(f"ROUGE-L F1: {metrics.rougeL * 100:.2f}%")
    print("-" * 70)

    # Save artifacts
    results_dir = PROJECT_ROOT / "experiments" / "results" / "summarization"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "summarization_results.json"

    result_payload = {
        "model_name": args.model_name,
        "num_beams": args.num_beams,
        "metrics": metrics.model_dump(),
        "evaluations": [
            {
                "title": doc["title"],
                "reference": doc["reference"],
                "generated": gen,
            }
            for doc, gen in zip(SAMPLE_RESEARCH_DOCS, generated_summaries)
        ],
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    logger.info(f"Saved evaluation results to {results_file}")


if __name__ == "__main__":
    main()
