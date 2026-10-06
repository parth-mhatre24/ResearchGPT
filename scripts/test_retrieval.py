"""CLI Verification script for Vector Indexing and Semantic Retrieval (Task 15 / FR-08).

Demonstrates indexing landmark AI research papers and executing semantic queries.

Usage:
    python scripts/test_retrieval.py
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.retrieval import RetrievalQuery
from backend.app.services.faiss_retrieval_service import VectorRetrievalService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_retrieval")

SAMPLE_PAPERS = [
    {
        "doc_id": "paper_attention_2017",
        "title": "Attention Is All You Need (Vaswani et al., 2017)",
        "content": (
            "1. Introduction\n"
            "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, "
            "have been firmly established as state of the art approaches in sequence modeling and transduction problems "
            "such as language modeling and machine translation. "
            "The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism "
            "to draw global dependencies between input and output. The Transformer allows for significantly more parallelization "
            "and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.\n\n"
            "2. Scaled Dot-Product Attention\n"
            "We call our particular attention Scaled Dot-Product Attention. The input consists of queries and keys of dimension d_k, "
            "and values of dimension d_v. We compute the dot products of the query with all keys, divide each by sqrt(d_k), "
            "and apply a softmax function to obtain the weights on the values."
        ),
    },
    {
        "doc_id": "paper_bert_2018",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "content": (
            "1. Introduction\n"
            "Language model pre-training has been shown to be effective for improving many natural language processing tasks. "
            "There are two existing strategies for applying pre-trained language representations to downstream tasks: "
            "feature-based and fine-tuning. In this paper, we improve the fine-tuning based approaches by proposing BERT: "
            "Bidirectional Encoder Representations from Transformers.\n\n"
            "2. Masked Language Model\n"
            "In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, "
            "and then predict those masked tokens. We refer to this procedure as a Masked Language Model (MLM)."
        ),
    },
]


def main():
    logger.info("Initializing Vector Retrieval Service...")
    retrieval_service = VectorRetrievalService()

    # Index sample research papers
    for paper in SAMPLE_PAPERS:
        logger.info(f"Chunking and indexing [{paper['title']}]...")
        count = retrieval_service.add_document(
            document_id=paper["doc_id"],
            text=paper["content"],
            chunk_size=50,
            chunk_overlap=10,
        )
        logger.info(f"Indexed {count} chunks for {paper['doc_id']}.")

    info = retrieval_service.get_index_info()
    logger.info(f"Total indexed vectors: {info.total_vectors} across {info.unique_documents} documents.")

    # Execute test research queries
    test_queries = [
        "How does Scaled Dot-Product Attention compute weights between query and key vectors?",
        "What pre-training technique does BERT use to learn bidirectional representations?",
        "Why is the Transformer more parallelizable than recurrent neural networks?",
    ]

    print("\n" + "=" * 80)
    print("           SEMANTIC RETRIEVAL VERIFICATION (Task 15 / FR-08)")
    print("=" * 80)

    for q in test_queries:
        print(f"\n[QUERY]: \"{q}\"")
        resp = retrieval_service.search(query=q, top_k=2)
        print(f"[LATENCY]: {resp.retrieval_latency_ms} ms | Chunks Retrieved: {resp.total_retrieved}")
        for chunk in resp.results:
            print(f"   [Rank #{chunk.rank}] (Score: {chunk.similarity_score:.4f} | Doc: {chunk.metadata.document_id})")
            print(f"   Snippet: \"{chunk.text[:120]}...\"\n")

    # Verify saving and loading
    save_path = Path("models/saved/retrieval_index")
    retrieval_service.save_index(save_path)
    logger.info(f"Index successfully saved to {save_path.resolve()}")


if __name__ == "__main__":
    main()
