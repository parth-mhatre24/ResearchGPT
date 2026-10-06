"""CLI Verification script for End-to-End RAG Pipeline (Task 17 / FR-10).

Indexes sample research papers and tests grounded question answering with source citations.

Usage:
    python scripts/test_rag_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.rag import RAGQueryRequest
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.qa_service import QuestionAnsweringService
from backend.app.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_rag_pipeline")

RESEARCH_CORPUS = [
    {
        "doc_id": "vaswani_2017_transformer",
        "title": "Attention Is All You Need (Vaswani et al., 2017)",
        "text": (
            "1. Introduction\n"
            "The Transformer is the first transduction model relying entirely on self-attention to compute "
            "representations of its input and output without using sequence-aligned RNNs or convolution. "
            "In our experiments, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.\n\n"
            "2. Model Architecture\n"
            "The Transformer follows an encoder-decoder architecture using stacked self-attention and point-wise, fully connected layers. "
            "The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, "
            "and a simple, position-wise fully connected feed-forward network."
        ),
    },
    {
        "doc_id": "devlin_2018_bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "text": (
            "1. Introduction\n"
            "We introduce BERT: Bidirectional Encoder Representations from Transformers. BERT alleviates the unidirectionality constraint "
            "by using a Masked Language Model (MLM) pre-training objective. The masked language model randomly masks some of the tokens "
            "from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context.\n\n"
            "2. Pre-training Tasks\n"
            "We also use a Next Sentence Prediction (NSP) task that jointly pre-trains text-pair representations. "
            "BERT base has 12 transformer layers, 768 hidden dimensions, and 110 million parameters."
        ),
    },
]


def main():
    logger.info("Setting up RAG Pipeline components...")
    retrieval_service = VectorRetrievalService()
    qa_service = QuestionAnsweringService()

    # Index research corpus
    for doc in RESEARCH_CORPUS:
        count = retrieval_service.add_document(
            document_id=doc["doc_id"],
            text=doc["text"],
            chunk_size=40,
            chunk_overlap=10,
        )
        logger.info(f"Indexed {count} chunks for {doc['doc_id']}.")

    rag_service = RAGService(retrieval_service=retrieval_service, qa_service=qa_service)

    test_queries = [
        "How many layers are stacked in the Transformer encoder?",
        "What two pre-training tasks are used by BERT?",
        "How many total parameters does BERT base have?",
        "What is the average rainfall in the Amazon rainforest?",  # Unanswerable query
    ]

    print("\n" + "=" * 80)
    print("           RESEARCHGPT GROUNDED RAG PIPELINE VERIFICATION")
    print("=" * 80)

    for q in test_queries:
        req = RAGQueryRequest(query=q, top_k=2, similarity_threshold=0.25)
        resp = rag_service.query(req)

        print(f"\n[QUERY]: \"{resp.query}\"")
        print(f"[GROUNDED ANSWER]: {resp.answer}")
        print(f"[STATUS]: Grounded={resp.is_grounded} | Confidence={resp.confidence_score:.4f} | Latency={resp.total_latency_ms} ms (Retrieval: {resp.retrieval_latency_ms} ms, QA: {resp.qa_latency_ms} ms)")
        if resp.citations:
            print("[SOURCE CITATIONS]:")
            for c in resp.citations:
                print(f"   [{c.citation_index}] Document: {c.document_id} | Chunk: {c.chunk_id} | Similarity: {c.similarity_score:.4f}")
                print(f"       Excerpt: \"{c.text_snippet}\"")
        print("-" * 80)


if __name__ == "__main__":
    main()
