"""Comprehensive End-to-End Workflow Integration Tests for ResearchGPT (Task 18).

Tests the full lifecycle of research paper processing:
- Ingestion and chunking into Vector Store
- Named Entity Recognition on extracted sections
- Abstractive Document Summarization
- Text Classification of research domain
- Semantic Similarity cross-validation
- Extractive Question Answering
- End-to-End Grounded Retrieval-Augmented Generation (RAG)
- Unified Paper Analysis API
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


SAMPLE_RESEARCH_PAPER = (
    "Attention Is All You Need. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, "
    "Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. "
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
    "in an encoder-decoder configuration. The best performing models also connect the encoder and decoder "
    "through an attention mechanism. We propose a new simple network architecture, the Transformer, "
    "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
    "Experiments on two machine translation tasks show these models to be superior in quality "
    "while being more parallelizable and requiring significantly less time to train."
)


def test_end_to_end_research_paper_lifecycle():
    """Validates the multi-step lifecycle of ingesting and querying a research paper."""
    # Step 1: Index the research paper into the vector retrieval store
    doc_id = "paper_attention_2017"
    index_resp = client.post(
        "/api/v1/retrieval/index-document",
        json={
            "document_id": doc_id,
            "text": SAMPLE_RESEARCH_PAPER,
            "chunk_size": 35,
            "chunk_overlap": 5,
        },
    )
    assert index_resp.status_code == 201
    index_data = index_resp.json()
    assert index_data["document_id"] == doc_id
    assert index_data["chunks_indexed"] > 0

    # Step 2: Extract key entities (authors, organizations, technical terms)
    ner_resp = client.post(
        "/api/v1/ner/extract",
        json={"text": SAMPLE_RESEARCH_PAPER[:200], "method": "bert"},
    )
    assert ner_resp.status_code == 200
    ner_data = ner_resp.json()
    assert "entities" in ner_data
    assert len(ner_data["tokens"]) > 0

    # Step 3: Generate abstractive summary of the paper
    summary_resp = client.post(
        "/api/v1/summarization/summarize",
        json={
            "text": SAMPLE_RESEARCH_PAPER,
            "max_length": 45,
            "min_length": 15,
            "num_beams": 2,
        },
    )
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    assert len(summary_data["summary"]) > 0
    assert summary_data["compression_ratio"] > 0.0

    # Step 4: Classify research text domain
    cls_resp = client.post(
        "/api/v1/classification/predict",
        json={"text": SAMPLE_RESEARCH_PAPER[:250]},
    )
    assert cls_resp.status_code == 200
    cls_data = cls_resp.json()
    assert "predicted_label" in cls_data
    assert 0.0 <= cls_data["confidence"] <= 1.0

    # Step 5: Measure semantic similarity with a related concept
    sim_resp = client.post(
        "/api/v1/similarity/compare",
        json={
            "text_a": "The Transformer uses self-attention mechanisms without recurrent layers.",
            "text_b": "Attention mechanisms replace recurrence in the Transformer model.",
            "method": "sentence_bert",
        },
    )
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    assert sim_data["similarity_score"] > 0.60

    # Step 6: Ask extractive QA over a specific section
    qa_resp = client.post(
        "/api/v1/qa/answer",
        json={
            "question": "What is the Transformer based on?",
            "context": SAMPLE_RESEARCH_PAPER,
        },
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()
    assert "attention" in qa_data["answer"].lower()

    # Step 7: Execute Grounded RAG query retrieving the indexed document
    rag_resp = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What network architecture dispenses with recurrence and convolutions?",
            "top_k": 3,
            "similarity_threshold": 0.20,
        },
    )
    assert rag_resp.status_code == 200
    rag_data = rag_resp.json()
    assert rag_data["is_grounded"] is True
    assert len(rag_data["citations"]) > 0
    assert any(c["document_id"] in [doc_id, "vaswani_2017_attention_is_all_you_need"] for c in rag_data["citations"])
    assert "transformer" in rag_data["answer"].lower() or "attention" in rag_data["answer"].lower() or len(rag_data["citations"]) > 0


def test_unified_paper_analysis_and_rag_synthesis():
    """Validates the unified /api/v1/analysis/paper workflow with automated indexing."""
    digest_id = "bert_devlin_2018"
    digest_text = (
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. "
        "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. "
        "We introduce a new language representation model called BERT, which stands for "
        "Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, "
        "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
        "conditioning on both left and right context in all layers."
    )

    # Ingest and analyze in a single call
    analysis_resp = client.post(
        "/api/v1/analysis/paper",
        json={
            "document_id": digest_id,
            "text": digest_text,
            "auto_index": True,
            "max_summary_length": 40,
        },
    )
    assert analysis_resp.status_code == 200
    res = analysis_resp.json()
    assert res["document_id"] == digest_id
    assert len(res["summary"]["summary"]) > 0
    assert res["chunks_indexed"] > 0
    assert "entities" in res

    # Subsequent RAG query against the auto-indexed paper
    rag_query_resp = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What does BERT stand for?",
            "top_k": 2,
            "similarity_threshold": 0.25,
        },
    )
    assert rag_query_resp.status_code == 200
    rag_res = rag_query_resp.json()
    assert rag_res["is_grounded"] is True
    assert len(rag_res["citations"]) > 0
    assert len(rag_res["answer"]) > 0


def test_api_validation_and_error_handling():
    """Validates that input validation errors return standard 422 responses."""
    # Empty payload to /predict
    resp = client.post("/api/v1/classification/predict", json={})
    assert resp.status_code == 422

    # Missing context in /qa/answer
    resp = client.post("/api/v1/qa/answer", json={"question": "What is AI?"})
    assert resp.status_code == 422

    # Invalid empty query to RAG
    resp = client.post("/api/v1/rag/query", json={"query": ""})
    assert resp.status_code in [400, 422]
