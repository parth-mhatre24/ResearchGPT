"""Integration tests for all ResearchGPT API Endpoints (Task 18)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data
    assert "health" in data["endpoints"]
    assert "rag" in data["endpoints"]


def test_health_endpoints():
    # Liveness check
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Detailed readiness check
    resp_detailed = client.get("/api/v1/health/detailed")
    assert resp_detailed.status_code == 200
    data = resp_detailed.json()
    assert data["status"] == "healthy"
    assert "services" in data
    assert data["services"]["rag_pipeline"] is True


def test_middleware_headers():
    resp = client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers
    assert "X-Process-Time-Ms" in resp.headers
    assert float(resp.headers["X-Process-Time-Ms"]) >= 0.0


def test_classification_predict_endpoint():
    payload = {"text": "A deep learning algorithm for natural language processing."}
    resp = client.post("/api/v1/classification/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_label" in data
    assert "confidence" in data


def test_classification_batch_endpoint():
    payload = {"texts": ["Sentence one about science.", "Urgent lottery prize winner."]}
    resp = client.post("/api/v1/classification/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_samples"] == 2
    assert len(data["predictions"]) == 2


def test_ner_extract_endpoint():
    payload = {"text": "Alice visited Paris and met with Google engineers.", "method": "bert"}
    resp = client.post("/api/v1/ner/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert "tokens" in data


def test_summarization_endpoint():
    payload = {
        "text": (
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."
        ),
        "max_length": 60,
        "min_length": 15,
        "num_beams": 2,
    }
    resp = client.post("/api/v1/summarization/summarize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert data["compression_ratio"] > 0.0


def test_similarity_compare_endpoint():
    payload = {
        "text_a": "The airplane is landing on the runway.",
        "text_b": "A plane is touching down on the airstrip.",
        "method": "sentence_bert",
    }
    resp = client.post("/api/v1/similarity/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["similarity_score"] >= 0.60
    assert data["score_stsb_scale"] >= 3.0


def test_similarity_batch_endpoint():
    payload = {
        "pairs": [
            {"text_a": "The dog ran.", "text_b": "A puppy ran.", "method": "sentence_bert"},
            {"text_a": "Physics is fun.", "text_b": "Cooking lasagna.", "method": "sentence_bert"},
        ]
    }
    resp = client.post("/api/v1/similarity/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_pairs"] == 2


def test_retrieval_index_and_search_endpoints():
    # 1. Index document
    doc_payload = {
        "document_id": "api_test_doc_1",
        "text": "The Transformer architecture relies entirely on multi-head self-attention mechanisms.",
        "chunk_size": 20,
        "chunk_overlap": 5,
    }
    resp_idx = client.post("/api/v1/retrieval/index-document", json=doc_payload)
    assert resp_idx.status_code == 201
    assert resp_idx.json()["chunks_indexed"] > 0

    # 2. Status check
    resp_status = client.get("/api/v1/retrieval/status")
    assert resp_status.status_code == 200
    assert resp_status.json()["total_vectors"] > 0

    # 3. Vector Search
    search_payload = {
        "query_text": "How does Transformer use self-attention?",
        "top_k": 2,
    }
    resp_search = client.post("/api/v1/retrieval/search", json=search_payload)
    assert resp_search.status_code == 200
    data = resp_search.json()
    assert data["total_retrieved"] > 0
    assert data["results"][0]["similarity_score"] > 0.50


def test_qa_answer_endpoint():
    payload = {
        "question": "What is the Transformer based on?",
        "context": "The Transformer is a deep learning architecture based solely on self-attention mechanisms.",
    }
    resp = client.post("/api/v1/qa/answer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "attention" in data["answer"].lower()
    assert data["confidence_score"] > 0.30


def test_rag_query_endpoint():
    payload = {
        "query": "How does the Transformer utilize self-attention mechanisms?",
        "top_k": 2,
        "similarity_threshold": 0.25,
    }
    resp = client.post("/api/v1/rag/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_grounded"] is True
    assert len(data["citations"]) > 0
    assert data["total_latency_ms"] > 0.0


def test_paper_analysis_endpoint():
    payload = {
        "document_id": "paper_digest_demo",
        "text": (
            "We introduce BERT: Bidirectional Encoder Representations from Transformers. "
            "BERT was created by Google researchers in Paris and New York. "
            "It is pre-trained using a Masked Language Model objective."
        ),
        "auto_index": True,
        "max_summary_length": 50,
    }
    resp = client.post("/api/v1/analysis/paper", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == "paper_digest_demo"
    assert "summary" in data
    assert "entities" in data
    assert data["chunks_indexed"] > 0
