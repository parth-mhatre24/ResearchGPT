"""Unit tests for document listing, indexing, and vector store purging."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.deps import get_retrieval_service

client = TestClient(app)


def test_list_documents_endpoint():
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    # Check that pre-seeded documents are present
    doc_ids = [d["document_id"] for d in docs]
    assert any("vaswani" in d or "bert" in d for d in doc_ids)


def test_delete_document_purges_vectors():
    retrieval = get_retrieval_service()
    doc_id = "test_doc_to_delete"
    sample_text = "This is a temporary test document to verify vector purging behavior in ResearchGPT."

    # 1. Add document to retrieval store
    chunks_added = retrieval.add_document(document_id=doc_id, text=sample_text, chunk_size=20)
    assert chunks_added > 0
    assert any(c.metadata.document_id == doc_id for c in retrieval.chunks)

    # 2. Call DELETE /api/v1/documents/{document_id}
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["document_id"] == doc_id
    assert del_data["chunks_purged"] == chunks_added

    # 3. Verify chunks are completely gone from retrieval index
    assert not any(c.metadata.document_id == doc_id for c in retrieval.chunks)
    assert doc_id not in retrieval.get_index_info().document_ids
