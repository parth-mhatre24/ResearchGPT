"""
Tests for System Hardening, Edge Cases, and Boundary Resiliency (Task 20).
"""
import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.rag_service import RAGService
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.flan_t5_service import FlanT5GenerativeService


client = TestClient(app)


def test_corrupt_pdf_magic_bytes_rejected():
    """Verify that a corrupted or non-PDF file is rejected with 400 Bad Request."""
    corrupted_data = b"NOT_A_REAL_PDF_HEADER_12345"
    file = ("fake.pdf", io.BytesIO(corrupted_data), "application/pdf")
    
    response = client.post("/api/v1/documents/upload", files={"file": file})
    assert response.status_code == 400
    assert "header does not match valid PDF" in response.json()["detail"]


def test_zero_byte_pdf_rejected():
    """Verify that an empty zero-byte file is rejected with 400 Bad Request."""
    empty_file = ("empty.pdf", io.BytesIO(b""), "application/pdf")
    
    response = client.post("/api/v1/documents/upload", files={"file": empty_file})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_rag_with_no_relevant_context_returns_grounded_refusal():
    """Verify that querying with empty/irrelevant context produces a grounded refusal without hallucination."""
    rag_service = RAGService()
    
    # Query with a very high similarity threshold to simulate zero relevant chunks found
    result = rag_service.query(
        query="What is the recipe for chocolate chip cookies?",
        top_k=3,
        similarity_threshold=0.99,
    )
    assert result is not None
    assert "cannot answer" in result.answer.lower() or "not provided" in result.answer.lower() or len(result.citations) == 0


def test_flan_t5_special_unicode_handling():
    """Verify that scientific unicode symbols (alpha, beta, equations) do not crash generation."""
    flan_service = FlanT5GenerativeService()
    context = "In our experiments, the learning rate alpha was set to 0.001 with weight decay lambda = 0.01."
    
    answer = flan_service.generate_answer(
        question="What was the learning rate \u03b1 and decay \u03bb?",
        contexts=[context],
    )
    assert len(answer) > 0
    assert answer != "Error"


def test_delete_nonexistent_document_returns_404():
    """Verify deleting a non-existent document ID returns a clean 404."""
    response = client.delete("/api/v1/documents/non_existent_doc_id_999999")
    assert response.status_code == 404
