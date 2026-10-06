"""Unit tests for Retrieval-Augmented Generation (RAG) Service (Task 17 / FR-10)."""

import pytest
from backend.app.models.embeddings import ChunkMetadata, DocumentChunk
from backend.app.models.rag import RAGQueryRequest
from backend.app.services.embedding_service import _DEPS_AVAILABLE
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.qa_service import QuestionAnsweringService
from backend.app.services.rag_service import RAGService


@pytest.fixture
def rag_service_with_corpus():
    chunks = [
        DocumentChunk(
            chunk_id="transformer_p1",
            text="The Transformer was introduced in 2017 by Vaswani et al. and is based solely on attention mechanisms.",
            metadata=ChunkMetadata(
                document_id="attention_paper",
                chunk_index=0,
                page_number=1,
                section_title="Introduction",
                token_count=18,
                char_start=0,
                char_end=110,
            ),
        ),
        DocumentChunk(
            chunk_id="bert_p1",
            text="BERT was created by Google researchers in 2018 to learn bidirectional contextual representations.",
            metadata=ChunkMetadata(
                document_id="bert_paper",
                chunk_index=0,
                page_number=1,
                section_title="Abstract",
                token_count=15,
                char_start=0,
                char_end=100,
            ),
        ),
    ]

    retrieval_service = VectorRetrievalService()
    retrieval_service.add_chunks(chunks)
    qa_service = QuestionAnsweringService()
    rag = RAGService(retrieval_service=retrieval_service, qa_service=qa_service)
    return rag


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for RAGService")
def test_rag_query_grounded_answer(rag_service_with_corpus):
    req = RAGQueryRequest(query="Who introduced the Transformer architecture?", top_k=2, generation_mode="generative")
    resp = rag_service_with_corpus.query(req)

    assert resp.is_grounded is True
    assert "Vaswani" in resp.answer
    assert resp.confidence_score > 0.0
    assert len(resp.citations) > 0
    assert resp.citations[0].document_id == "attention_paper"
    assert resp.total_latency_ms > 0.0


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for RAGService")
def test_rag_query_extractive_mode(rag_service_with_corpus):
    req = RAGQueryRequest(query="Who introduced the Transformer architecture?", top_k=2, generation_mode="extractive")
    resp = rag_service_with_corpus.query(req)

    assert resp.is_grounded is True
    assert "Vaswani" in resp.answer
    assert resp.confidence_score > 0.0
    assert len(resp.citations) > 0


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for RAGService")
def test_rag_query_unanswerable_threshold():
    # Empty retrieval service
    empty_rag = RAGService()
    resp = empty_rag.query(query="What is the boiling point of liquid nitrogen?", similarity_threshold=0.8)

    assert resp.is_grounded is False
    assert "could not find sufficient relevant evidence" in resp.answer
    assert len(resp.citations) == 0


def test_rag_empty_query_raises():
    rag = RAGService()
    with pytest.raises(ValueError, match="Query cannot be empty"):
        rag.query(query="   ")
