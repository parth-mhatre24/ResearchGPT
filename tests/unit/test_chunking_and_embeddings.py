"""Unit tests for Document Chunking and Dense Embedding Services (Task 14 / FR-08)."""

import numpy as np
import pytest
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import (
    EmbeddingService,
    _DEPS_AVAILABLE,
)


def test_chunking_service_basic_split():
    service = ChunkingService(default_chunk_size=50, default_overlap=10)
    sample_text = (
        "The Transformer is a neural network architecture based on self-attention mechanisms. "
        "It was introduced in the seminal paper Attention Is All You Need. "
        "Unlike recurrent models, Transformers process all tokens simultaneously. "
        "This architectural innovation enables unprecedented parallelization during training. "
        "As a result, large language models such as BERT and GPT were made possible."
    )

    resp = service.chunk_text(sample_text, document_id="paper_1", chunk_size=20, chunk_overlap=5)
    assert resp.total_chunks > 1
    assert resp.document_id == "paper_1"
    for chunk in resp.chunks:
        assert chunk.chunk_id.startswith("paper_1_chunk_")
        assert len(chunk.text) > 0
        assert chunk.metadata.token_count > 0
        assert chunk.metadata.char_end > chunk.metadata.char_start


def test_chunking_service_empty_text():
    service = ChunkingService()
    resp = service.chunk_text("", document_id="empty_doc")
    assert resp.total_chunks == 0
    assert len(resp.chunks) == 0


def test_chunking_service_section_title_detection():
    service = ChunkingService()
    text = "1. Introduction\nLarge language models have revolutionized natural language processing."
    resp = service.chunk_text(text, document_id="doc_sec")
    assert resp.chunks[0].metadata.section_title is not None
    assert "Introduction" in resp.chunks[0].metadata.section_title


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="PyTorch/Transformers required for EmbeddingService")
def test_embedding_service_dimension_and_norm():
    service = EmbeddingService()
    texts = [
        "Self-attention mechanisms allow modeling dependencies without regard to distance.",
        "Deep learning architectures achieve state of the art results on benchmarks.",
    ]

    embs = service.embed_texts(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 384)

    # Check L2 unit normalization: norm of each vector ~ 1.0
    norms = np.linalg.norm(embs, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-5)


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="PyTorch/Transformers required for EmbeddingService")
def test_embedding_service_query_and_response():
    service = EmbeddingService()
    query = "What is the Transformer architecture?"
    q_vec = service.embed_query(query)
    assert q_vec.shape == (384,)
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)

    resp = service.get_embedding_response([query])
    assert resp.dimension == 384
    assert resp.sample_count == 1
    assert len(resp.embeddings[0]) == 384
