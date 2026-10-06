"""Unit tests for Vector Indexing and Semantic Retrieval Service (Task 15 / FR-08)."""

import shutil
from pathlib import Path

import pytest
from backend.app.models.embeddings import ChunkMetadata, DocumentChunk
from backend.app.models.retrieval import RetrievalQuery
from backend.app.services.embedding_service import _DEPS_AVAILABLE
from backend.app.services.faiss_retrieval_service import VectorRetrievalService


@pytest.fixture
def sample_research_chunks():
    chunks = [
        DocumentChunk(
            chunk_id="paper_transformer_chunk_0",
            text="The Transformer model architecture relies entirely on self-attention mechanisms without recurrence or convolutions.",
            metadata=ChunkMetadata(
                document_id="paper_transformer",
                chunk_index=0,
                page_number=1,
                section_title="Abstract",
                token_count=16,
                char_start=0,
                char_end=115,
            ),
        ),
        DocumentChunk(
            chunk_id="paper_bert_chunk_0",
            text="BERT introduces deep bidirectional transformer representations pre-trained on masked language modeling.",
            metadata=ChunkMetadata(
                document_id="paper_bert",
                chunk_index=0,
                page_number=1,
                section_title="Abstract",
                token_count=14,
                char_start=0,
                char_end=103,
            ),
        ),
        DocumentChunk(
            chunk_id="paper_cooking_chunk_0",
            text="Preheat the oven to 375 degrees Fahrenheit and knead the dough until smooth and elastic.",
            metadata=ChunkMetadata(
                document_id="recipe_book",
                chunk_index=0,
                page_number=5,
                section_title="Baking Guide",
                token_count=15,
                char_start=0,
                char_end=90,
            ),
        ),
    ]
    return chunks


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for VectorRetrievalService")
def test_vector_retrieval_indexing_and_search(sample_research_chunks):
    service = VectorRetrievalService()
    added = service.add_chunks(sample_research_chunks)
    assert added == 3

    # Query about Transformers
    query = RetrievalQuery(query_text="How does the Transformer model replace recurrence with self-attention?", top_k=2)
    resp = service.search(query)

    assert resp.total_retrieved == 2
    # The top retrieved chunk should be the Transformer chunk
    assert resp.results[0].chunk_id == "paper_transformer_chunk_0"
    assert resp.results[0].rank == 1
    assert resp.results[0].similarity_score > 0.60

    # The recipe chunk should NOT be in top 1
    assert resp.results[0].chunk_id != "paper_cooking_chunk_0"


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for VectorRetrievalService")
def test_vector_retrieval_document_filtering(sample_research_chunks):
    service = VectorRetrievalService()
    service.add_chunks(sample_research_chunks)

    # Filter to only paper_bert
    resp = service.search(
        query="Explain language representation models",
        top_k=5,
        document_id="paper_bert",
    )

    assert resp.total_retrieved == 1
    assert resp.results[0].metadata.document_id == "paper_bert"


def test_vector_retrieval_empty_index_behavior():
    service = VectorRetrievalService()
    info = service.get_index_info()
    assert info.total_vectors == 0
    assert info.unique_documents == 0

    resp = service.search(query="any random question", top_k=3)
    assert resp.total_retrieved == 0
    assert len(resp.results) == 0


def test_vector_retrieval_empty_query_raises():
    service = VectorRetrievalService()
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        service.search(query="   ")


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for VectorRetrievalService")
def test_vector_retrieval_save_and_load_roundtrip(sample_research_chunks, tmp_path):
    save_dir = tmp_path / "test_vector_index"

    service = VectorRetrievalService()
    service.add_chunks(sample_research_chunks)
    service.save_index(save_dir)

    loaded_service = VectorRetrievalService.load_index(save_dir)
    assert loaded_service.get_index_info().total_vectors == 3
    assert len(loaded_service.chunks) == 3

    resp = loaded_service.search(query="What is BERT?", top_k=1)
    assert resp.total_retrieved == 1
    assert resp.results[0].chunk_id == "paper_bert_chunk_0"
