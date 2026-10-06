"""Pydantic models for Vector Indexing and Semantic Retrieval (Task 15 / FR-08)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.models.embeddings import ChunkMetadata, DocumentChunk


class RetrievalQuery(BaseModel):
    """Query model for semantic nearest-neighbor retrieval."""

    query_text: str = Field(..., description="User query or research question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of relevant chunks to retrieve")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional minimum cosine similarity threshold"
    )
    document_id: Optional[str] = Field(
        default=None, description="Optional filter to retrieve only chunks from a specific document"
    )


class RetrievedDocumentChunk(BaseModel):
    """Retrieved document chunk with similarity score and ranking."""

    rank: int = Field(..., description="1-indexed relevance rank")
    chunk_id: str = Field(..., description="Document chunk identifier")
    text: str = Field(..., description="Text content of retrieved chunk")
    similarity_score: float = Field(..., description="Cosine similarity / inner-product score [0.0, 1.0]")
    metadata: ChunkMetadata


class RetrievalResponse(BaseModel):
    """Response containing ranked retrieved document chunks."""

    query: str
    total_retrieved: int
    top_k: int
    results: List[RetrievedDocumentChunk]
    retrieval_latency_ms: float


class IndexInfo(BaseModel):
    """Status and metadata of the active vector index."""

    total_vectors: int
    dimension: int
    index_type: str
    unique_documents: int
    document_ids: List[str]
