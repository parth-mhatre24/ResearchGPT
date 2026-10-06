"""Pydantic models for Document Chunking and Dense Embeddings (Task 14 / FR-08)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with an individual document chunk."""

    document_id: str = Field(..., description="ID or filename of parent document")
    chunk_index: int = Field(..., description="0-indexed position in document")
    page_number: Optional[int] = Field(default=None, description="PDF page number if available")
    section_title: Optional[str] = Field(default=None, description="Heading/section title if detected")
    token_count: int = Field(..., description="Approximate word/token count in chunk")
    char_start: int = Field(..., description="Starting character index in full document")
    char_end: int = Field(..., description="Ending character index in full document")


class DocumentChunk(BaseModel):
    """A semantic text chunk with attached provenance metadata."""

    chunk_id: str = Field(..., description="Globally unique chunk identifier, e.g. doc_1_chunk_0")
    text: str = Field(..., description="Raw text content of the chunk")
    metadata: ChunkMetadata


class ChunkingRequest(BaseModel):
    """Request to partition document text into semantic chunks."""

    text: str = Field(..., description="Full extracted document text")
    document_id: str = Field(default="doc_default", description="Identifier of the document")
    chunk_size: int = Field(default=256, description="Target chunk size in approximate words/tokens")
    chunk_overlap: int = Field(default=32, description="Number of overlapping words/tokens between consecutive chunks")
    page_number: Optional[int] = Field(default=None, description="Optional page number")


class ChunkingResponse(BaseModel):
    """Response containing extracted chunks and summary statistics."""

    document_id: str
    total_chunks: int
    chunks: List[DocumentChunk]
    avg_chunk_length: float


class EmbeddingRequest(BaseModel):
    """Request to compute dense vector embeddings for texts."""

    texts: List[str] = Field(..., description="List of text chunks to embed")
    model_name: Optional[str] = Field(default=None, description="Optional embedding model override")


class EmbeddingResponse(BaseModel):
    """Response containing dense float vector embeddings."""

    embeddings: List[List[float]] = Field(..., description="List of dense float vectors")
    dimension: int = Field(..., description="Vector dimensionality, e.g. 384")
    model_name: str
    sample_count: int
    latency_ms: float
