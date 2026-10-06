"""Pydantic models for Retrieval-Augmented Generation / RAG Pipeline (Task 17 / FR-10)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Source provenance citation linking answer to specific document chunk."""

    citation_index: int = Field(..., description="1-indexed citation reference, e.g. [1]")
    document_id: str = Field(..., description="Document identifier/title")
    page_number: Optional[int] = Field(default=None, description="Page number in original document")
    section_title: Optional[str] = Field(default=None, description="Section heading")
    chunk_id: str = Field(..., description="Unique chunk ID")
    similarity_score: float = Field(..., description="Cosine retrieval similarity score")
    text_snippet: str = Field(..., description="Relevant text excerpt supporting the answer")


class RAGQueryRequest(BaseModel):
    """Request for grounded question answering across indexed research papers."""

    query: str = Field(..., description="User question or research inquiry")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of document chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Minimum similarity threshold to consider chunk relevant"
    )
    document_id: Optional[str] = Field(
        default=None, description="Optional document filter to restrict retrieval to a single paper"
    )
    generation_mode: Optional[str] = Field(
        default="generative", description="Generation strategy: 'generative' (FLAN-T5) or 'extractive' (DistilBERT)"
    )


class RAGQueryResponse(BaseModel):
    """Grounded RAG response with provenance source citations."""

    query: str
    answer: str = Field(..., description="Grounded answer to user question")
    confidence_score: float = Field(..., description="QA confidence probability [0.0, 1.0]")
    is_grounded: bool = Field(..., description="True if answer is supported by retrieved document evidence")
    citations: List[SourceCitation] = Field(default_factory=list, description="List of source citations")
    retrieval_latency_ms: float
    qa_latency_ms: float
    total_latency_ms: float
