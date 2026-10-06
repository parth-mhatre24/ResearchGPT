"""Vector Indexing and Semantic Retrieval API Router (Task 18 / FR-08)."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.deps import get_retrieval_service
from backend.app.models.retrieval import (
    IndexInfo,
    RetrievalQuery,
    RetrievalResponse,
)
from backend.app.services.faiss_retrieval_service import VectorRetrievalService

router = APIRouter(prefix="/retrieval", tags=["Vector Retrieval & Indexing"])


class IndexDocumentRequest(BaseModel):
    """Request payload to index a document into the vector database."""

    document_id: str = Field(..., description="Unique document ID or filename")
    text: str = Field(..., description="Extracted document text")
    chunk_size: Optional[int] = Field(default=256, description="Target chunk size in words")
    chunk_overlap: Optional[int] = Field(default=32, description="Chunk overlap in words")
    page_number: Optional[int] = Field(default=None, description="Optional page number")


class IndexDocumentResponse(BaseModel):
    """Response payload after indexing a document."""

    document_id: str
    chunks_indexed: int
    total_vectors_in_index: int
    message: str


@router.post(
    "/index-document",
    response_model=IndexDocumentResponse,
    summary="Chunk, embed, and index a document into vector storage",
    status_code=status.HTTP_201_CREATED,
)
def index_document(
    payload: IndexDocumentRequest,
    retrieval_service: VectorRetrievalService = Depends(get_retrieval_service),
) -> IndexDocumentResponse:
    """Chunk document text, compute Sentence-BERT embeddings, and store in the vector index."""
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text cannot be empty.",
        )

    try:
        count = retrieval_service.add_document(
            document_id=payload.document_id,
            text=payload.text,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            page_number=payload.page_number,
        )
        info = retrieval_service.get_index_info()

        return IndexDocumentResponse(
            document_id=payload.document_id,
            chunks_indexed=count,
            total_vectors_in_index=info.total_vectors,
            message=f"Successfully indexed {count} chunks for document [{payload.document_id}].",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {exc}",
        ) from exc


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Semantic nearest-neighbor vector search",
    status_code=status.HTTP_200_OK,
)
def search_vectors(
    payload: RetrievalQuery,
    retrieval_service: VectorRetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    """Retrieve top-k most semantically relevant document chunks matching user query."""
    if not payload.query_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty.",
        )

    try:
        return retrieval_service.search(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval search failed: {exc}",
        ) from exc


@router.get(
    "/status",
    response_model=IndexInfo,
    summary="Get vector index status and statistics",
    status_code=status.HTTP_200_OK,
)
def get_index_status(
    retrieval_service: VectorRetrievalService = Depends(get_retrieval_service),
) -> IndexInfo:
    """Return total vectors, dimensionality, and indexed document list."""
    return retrieval_service.get_index_info()
