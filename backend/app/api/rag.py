"""Retrieval-Augmented Generation (RAG) API Router (Task 18 / FR-10)."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.deps import get_rag_service
from backend.app.models.rag import RAGQueryRequest, RAGQueryResponse
from backend.app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["Retrieval-Augmented Generation"])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Execute grounded research question answering with source citations",
    status_code=status.HTTP_200_OK,
)
def query_rag(
    payload: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Perform end-to-end vector retrieval and extractive QA to generate grounded answers with provenance citations."""
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    try:
        return rag_service.query(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {exc}",
        ) from exc
