"""Semantic Similarity API Router (Task 18 / FR-07)."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.deps import get_similarity_service
from backend.app.models.similarity import (
    SimilarityBatchRequest,
    SimilarityBatchResponse,
    SimilarityPairRequest,
    SimilarityPairResponse,
)
from backend.app.services.similarity_service import SemanticSimilarityService

router = APIRouter(prefix="/similarity", tags=["Semantic Similarity"])


@router.post(
    "/compare",
    response_model=SimilarityPairResponse,
    summary="Compute semantic textual similarity between a pair of sentences",
    status_code=status.HTTP_200_OK,
)
def compare_pair(
    payload: SimilarityPairRequest,
    service: SemanticSimilarityService = Depends(get_similarity_service),
) -> SimilarityPairResponse:
    """Compare two sentences/texts using Sentence-BERT or TF-IDF representations."""
    if not payload.text_a.strip() or not payload.text_b.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both text_a and text_b must be non-empty strings.",
        )

    try:
        return service.compute_similarity_pair(
            text_a=payload.text_a,
            text_b=payload.text_b,
            method=payload.method,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity comparison failed: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=SimilarityBatchResponse,
    summary="Batch semantic similarity computation",
    status_code=status.HTTP_200_OK,
)
def compare_batch(
    payload: SimilarityBatchRequest,
    service: SemanticSimilarityService = Depends(get_similarity_service),
) -> SimilarityBatchResponse:
    """Compute semantic similarity scores for a batch of text pairs."""
    if not payload.pairs:
        return SimilarityBatchResponse(results=[], total_pairs=0, mean_similarity=0.0)

    try:
        pairs = [(p.text_a, p.text_b) for p in payload.pairs]
        method = payload.pairs[0].method if payload.pairs else "sentence_bert"
        return service.compute_similarity_batch(pairs, method=method)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch similarity computation failed: {exc}",
        ) from exc
