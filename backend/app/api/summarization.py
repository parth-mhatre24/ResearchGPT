"""Summarization API Router (Task 18 / FR-06)."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.deps import get_summarization_service
from backend.app.models.summarization import (
    SummarizationRequest,
    SummarizationResponse,
)
from backend.app.services.summarization_service import SummarizationService

router = APIRouter(prefix="/summarization", tags=["Summarization"])


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Generate abstractive summary of document or research paper",
    status_code=status.HTTP_200_OK,
)
def summarize_text(
    payload: SummarizationRequest,
    summarizer: SummarizationService = Depends(get_summarization_service),
) -> SummarizationResponse:
    """Generate concise abstractive summary using T5 / BART transformer models."""
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text to summarize cannot be empty.",
        )

    try:
        return summarizer.summarize(
            text=payload.text,
            max_length=payload.max_length or 150,
            min_length=payload.min_length or 30,
            num_beams=payload.num_beams or 4,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {exc}",
        ) from exc
