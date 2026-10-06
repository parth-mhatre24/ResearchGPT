"""Extractive Question Answering API Router (Task 18 / FR-09)."""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.deps import get_qa_service
from backend.app.models.qa import (
    QABatchRequest,
    QABatchResponse,
    QARequest,
    QAResponse,
)
from backend.app.services.qa_service import QuestionAnsweringService

router = APIRouter(prefix="/qa", tags=["Question Answering"])


@router.post(
    "/answer",
    response_model=QAResponse,
    summary="Answer a question from given context text",
    status_code=status.HTTP_200_OK,
)
def answer_question(
    payload: QARequest,
    qa_service: QuestionAnsweringService = Depends(get_qa_service),
) -> QAResponse:
    """Extract answer span and confidence score for a question given context."""
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )
    if not payload.context.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Context cannot be empty.",
        )

    try:
        return qa_service.answer_question(
            question=payload.question,
            context=payload.context,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering inference failed: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=QABatchResponse,
    summary="Batch question answering",
    status_code=status.HTTP_200_OK,
)
def answer_batch(
    payload: QABatchRequest,
    qa_service: QuestionAnsweringService = Depends(get_qa_service),
) -> QABatchResponse:
    """Answer a batch of questions over corresponding context texts."""
    try:
        return qa_service.answer_batch(payload.items)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch question answering failed: {exc}",
        ) from exc
