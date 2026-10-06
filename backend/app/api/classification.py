"""Classification API Router (Task 18 / FR-04)."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.deps import get_classical_classifier_service
from backend.app.models.classification import (
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.services.classical_classifier_service import ClassicalClassifierService

router = APIRouter(prefix="/classification", tags=["Classification"])


class ClassificationBatchRequest(BaseModel):
    """Batch text classification request."""

    texts: List[str] = Field(..., description="List of raw texts to classify")
    model_type: str = Field(default="logistic_regression", description="Classifier model to use")


class ClassificationBatchResponse(BaseModel):
    """Batch text classification response."""

    predictions: List[str]
    total_samples: int


@router.post(
    "/predict",
    response_model=ClassificationResponse,
    summary="Classify a single text snippet",
    status_code=status.HTTP_200_OK,
)
def predict_text(
    payload: ClassificationRequest,
    classifier: ClassicalClassifierService = Depends(get_classical_classifier_service),
) -> ClassificationResponse:
    """Classify input text using the trained classifier model."""
    try:
        if not classifier.is_trained:
            # Auto-train on tiny fallback or raise friendly message
            sample_corpus = [
                "This research paper introduces novel deep learning architectures.",
                "Free prize! Call now to claim your urgent mobile lottery cash reward.",
            ]
            sample_labels = ["research", "spam"]
            classifier.train(sample_corpus, sample_labels)

        return classifier.predict_single(payload.text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification inference failed: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=ClassificationBatchResponse,
    summary="Batch text classification",
    status_code=status.HTTP_200_OK,
)
def predict_batch(
    payload: ClassificationBatchRequest,
    classifier: ClassicalClassifierService = Depends(get_classical_classifier_service),
) -> ClassificationBatchResponse:
    """Classify a list of texts in batch."""
    try:
        if not classifier.is_trained:
            sample_corpus = ["Sample research document.", "Urgent spam notice."]
            sample_labels = ["research", "spam"]
            classifier.train(sample_corpus, sample_labels)

        preds = classifier.predict(payload.texts)
        return ClassificationBatchResponse(
            predictions=[str(p) for p in preds],
            total_samples=len(payload.texts),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch classification failed: {exc}",
        ) from exc
