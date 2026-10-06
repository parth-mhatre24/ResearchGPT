"""Pydantic models for Extractive Question Answering (Task 16 / FR-09)."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    """Request to extract an answer from context given a question."""

    question: str = Field(..., description="The question string")
    context: str = Field(..., description="The context paragraph containing the answer")
    model_name: Optional[str] = Field(default=None, description="Optional QA model override")


class QAResponse(BaseModel):
    """Response containing extracted answer span and confidence metadata."""

    question: str
    answer: str = Field(..., description="Extracted answer substring from context")
    start_char: int = Field(..., description="Starting character index in context")
    end_char: int = Field(..., description="Ending character index in context")
    confidence_score: float = Field(..., description="Softmax confidence probability [0.0, 1.0]")
    model_name: str
    latency_ms: float


class QABatchRequest(BaseModel):
    """Batch question answering request."""

    items: List[QARequest]


class QABatchResponse(BaseModel):
    """Batch question answering response."""

    results: List[QAResponse]
    total_items: int
    avg_confidence: float


class QAEvaluationMetrics(BaseModel):
    """Evaluation metrics for Question Answering on benchmark pairs."""

    exact_match: float = Field(..., description="Exact match percentage (0.0 to 100.0)")
    token_f1: float = Field(..., description="Token-level F1 score (0.0 to 100.0)")
    precision: float = Field(..., description="Token-level Precision (0.0 to 100.0)")
    recall: float = Field(..., description="Token-level Recall (0.0 to 100.0)")
    sample_count: int
    model_name: str
