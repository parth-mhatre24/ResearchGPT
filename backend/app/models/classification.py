"""Pydantic schemas for classification models, requests, and evaluation metrics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    """Request for running text classification."""
    text: str = Field(..., description="Text input to classify")
    model_type: Optional[str] = Field("logistic_regression", description="Model architecture: naive_bayes, logistic_regression, or svm")


class ClassificationResponse(BaseModel):
    """Response returned by a classifier."""
    text: str = Field(..., description="Original input text")
    predicted_label: Any = Field(..., description="Predicted class label")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities if supported")
    confidence: Optional[float] = Field(None, description="Confidence score for top predicted class")
    model_type: str = Field(..., description="Model architecture used")


class ClassificationEvaluationMetrics(BaseModel):
    """Standardized classification evaluation metrics across models."""
    accuracy: float = Field(..., description="Accuracy score (0.0 to 1.0)")
    precision_macro: float = Field(..., description="Macro-averaged precision")
    recall_macro: float = Field(..., description="Macro-averaged recall")
    f1_macro: float = Field(..., description="Macro-averaged F1 score")
    precision_weighted: float = Field(..., description="Weighted precision")
    recall_weighted: float = Field(..., description="Weighted recall")
    f1_weighted: float = Field(..., description="Weighted F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix rows=actual, cols=predicted")
    classes: List[str] = Field(..., description="Class label names")
    sample_count: int = Field(..., description="Number of evaluated samples")
    classification_report: Optional[Dict[str, Any]] = Field(None, description="Detailed per-class report")
