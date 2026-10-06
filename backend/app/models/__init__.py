"""Models package for data structures and domain entities."""

from backend.app.models.classification import (
    ClassificationEvaluationMetrics,
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.models.document import (
    DocumentExtractionResponse,
    DocumentUploadResponse,
    PageExtraction,
)
from backend.app.models.ner import (
    NEREvaluationMetrics,
    NEREntity,
    NEREntityMetrics,
    NERRequest,
    NERResponse,
)
from backend.app.models.preprocessing import (
    ClassicalPreprocessingRequest,
    ClassicalPreprocessingResponse,
    TransformerPreprocessingRequest,
    TransformerPreprocessingResponse,
)
from backend.app.models.summarization import (
    SummarizationEvaluationMetrics,
    SummarizationRequest,
    SummarizationResponse,
)

__all__ = [
    "ClassificationEvaluationMetrics",
    "ClassificationRequest",
    "ClassificationResponse",
    "DocumentExtractionResponse",
    "DocumentUploadResponse",
    "PageExtraction",
    "NEREvaluationMetrics",
    "NEREntity",
    "NEREntityMetrics",
    "NERRequest",
    "NERResponse",
    "ClassicalPreprocessingRequest",
    "ClassicalPreprocessingResponse",
    "TransformerPreprocessingRequest",
    "TransformerPreprocessingResponse",
    "SummarizationEvaluationMetrics",
    "SummarizationRequest",
    "SummarizationResponse",
]
