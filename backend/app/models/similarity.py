"""Pydantic models for Semantic Similarity (Task 13 / FR-07)."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SimilarityPairRequest(BaseModel):
    """Request to compute similarity between two sentences/texts."""

    text_a: str = Field(..., description="First text string")
    text_b: str = Field(..., description="Second text string")
    method: Literal["sentence_bert", "tfidf"] = Field(
        default="sentence_bert", description="Method to compute similarity"
    )


class SimilarityPairResponse(BaseModel):
    """Response containing semantic similarity scores."""

    text_a: str
    text_b: str
    similarity_score: float = Field(
        ..., description="Cosine similarity score normalized between 0.0 and 1.0"
    )
    score_stsb_scale: float = Field(
        ..., description="Similarity score scaled to STS-B benchmark scale [0.0, 5.0]"
    )
    method: str
    latency_ms: float


class SimilarityBatchRequest(BaseModel):
    """Batch similarity request for multiple text pairs."""

    pairs: List[SimilarityPairRequest]


class SimilarityBatchResponse(BaseModel):
    """Batch similarity response containing individual pair scores."""

    results: List[SimilarityPairResponse]
    total_pairs: int
    mean_similarity: float


class SimilarityEvaluationMetrics(BaseModel):
    """Evaluation metrics on STS Benchmark (STS-B)."""

    pearson_correlation: float = Field(..., description="Pearson correlation coefficient (r)")
    spearman_correlation: float = Field(..., description="Spearman rank correlation coefficient (rho)")
    mse: float = Field(..., description="Mean squared error against STS-B labels [0, 5]")
    sample_count: int
    method: str
    model_name: Optional[str] = None
