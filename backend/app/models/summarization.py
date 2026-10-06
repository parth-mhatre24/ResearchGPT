"""Pydantic schemas for Summarization models, requests, and evaluation metrics."""

from typing import Optional
from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """Request schema for text summarization."""

    text: str = Field(..., description="Raw input text to be summarized")
    max_length: Optional[int] = Field(
        150, description="Maximum token/word length of generated summary"
    )
    min_length: Optional[int] = Field(
        30, description="Minimum token/word length of generated summary"
    )
    num_beams: Optional[int] = Field(
        4, description="Number of beams for beam search generation"
    )
    model_type: Optional[str] = Field(
        "t5", description="Model architecture to use: 't5' or 'bart'"
    )


class SummarizationResponse(BaseModel):
    """Response schema returned by the summarization service."""

    summary: str = Field(..., description="Generated abstractive summary text")
    input_length_words: int = Field(..., description="Word count of the input text")
    summary_length_words: int = Field(..., description="Word count of the generated summary")
    compression_ratio: float = Field(
        ..., description="Compression ratio (summary_length / input_length)"
    )
    model_type: str = Field(..., description="Model architecture used for generation")
    execution_time_sec: float = Field(
        ..., description="Time taken to generate the summary in seconds"
    )


class SummarizationEvaluationMetrics(BaseModel):
    """Top-level summarization evaluation result using ROUGE scores."""

    rouge1: float = Field(..., description="ROUGE-1 F1 score (unigram overlap)")
    rouge2: float = Field(..., description="ROUGE-2 F1 score (bigram overlap)")
    rougeL: float = Field(
        ..., description="ROUGE-L F1 score (longest common subsequence)"
    )
    sample_count: int = Field(..., description="Number of document-summary pairs evaluated")
    model_type: str = Field(..., description="Model architecture evaluated")
