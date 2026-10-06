"""Unit tests for Summarization Service (Task 12)."""

import pytest
from backend.app.models.summarization import SummarizationRequest, SummarizationResponse
from backend.app.services.summarization_service import (
    _DEPS_AVAILABLE,
    _ROUGE_AVAILABLE,
    SummarizationService,
)

pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE, reason="Transformers & PyTorch required for Summarization tests."
)


def test_summarization_schema_validation():
    req = SummarizationRequest(text="This is a research paper on natural language processing.")
    assert req.max_length == 150
    assert req.model_type == "t5"

    resp = SummarizationResponse(
        summary="Research paper summary.",
        input_length_words=10,
        summary_length_words=3,
        compression_ratio=0.3,
        model_type="t5-small",
        execution_time_sec=0.1,
    )
    assert resp.compression_ratio == 0.3


@pytest.mark.skipif(not _ROUGE_AVAILABLE, reason="rouge-score package required.")
def test_summarization_evaluate_rouge():
    service = SummarizationService()
    references = ["The Transformer model uses self-attention."]
    generated = ["The Transformer relies on attention mechanisms."]

    metrics = service.evaluate(references, generated)
    assert metrics.rouge1 > 0.0
    assert metrics.rouge2 > 0.0
    assert metrics.rougeL > 0.0
    assert metrics.sample_count == 1
