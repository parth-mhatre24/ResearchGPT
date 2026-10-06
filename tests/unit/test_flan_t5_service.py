"""Unit tests for FLAN-T5 Generative QA Service."""

import pytest
from backend.app.services.flan_t5_service import FlanT5GenerativeService, _DEPS_AVAILABLE


@pytest.fixture
def flan_service():
    return FlanT5GenerativeService()


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for FlanT5GenerativeService")
def test_flan_t5_generation_basic(flan_service):
    context = (
        "The Transformer architecture was introduced in 2017 by Vaswani et al. "
        "It replaces recurrent neural networks with multi-head self-attention mechanisms."
    )
    question = "Who introduced the Transformer architecture and in what year?"

    ans = flan_service.generate_answer(question=question, contexts=context)
    assert isinstance(ans, str)
    assert len(ans) > 0
    assert "Vaswani" in ans or "2017" in ans


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Transformers / PyTorch required for FlanT5GenerativeService")
def test_flan_t5_multiple_context_passages(flan_service):
    passages = [
        "BERT stands for Bidirectional Encoder Representations from Transformers.",
        "It was created by Devlin et al. at Google AI Language in 2018.",
    ]
    question = "What does BERT stand for?"

    ans = flan_service.generate_answer(question=question, contexts=passages)
    assert "Bidirectional" in ans or "Transformers" in ans


def test_flan_t5_empty_inputs(flan_service):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        flan_service.generate_answer(question="   ", contexts="some context")

    ans = flan_service.generate_answer(question="Valid question", contexts="")
    assert "could not find sufficient relevant evidence" in ans
