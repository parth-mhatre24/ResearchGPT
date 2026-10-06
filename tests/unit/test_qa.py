"""Unit tests for Extractive Question Answering Service (Task 16 / FR-09)."""

import pytest
from backend.app.models.qa import QARequest
from backend.app.services.qa_service import (
    QuestionAnsweringService,
    _DEPS_AVAILABLE,
    compute_f1_and_em,
)


def test_compute_f1_and_em_exact():
    em, f1, p, r = compute_f1_and_em("Attention mechanisms", "attention mechanisms")
    assert em == 1.0
    assert f1 == 1.0
    assert p == 1.0
    assert r == 1.0


def test_compute_f1_and_em_partial():
    em, f1, p, r = compute_f1_and_em("solely on attention mechanisms", "attention mechanisms")
    assert em == 0.0
    assert f1 > 0.60
    assert r == 1.0
    assert p < 1.0


def test_compute_f1_and_em_disjoint():
    em, f1, p, r = compute_f1_and_em("convolutional filters", "recurrent memory")
    assert em == 0.0
    assert f1 == 0.0
    assert p == 0.0
    assert r == 0.0


def test_qa_service_empty_context():
    service = QuestionAnsweringService()
    resp = service.answer_question(question="What is BERT?", context="")
    assert resp.answer == ""
    assert resp.confidence_score == 0.0


def test_qa_service_empty_question_raises():
    service = QuestionAnsweringService()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.answer_question(question="  ", context="Some valid context text.")


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="PyTorch/Transformers required for QuestionAnsweringService")
def test_qa_service_extractive_answer():
    service = QuestionAnsweringService()
    context = (
        "The Transformer architecture was introduced by Vaswani et al. in 2017. "
        "It achieves high performance on machine translation benchmarks."
    )
    resp = service.answer_question(
        question="Who introduced the Transformer architecture?",
        context=context,
    )

    assert "Vaswani" in resp.answer
    assert resp.confidence_score > 0.30
    assert resp.start_char >= 0
    assert resp.end_char > resp.start_char


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="PyTorch/Transformers required for QuestionAnsweringService")
def test_qa_service_batch_and_evaluation():
    service = QuestionAnsweringService()
    qa_pairs = [
        {
            "question": "What year was the Transformer introduced?",
            "context": "The Transformer was introduced in 2017.",
            "answers": ["2017"],
        },
        {
            "question": "What does CNN stand for?",
            "context": "A Convolutional Neural Network (CNN) is a class of deep neural network.",
            "answers": ["Convolutional Neural Network"],
        },
    ]

    metrics = service.evaluate_qa(qa_pairs)
    assert metrics.sample_count == 2
    assert metrics.exact_match >= 50.0
    assert metrics.token_f1 >= 50.0
