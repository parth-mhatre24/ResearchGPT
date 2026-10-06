"""Unit tests for BERT NER Service (Task 11)."""

import pytest
from backend.app.services.bert_ner_service import (
    _DEPS_AVAILABLE,
    BERTNERService,
    align_labels_with_tokens,
)

pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE, reason="Transformers & PyTorch required for BERT NER tests."
)


def test_align_labels_with_tokens():
    label_ids = [1, 5, 0]  # B-PER, B-LOC, O
    word_ids = [None, 0, 0, 1, 2, None]  # [CLS], subword1, subword2, word2, word3, [SEP]

    aligned = align_labels_with_tokens(label_ids, word_ids)
    expected = [-100, 1, -100, 5, 0, -100]
    assert aligned == expected


def test_bert_ner_predict_single_mock():
    service = BERTNERService()
    tokens = ["Alice", "visited", "Paris", "."]

    res = service.predict_single(tokens)
    assert res.tokens == tokens
    assert len(res.predicted_labels) == len(tokens)
    assert res.model_type == "bert_ner"
