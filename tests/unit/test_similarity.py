"""Unit tests for Semantic Similarity Service (Task 13 / FR-07)."""

import pytest
from backend.app.services.similarity_service import (
    SemanticSimilarityService,
    _TRANSFORMERS_AVAILABLE,
)


def test_similarity_tfidf_identical_sentences():
    service = SemanticSimilarityService()
    text = "Machine learning algorithms learn patterns from data."
    resp = service.compute_similarity_pair(text, text, method="tfidf")

    assert resp.similarity_score >= 0.99
    assert resp.score_stsb_scale >= 4.95
    assert resp.method == "tfidf"


def test_similarity_tfidf_unrelated_sentences():
    service = SemanticSimilarityService()
    text_a = "The astronomical observatory discovered a new exoplanet in the Andromeda galaxy."
    text_b = "Italian cuisine features pasta, tomatoes, olive oil, and mozzarella cheese."
    resp = service.compute_similarity_pair(text_a, text_b, method="tfidf")

    assert resp.similarity_score <= 0.15
    assert resp.score_stsb_scale <= 1.0


def test_similarity_batch_computation():
    service = SemanticSimilarityService()
    pairs = [
        ("A dog is running in the park.", "A puppy runs through the grass."),
        ("Deep learning uses neural networks.", "Baking bread requires flour and yeast."),
    ]
    resp = service.compute_similarity_batch(pairs, method="tfidf")

    assert resp.total_pairs == 2
    assert len(resp.results) == 2
    assert 0.0 <= resp.mean_similarity <= 1.0


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="Transformers / PyTorch required for Sentence-BERT")
def test_similarity_sentence_bert_identical_and_paraphrase():
    service = SemanticSimilarityService()
    text_a = "A plane is landing on the runway."
    text_b = "An airplane is touching down on the airstrip."
    resp = service.compute_similarity_pair(text_a, text_b, method="sentence_bert")

    assert 0.70 <= resp.similarity_score <= 1.0
    assert 3.5 <= resp.score_stsb_scale <= 5.0
    assert resp.method == "sentence_bert"


def test_similarity_evaluate_stsb_metrics_range():
    service = SemanticSimilarityService()
    texts_a = [
        "A man is playing soccer.",
        "A woman is reading a book.",
        "The sun is shining brightly.",
        "A chef is cooking dinner.",
    ]
    texts_b = [
        "A person is kicking a football.",
        "Someone is turning pages in a novel.",
        "It is a bright sunny day outside.",
        "A cook is preparing a meal in the kitchen.",
    ]
    gold_scores = [4.8, 4.5, 4.7, 4.9]

    metrics = service.evaluate_stsb(texts_a, texts_b, gold_scores, method="tfidf")
    assert -1.0 <= metrics.pearson_correlation <= 1.0
    assert -1.0 <= metrics.spearman_correlation <= 1.0
    assert metrics.mse >= 0.0
    assert metrics.sample_count == 4
