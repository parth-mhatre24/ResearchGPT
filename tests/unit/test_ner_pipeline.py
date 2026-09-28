"""Unit tests for CRFNERService and NER dataset pipeline (Task 09).

Coverage:
- sentence_to_features() produces the expected feature keys.
- extract_entities() correctly parses BIO sequences into entity spans.
- compute_ner_metrics() produces correct P/R/F1 on a known example.
- DatasetService.load_ner_data() loads CoNLL-2003 and returns the right shapes.
- CRFNERService.train() fits without error on toy data.
- CRFNERService.predict_sequence() returns one label per token.
- CRFNERService.predict_single() returns NERResponse with correct fields.
- CRFNERService.evaluate() returns NEREvaluationMetrics in valid ranges.
- CRFNERService.save() / .load() round-trip preserves predictions.
- Calling predict before training raises RuntimeError.
"""

from pathlib import Path

import pytest

pytest.importorskip("sklearn_crfsuite")

from backend.app.models.ner import (
    CONLL_ID2LABEL,
    CONLL_LABEL_NAMES,
    NEREvaluationMetrics,
    NEREntity,
    NERResponse,
)
from backend.app.services.crf_ner_service import (
    CRFNERService,
    compute_ner_metrics,
    extract_entities,
    sentence_to_features,
)
from backend.app.services.dataset_service import DatasetService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_sentences():
    """Mini NER corpus: 10 sentences with simple entities."""
    # fmt: off
    token_sequences = [
        ["London",   "is",   "the",  "capital",  "of",   "England",  "."],
        ["Angela",   "Merkel", "visited",  "Berlin", "last",  "week",    "."],
        ["Apple",    "Inc.",   "was",  "founded",  "by",   "Steve",    "Jobs",  "."],
        ["The",      "United", "States",  "won",     "the",  "Olympic",  "Games",  "."],
        ["IBM",      "acquired", "Red",   "Hat",     "for",  "34",       "billion", "."],
        ["Paris",    "is",   "known",  "for",    "the",  "Eiffel",   "Tower",  "."],
        ["Barack",   "Obama",  "was",   "born",    "in",   "Hawaii",   "."],
        ["Google",   "released", "TensorFlow", "2", ".", ],
        ["Microsoft", "bought", "LinkedIn",  "in",   "2016",  "."],
        ["Rome",     "was",   "not",    "built",   "in",   "a",        "day",   "."],
    ]
    label_sequences = [
        ["B-LOC",  "O",    "O",    "O",     "O",   "B-LOC", "O"],
        ["B-PER",  "I-PER","O",    "B-LOC", "O",   "O",     "O"],
        ["B-ORG",  "I-ORG","O",    "O",     "O",   "B-PER", "I-PER", "O"],
        ["O",      "B-LOC","I-LOC","O",     "O",   "B-MISC","I-MISC","O"],
        ["B-ORG",  "O",    "B-ORG","I-ORG", "O",   "O",     "O",     "O"],
        ["B-LOC",  "O",    "O",    "O",     "O",   "B-LOC", "I-LOC", "O"],
        ["B-PER",  "I-PER","O",    "O",     "O",   "B-LOC", "O"],
        ["B-ORG",  "O",    "B-MISC","O",    "O"],
        ["B-ORG",  "O",    "B-ORG", "O",    "O",   "O"],
        ["B-LOC",  "O",    "O",     "O",    "O",   "O",     "O",     "O"],
    ]
    # fmt: on
    return token_sequences, label_sequences


@pytest.fixture(scope="module")
def trained_crf(toy_sentences):
    token_sequences, label_sequences = toy_sentences
    service = CRFNERService(max_iterations=50)
    service.train(token_sequences, label_sequences)
    return service


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

def test_sentence_to_features_length():
    """sentence_to_features returns one dict per token."""
    tokens = ["London", "is", "the", "capital"]
    feats = sentence_to_features(tokens)
    assert len(feats) == len(tokens)


def test_sentence_to_features_keys():
    """Feature dict contains mandatory keys for first token."""
    tokens = ["London", "is", "the"]
    feat = sentence_to_features(tokens)[0]
    assert "bias" in feat
    assert "word.lower" in feat
    assert "word.istitle" in feat
    assert "BOS" in feat  # first token


def test_sentence_to_features_eos():
    """Last token should have EOS marker."""
    tokens = ["Hello", "World"]
    feat = sentence_to_features(tokens)[-1]
    assert "EOS" in feat


# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------

def test_extract_entities_simple():
    """Single PER entity extracted from BIO sequence."""
    tokens = ["Angela", "Merkel", "visited", "Berlin", "."]
    labels = ["B-PER",  "I-PER",  "O",       "B-LOC",  "O"]
    entities = extract_entities(tokens, labels)
    assert len(entities) == 2
    per = entities[0]
    assert per.label == "PER"
    assert per.text == "Angela Merkel"
    assert per.start_token == 0
    assert per.end_token == 1


def test_extract_entities_no_entities():
    """All-O sequence returns empty list."""
    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    labels = ["O"] * len(tokens)
    assert extract_entities(tokens, labels) == []


def test_extract_entities_single_token():
    """Single-token entity correctly captured."""
    tokens = ["London", "is", "famous", "."]
    labels = ["B-LOC", "O", "O", "O"]
    entities = extract_entities(tokens, labels)
    assert len(entities) == 1
    assert entities[0].label == "LOC"
    assert entities[0].text == "London"
    assert entities[0].start_token == entities[0].end_token == 0


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

def test_compute_ner_metrics_perfect():
    """Perfect predictions should give F1 = 1.0."""
    true = [["B-PER", "I-PER", "O", "B-LOC"]]
    metrics = compute_ner_metrics(true, true, model_type="crf")
    assert metrics.f1_micro == pytest.approx(1.0)
    assert metrics.token_accuracy == pytest.approx(1.0)


def test_compute_ner_metrics_all_wrong():
    """Completely wrong predictions should give F1 = 0.0."""
    true = [["B-PER", "O"]]
    pred = [["O",     "B-LOC"]]
    metrics = compute_ner_metrics(true, pred, model_type="crf")
    assert metrics.f1_micro == pytest.approx(0.0)


def test_compute_ner_metrics_ranges(toy_sentences):
    """Metrics computed on toy data must all lie in [0, 1]."""
    token_sequences, label_sequences = toy_sentences
    # Use label sequences as both true and pred for a deterministic check
    metrics = compute_ner_metrics(label_sequences, label_sequences, model_type="test")
    assert 0.0 <= metrics.precision_micro <= 1.0
    assert 0.0 <= metrics.recall_micro <= 1.0
    assert 0.0 <= metrics.f1_micro <= 1.0
    assert 0.0 <= metrics.token_accuracy <= 1.0
    assert metrics.sample_count == len(label_sequences)


# ---------------------------------------------------------------------------
# DatasetService NER loader tests
# ---------------------------------------------------------------------------

class TestDatasetServiceNER:
    def test_load_ner_data_shapes(self):
        """load_ner_data() returns lists of lists of correct types."""
        svc = DatasetService()
        # Use a small slice via direct parquet read to avoid long waits
        token_seqs, label_seqs = svc.load_ner_data("conll2003", "validation")

        assert isinstance(token_seqs, list)
        assert isinstance(label_seqs, list)
        assert len(token_seqs) == len(label_seqs)
        assert len(token_seqs) > 0

        # Spot-check first sentence
        assert isinstance(token_seqs[0], list)
        assert isinstance(label_seqs[0], list)
        assert len(token_seqs[0]) == len(label_seqs[0])

    def test_load_ner_data_labels_are_valid(self):
        """All returned labels must be valid CoNLL-2003 IOB2 strings."""
        svc = DatasetService()
        _, label_seqs = svc.load_ner_data("conll2003", "validation")

        valid = set(CONLL_LABEL_NAMES)
        for seq in label_seqs[:50]:  # sample first 50 sentences
            for lbl in seq:
                assert lbl in valid, f"Unknown label: {lbl}"

    def test_load_ner_data_unknown_dataset_raises(self):
        """load_ner_data() on a non-NER dataset must raise KeyError."""
        svc = DatasetService()
        with pytest.raises((KeyError, FileNotFoundError)):
            svc.load_ner_data("imdb", "train")


# ---------------------------------------------------------------------------
# CRF service API tests
# ---------------------------------------------------------------------------

class TestCRFNERService:

    def test_untrained_raises(self):
        """predict_sequence() before training must raise RuntimeError."""
        svc = CRFNERService()
        with pytest.raises(RuntimeError, match="has not been trained or loaded"):
            svc.predict_sequence(["London", "is", "great"])

    def test_train_sets_flag(self, trained_crf):
        """After training, _is_trained must be True."""
        assert trained_crf._is_trained

    def test_predict_sequence_length(self, trained_crf, toy_sentences):
        """predict_sequence() returns one label per input token."""
        token_sequences, _ = toy_sentences
        for tokens in token_sequences:
            preds = trained_crf.predict_sequence(tokens)
            assert len(preds) == len(tokens)

    def test_predict_labels_are_valid(self, trained_crf, toy_sentences):
        """All predicted labels must be valid IOB2 labels."""
        token_sequences, _ = toy_sentences
        valid = set(CONLL_LABEL_NAMES)
        for tokens in token_sequences:
            for lbl in trained_crf.predict_sequence(tokens):
                assert lbl in valid, f"Invalid label: {lbl}"

    def test_predict_single_response(self, trained_crf):
        """predict_single() returns a NERResponse with expected structure."""
        text = "Barack Obama visited London yesterday"
        response = trained_crf.predict_single(text)

        assert isinstance(response, NERResponse)
        assert response.model_type == "crf"
        assert response.tokens == text.split()
        assert len(response.predicted_labels) == len(response.tokens)
        assert isinstance(response.entities, list)

    def test_evaluate_metric_ranges(self, trained_crf, toy_sentences):
        """evaluate() returns metrics all within [0, 1]."""
        token_sequences, label_sequences = toy_sentences
        metrics = trained_crf.evaluate(token_sequences, label_sequences)

        assert isinstance(metrics, NEREvaluationMetrics)
        assert 0.0 <= metrics.precision_micro <= 1.0
        assert 0.0 <= metrics.recall_micro <= 1.0
        assert 0.0 <= metrics.f1_micro <= 1.0
        assert 0.0 <= metrics.token_accuracy <= 1.0
        assert metrics.sample_count == len(token_sequences)
        assert metrics.model_type == "crf"

    def test_save_and_load_roundtrip(self, tmp_path, trained_crf, toy_sentences):
        """Save + load produces identical label predictions."""
        token_sequences, _ = toy_sentences

        model_file = trained_crf.save(tmp_path, model_name="test_crf")
        assert model_file.exists()
        assert (tmp_path / "test_crf_meta.json").exists()

        loaded = CRFNERService.load(model_file)
        assert loaded._is_trained

        for tokens in token_sequences:
            orig = trained_crf.predict_sequence(tokens)
            reloaded = loaded.predict_sequence(tokens)
            assert orig == reloaded, f"Prediction mismatch for: {tokens}"
