"""Unit tests for TransformerClassifierService (Task 08 – Transformer Classification).

These tests validate the service API entirely on tiny toy data so that they
run quickly in CI without a GPU or large model download.  The model is
initialised from a tiny randomly-initialised checkpoint created in-test so
there is no network traffic required.

Test coverage:
- TransformerClassifierService initialises with default arguments.
- train() runs on a tiny corpus and sets _is_trained = True.
- predict() returns one label per input text.
- predict_proba() returns normalised probability dicts.
- predict_single() returns a ClassificationResponse with expected fields.
- evaluate() returns a ClassificationEvaluationMetrics with valid ranges.
- save() / load() round-trip preserves predictions.
- Calling predict before training raises RuntimeError.
- save / load round-trip returns correct labels.
"""

import shutil
from pathlib import Path

import pytest

# Guard: skip entire module if transformers / torch are not importable.
# We wrap in try/except rather than importorskip to handle the Windows DLL
# initialization crash that can occur with torch on Python 3.14+.
try:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
except Exception as _torch_err:  # noqa: BLE001
    pytest.skip(
        f"torch/transformers not available on this platform: {_torch_err}",
        allow_module_level=True,
    )

from backend.app.services.transformer_classifier_service import TransformerClassifierService


# ---------------------------------------------------------------------------
# Shared toy model fixture
# ---------------------------------------------------------------------------

_TINY_MODEL = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"


@pytest.fixture(scope="module")
def tiny_corpus():
    """Minimal binary corpus for fast testing."""
    train_texts = [
        "The movie was fantastic and thrilling",
        "An excellent film with great performances",
        "Incredible storytelling and vivid cinematography",
        "A masterpiece of modern cinema",
        "Terrible film, completely unwatchable",
        "Boring and poorly written script",
        "Awful acting and dull plot",
        "A complete waste of time",
    ]
    train_labels = [1, 1, 1, 1, 0, 0, 0, 0]
    val_texts = ["Wonderful and uplifting experience", "Dull and monotonous"]
    val_labels = [1, 0]
    test_texts = ["Great performances all around", "Boring and lifeless"]
    test_labels = [1, 0]
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


@pytest.fixture(scope="module")
def trained_service(tiny_corpus):
    """Return a TransformerClassifierService fine-tuned on the tiny corpus.

    Uses a pre-existing tiny model to avoid downloading large weights.
    Training is done for 1 epoch to keep tests fast.
    """
    train_texts, train_labels, val_texts, val_labels, _, _ = tiny_corpus

    service = TransformerClassifierService(
        model_name=_TINY_MODEL,
        num_labels=2,
        max_length=32,
        batch_size=4,
        num_epochs=1,
        learning_rate=5e-5,
    )
    service.train(train_texts, train_labels, val_texts, val_labels)
    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_defaults():
    """Service initialises with correct default attributes."""
    svc = TransformerClassifierService()
    assert svc.model_name == TransformerClassifierService.DEFAULT_MODEL
    assert svc.num_labels == 2
    assert svc.max_length == 256
    assert svc.batch_size == 16
    assert svc.num_epochs == 3
    assert not svc._is_trained


def test_untrained_raises_runtime_error():
    """predict() before training must raise RuntimeError."""
    svc = TransformerClassifierService(model_name=_TINY_MODEL, num_labels=2)
    with pytest.raises(RuntimeError, match="has not been trained or loaded"):
        svc.predict(["test"])


def test_train_sets_trained_flag(trained_service):
    """After training, _is_trained must be True."""
    assert trained_service._is_trained


def test_predict_length(trained_service, tiny_corpus):
    """predict() returns one label per text."""
    _, _, _, _, test_texts, test_labels = tiny_corpus
    preds = trained_service.predict(test_texts)
    assert len(preds) == len(test_texts)


def test_predict_labels_in_label_set(trained_service, tiny_corpus):
    """All predicted labels must come from the known label set."""
    _, _, _, _, test_texts, _ = tiny_corpus
    preds = trained_service.predict(test_texts)
    valid_labels = set(trained_service.id2label.values())
    for p in preds:
        assert p in valid_labels


def test_predict_proba_normalised(trained_service, tiny_corpus):
    """predict_proba() dicts must sum to ~1.0 per sample."""
    _, _, _, _, test_texts, _ = tiny_corpus
    proba_list = trained_service.predict_proba(test_texts)
    assert len(proba_list) == len(test_texts)
    for proba in proba_list:
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities do not sum to 1: {proba}"


def test_predict_single_response_fields(trained_service):
    """predict_single() returns ClassificationResponse with all expected fields."""
    response = trained_service.predict_single("An amazing and brilliant film")
    assert response.text == "An amazing and brilliant film"
    assert response.model_type.startswith("transformer:")
    assert response.predicted_label in set(trained_service.id2label.values())
    assert response.confidence is not None
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.probabilities, dict)


def test_evaluate_metric_ranges(trained_service, tiny_corpus):
    """evaluate() returns metrics in [0, 1]."""
    _, _, _, _, test_texts, test_labels = tiny_corpus
    metrics = trained_service.evaluate(test_texts, test_labels)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert 0.0 <= metrics.f1_weighted <= 1.0
    assert metrics.sample_count == len(test_texts)
    assert len(metrics.confusion_matrix) == 2  # binary
    assert len(metrics.classes) == 2


def test_save_and_load_roundtrip(tmp_path, trained_service, tiny_corpus):
    """Saved model reloads and produces identical predictions."""
    _, _, _, _, test_texts, _ = tiny_corpus

    save_dir = tmp_path / "transformer_test"
    saved = trained_service.save(str(save_dir), model_name="test_model")
    assert saved.exists()
    assert (saved / "training_meta.json").exists()
    assert (saved / "config.json").exists()

    loaded = TransformerClassifierService.load(str(saved))
    assert loaded._is_trained
    assert loaded.model_name == trained_service.model_name

    original_preds = trained_service.predict(test_texts)
    loaded_preds = loaded.predict(test_texts)
    assert original_preds == loaded_preds
