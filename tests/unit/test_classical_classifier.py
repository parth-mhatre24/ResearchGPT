"""Unit tests for ClassicalClassifierService (TF-IDF + Naive Bayes, Logistic Regression, SVM)."""

import pytest
from pathlib import Path
from backend.app.services.classical_classifier_service import ClassicalClassifierService


@pytest.fixture
def sample_corpus():
    """Simple toy binary classification corpus."""
    train_texts = [
        "This research paper is outstanding and novel",
        "Excellent contribution to natural language processing",
        "Great results and comprehensive benchmarks",
        "Very insightful analysis and clear methodology",
        "Flawed methodology and terrible experimental setup",
        "Poor results without adequate baselines",
        "Disappointing paper with unsupported claims",
        "Complete failure in reproducibility and evaluation",
    ]
    train_labels = [1, 1, 1, 1, 0, 0, 0, 0]

    test_texts = [
        "Outstanding benchmarks and great methodology",
        "Poor experimental setup and flawed claims",
    ]
    test_labels = [1, 0]

    return train_texts, train_labels, test_texts, test_labels


def test_invalid_model_type():
    """Verify ValueError is raised on unsupported model type."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        ClassicalClassifierService(model_type="gradient_boosting")


@pytest.mark.parametrize("model_type", ["naive_bayes", "logistic_regression", "svm"])
def test_train_and_predict(model_type, sample_corpus):
    """Test training and inference across all three classical classifiers."""
    train_texts, train_labels, test_texts, test_labels = sample_corpus

    service = ClassicalClassifierService(
        model_type=model_type,
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
    )
    service.train(train_texts, train_labels)

    predictions = service.predict(test_texts)
    assert len(predictions) == len(test_labels)
    assert predictions == [1, 0]


@pytest.mark.parametrize("model_type", ["naive_bayes", "logistic_regression", "svm"])
def test_predict_proba_and_single(model_type, sample_corpus):
    """Test probability and confidence outputs across all model types."""
    train_texts, train_labels, test_texts, _ = sample_corpus

    service = ClassicalClassifierService(model_type=model_type)
    service.train(train_texts, train_labels)

    # Probabilities
    probs = service.predict_proba(test_texts)
    assert len(probs) == len(test_texts)
    for p in probs:
        assert "0" in p and "1" in p
        assert pytest.approx(p["0"] + p["1"], abs=1e-3) == 1.0

    # Single prediction
    response = service.predict_single(test_texts[0])
    assert response.predicted_label == 1
    assert response.confidence is not None
    assert 0.0 <= response.confidence <= 1.0
    assert response.model_type == model_type


def test_evaluate_metrics(sample_corpus):
    """Verify evaluation metric calculations."""
    train_texts, train_labels, test_texts, test_labels = sample_corpus

    service = ClassicalClassifierService(model_type="logistic_regression")
    service.train(train_texts, train_labels)

    metrics = service.evaluate(test_texts, test_labels)

    assert metrics.sample_count == len(test_texts)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert 0.0 <= metrics.f1_weighted <= 1.0
    assert len(metrics.confusion_matrix) == 2
    assert len(metrics.classes) == 2


def test_save_and_load(tmp_path: Path, sample_corpus):
    """Test serializing to disk and reloading the classifier."""
    train_texts, train_labels, test_texts, test_labels = sample_corpus

    service = ClassicalClassifierService(model_type="svm", c_param=0.5)
    service.train(train_texts, train_labels)

    save_file = service.save(tmp_path, model_name="test_svm")
    assert save_file.exists()
    assert (tmp_path / "test_svm_meta.json").exists()

    loaded = ClassicalClassifierService.load(save_file)
    assert loaded.model_type == "svm"
    assert loaded.c_param == 0.5
    assert loaded.predict(test_texts) == service.predict(test_texts)


def test_untrained_error_handling():
    """Verify RuntimeError is raised when predicting before training."""
    service = ClassicalClassifierService(model_type="naive_bayes")
    with pytest.raises(RuntimeError, match="has not been trained"):
        service.predict(["test text"])
