"""Unit tests for BiLSTM-CRF NER Service (Task 10)."""

import pytest
from backend.app.services.bilstm_crf_ner_service import (
    _TORCH_AVAILABLE,
    BiLSTMCRFNERService,
)

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch is required for BiLSTM-CRF NER tests."
)


@pytest.fixture
def dummy_conll_data():
    tokens = [
        ["Jane", "lives", "in", "London", "."],
        ["Google", "is", "a", "tech", "company", "."],
    ]
    tags = [
        ["B-PER", "O", "O", "B-LOC", "O"],
        ["B-ORG", "O", "O", "O", "O", "O"],
    ]
    return tokens, tags


def test_bilstm_crf_vocab_building(dummy_conll_data):
    tokens, tags = dummy_conll_data
    service = BiLSTMCRFNERService()
    service.build_vocab(tokens)

    assert "<PAD>" in service.word_to_ix
    assert "<UNK>" in service.word_to_ix
    assert "jane" in service.word_to_ix
    assert "london" in service.word_to_ix


def test_bilstm_crf_training_and_prediction(dummy_conll_data, tmp_path):
    tokens, tags = dummy_conll_data
    service = BiLSTMCRFNERService(embedding_dim=16, hidden_dim=32)

    train_res = service.train(
        train_sentences=tokens,
        train_tags=tags,
        val_sentences=tokens,
        val_tags=tags,
        epochs=2,
        lr=1e-2,
        batch_size=2,
    )

    assert service.is_trained
    assert train_res["epochs"] == 2
    assert "val_metrics" in train_res

    # Test single prediction
    single_res = service.predict_single(tokens[0])
    assert single_res.tokens == tokens[0]
    assert len(single_res.predicted_labels) == len(tokens[0])

    # Test serialization roundtrip
    save_path = tmp_path / "bilstm_model"
    service.save(save_path)
    assert (save_path / "model.pt").exists()
    assert (save_path / "metadata.json").exists()

    loaded_service = BiLSTMCRFNERService()
    loaded_service.load(save_path)
    assert loaded_service.is_trained

    preds = loaded_service.predict([tokens[0]])
    assert len(preds[0]) == len(tokens[0])
