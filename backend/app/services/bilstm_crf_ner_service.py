"""ResearchGPT BiLSTM-CRF NER Service (Task 10).

Implements a Bidirectional LSTM with Conditional Random Field (BiLSTM-CRF)
sequence labelling architecture for Named Entity Recognition on CoNLL-2003.

Features:
- Word & character-level token embeddings
- Bidirectional LSTM sequence context encoder
- Linear-chain CRF for Viterbi decoding & negative log-likelihood training loss
- Evaluation using micro/macro entity-level Precision, Recall, and F1
- Persistence and state dictionary serialization
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("bilstm_crf_ner_service")

# ---------------------------------------------------------------------------
# Lazy import – PyTorch is required
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = "PyTorch is not installed. Install with: pip install torch"

from backend.app.models.ner import (
    CONLL_ID2LABEL,
    CONLL_LABEL2ID,
    CONLL_LABEL_NAMES,
    NEREvaluationMetrics,
    NEREntity,
    NERResponse,
)
from backend.app.services.crf_ner_service import compute_ner_metrics, extract_entities

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# PyTorch CRF Module
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class LinearCRF(nn.Module):
        """Linear-Chain Conditional Random Field (CRF) layer."""

        def __init__(self, num_tags: int):
            super().__init__()
            self.num_tags = num_tags
            # Transitions from tag j to tag i: transitions[i, j]
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.transitions, -0.1, 0.1)
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        def forward(
            self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            """Compute negative log likelihood loss."""
            gold_score = self._compute_score(emissions, tags, mask)
            forward_score = self._compute_forward_score(emissions, mask)
            return torch.mean(forward_score - gold_score)

        def _compute_score(
            self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            batch_size, seq_len, _ = emissions.shape
            score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(
                1, tags[:, 0].unsqueeze(1)
            ).squeeze(1)

            for i in range(1, seq_len):
                mask_i = mask[:, i]
                emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
                trans_score = self.transitions[tags[:, i], tags[:, i - 1]]
                score = score + (emit_score + trans_score) * mask_i

            # Add end transition
            last_indices = mask.long().sum(dim=1) - 1
            last_tags = tags.gather(1, last_indices.unsqueeze(1)).squeeze(1)
            score = score + self.end_transitions[last_tags]
            return score

        def _compute_forward_score(
            self, emissions: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            batch_size, seq_len, num_tags = emissions.shape
            init_alphas = self.start_transitions.unsqueeze(0) + emissions[:, 0]
            forward_var = init_alphas

            for i in range(1, seq_len):
                mask_i = mask[:, i].unsqueeze(1)
                emit_score = emissions[:, i].unsqueeze(2)
                trans_score = self.transitions.unsqueeze(0)
                next_tag_var = (
                    forward_var.unsqueeze(1) + trans_score + emit_score
                )
                alpha_t = torch.logsumexp(next_tag_var, dim=2)
                forward_var = torch.where(mask_i, alpha_t, forward_var)

            forward_var = forward_var + self.end_transitions.unsqueeze(0)
            return torch.logsumexp(forward_var, dim=1)

        def decode(
            self, emissions: torch.Tensor, mask: torch.Tensor
        ) -> List[List[int]]:
            """Viterbi decoding for best tag sequences."""
            batch_size, seq_len, num_tags = emissions.shape
            best_paths: List[List[int]] = []

            for b in range(batch_size):
                seq_len_b = int(mask[b].sum().item())
                if seq_len_b == 0:
                    best_paths.append([])
                    continue

                emits = emissions[b, :seq_len_b]
                viterbi_vars = self.start_transitions + emits[0]
                backpointers = []

                for t in range(1, seq_len_b):
                    next_tag_var = viterbi_vars.unsqueeze(0) + self.transitions
                    max_vars, bptrs = torch.max(next_tag_var, dim=1)
                    viterbi_vars = max_vars + emits[t]
                    backpointers.append(bptrs.tolist())

                viterbi_vars = viterbi_vars + self.end_transitions
                best_tag_id = int(torch.argmax(viterbi_vars).item())

                best_path = [best_tag_id]
                for bptrs_t in reversed(backpointers):
                    best_tag_id = bptrs_t[best_tag_id]
                    best_path.append(best_tag_id)

                best_path.reverse()
                best_paths.append(best_path)

            return best_paths

    class BiLSTMCRFModel(nn.Module):
        """PyTorch BiLSTM-CRF sequence labeller model."""

        def __init__(
            self,
            vocab_size: int,
            tag_to_ix: Dict[str, int],
            embedding_dim: int = 100,
            hidden_dim: int = 128,
            dropout: float = 0.5,
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.tag_to_ix = tag_to_ix
            self.num_tags = len(tag_to_ix)
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)
            self.bilstm = nn.LSTM(
                embedding_dim,
                hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.linear = nn.Linear(hidden_dim, self.num_tags)
            self.crf = LinearCRF(self.num_tags)

        def forward(
            self,
            inputs: torch.Tensor,
            tags: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            embeds = self.dropout(self.embedding(inputs))
            lstm_out, _ = self.bilstm(embeds)
            emissions = self.linear(self.dropout(lstm_out))
            loss = self.crf(emissions, tags, mask)
            return loss

        def decode(
            self, inputs: torch.Tensor, mask: torch.Tensor
        ) -> List[List[int]]:
            embeds = self.embedding(inputs)
            lstm_out, _ = self.bilstm(embeds)
            emissions = self.linear(lstm_out)
            return self.crf.decode(emissions, mask)


# ---------------------------------------------------------------------------
# BiLSTMCRFNERService Interface
# ---------------------------------------------------------------------------

class BiLSTMCRFNERService:
    """Service wrapper for BiLSTM-CRF NER model training and inference."""

    def __init__(self, embedding_dim: int = 100, hidden_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_to_ix: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.tag_to_ix: Dict[str, int] = CONLL_LABEL2ID
        self.ix_to_tag: Dict[int, str] = CONLL_ID2LABEL
        self.model: Optional[Any] = None
        self.is_trained: bool = False

    def _check_torch(self) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(_TORCH_IMPORT_ERROR)

    def build_vocab(self, sentences: List[List[str]]) -> None:
        """Build word vocabulary from a dataset of tokenized sentences."""
        for sent in sentences:
            for word in sent:
                w_lower = word.lower()
                if w_lower not in self.word_to_ix:
                    self.word_to_ix[w_lower] = len(self.word_to_ix)

    def _prepare_inputs(
        self, sentences: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_torch()
        max_len = max(len(s) for s in sentences) if sentences else 0
        batch_size = len(sentences)

        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for b, sent in enumerate(sentences):
            for t, word in enumerate(sent):
                input_ids[b, t] = self.word_to_ix.get(word.lower(), self.word_to_ix[UNK_TOKEN])
                mask[b, t] = True

        return input_ids, mask

    def _prepare_tags(
        self, tags_list: List[List[str]], max_len: int
    ) -> torch.Tensor:
        self._check_torch()
        batch_size = len(tags_list)
        tag_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        for b, tags in enumerate(tags_list):
            for t, tag in enumerate(tags):
                tag_ids[b, t] = self.tag_to_ix.get(tag, 0)

        return tag_ids

    def train(
        self,
        train_sentences: List[List[str]],
        train_tags: List[List[str]],
        val_sentences: Optional[List[List[str]]] = None,
        val_tags: Optional[List[List[str]]] = None,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Train the BiLSTM-CRF model on tokenized sentences and BIO tags."""
        self._check_torch()
        self.build_vocab(train_sentences)

        vocab_size = len(self.word_to_ix)
        self.model = BiLSTMCRFModel(
            vocab_size=vocab_size,
            tag_to_ix=self.tag_to_ix,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()

        start_time = time.time()
        num_samples = len(train_sentences)

        for epoch in range(epochs):
            total_loss = 0.0
            # Shuffle indices
            indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                b_sents = [train_sentences[k] for k in batch_idx]
                b_tags = [train_tags[k] for k in batch_idx]

                input_ids, mask = self._prepare_inputs(b_sents)
                tag_ids = self._prepare_tags(b_tags, input_ids.shape[1])

                optimizer.zero_grad()
                loss = self.model(input_ids, tag_ids, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * len(b_sents)

            avg_loss = total_loss / num_samples
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        training_time = time.time() - start_time
        self.is_trained = True

        val_metrics = None
        if val_sentences and val_tags:
            val_metrics = self.evaluate(val_sentences, val_tags)

        return {
            "epochs": epochs,
            "training_time_sec": round(training_time, 2),
            "vocab_size": vocab_size,
            "val_metrics": val_metrics.model_dump() if val_metrics else None,
        }

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predict list of IOB2 tag sequences for a batch of tokenized sentences."""
        self._check_torch()
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet.")

        self.model.eval()
        predicted_sequences: List[List[str]] = []

        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                b_sents = sentences[i : i + batch_size]
                input_ids, mask = self._prepare_inputs(b_sents)
                paths = self.model.decode(input_ids, mask)

                for path in paths:
                    predicted_sequences.append([self.ix_to_tag.get(tag_id, "O") for tag_id in path])

        return predicted_sequences

    def predict_single(self, tokens: List[str]) -> NERResponse:
        """Predict NER tags and extract entity spans for a single token sequence."""
        pred_labels = self.predict([tokens])[0]
        entities = extract_entities(tokens, pred_labels)
        return NERResponse(
            tokens=tokens,
            predicted_labels=pred_labels,
            entities=entities,
            model_type="bilstm_crf",
        )

    def evaluate(
        self, test_sentences: List[List[str]], test_tags: List[List[str]]
    ) -> NEREvaluationMetrics:
        """Evaluate entity-level precision, recall, and F1 on a test set."""
        pred_sequences = self.predict(test_sentences)
        return compute_ner_metrics(test_tags, pred_sequences, model_type="bilstm_crf")

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save model state dictionary and vocabulary to directory."""
        self._check_torch()
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save an untrained model.")

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch weights
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save metadata / vocab
        meta_path = path / "metadata.json"
        meta = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "word_to_ix": self.word_to_ix,
            "tag_to_ix": self.tag_to_ix,
            "model_type": "bilstm_crf",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved BiLSTM-CRF model artifact to {path}")
        return path

    def load(self, model_dir: Union[str, Path]) -> "BiLSTMCRFNERService":
        """Load model state dictionary and vocabulary from directory."""
        self._check_torch()
        path = Path(model_dir)

        meta_path = path / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.embedding_dim = meta["embedding_dim"]
        self.hidden_dim = meta["hidden_dim"]
        self.word_to_ix = meta["word_to_ix"]
        self.tag_to_ix = meta["tag_to_ix"]
        self.ix_to_tag = {int(k) if isinstance(k, int) or k.isdigit() else v: v for k, v in CONLL_ID2LABEL.items()}

        vocab_size = len(self.word_to_ix)
        self.model = BiLSTMCRFModel(
            vocab_size=vocab_size,
            tag_to_ix=self.tag_to_ix,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )

        model_path = path / "model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.is_trained = True

        logger.info(f"Loaded BiLSTM-CRF model artifact from {path}")
        return self
