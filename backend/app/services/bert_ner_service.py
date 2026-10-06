"""ResearchGPT BERT NER Service (Task 11).

Fine-tunes a pre-trained Transformer model (e.g. `dslim/bert-base-NER` or `bert-base-cased`)
for token classification / sequence labelling on CoNLL-2003.

Features:
- Subword/WordPiece tokenization alignment with label sequence masking (-100)
- Trainer API fine-tuning with evaluation on validation entity F1 score
- Subword prediction decoding back to word-level tokens
- Entity span extraction and evaluation using entity-level Precision, Recall, F1
- Artifact serialization (tokenizer + model state)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("bert_ner_service")

# ---------------------------------------------------------------------------
# Lazy imports – transformers and torch required
# ---------------------------------------------------------------------------
try:
    import torch
    import transformers.utils as _u
    import transformers.utils.import_utils as _iu
    _iu.is_datasets_available = lambda: False
    _u.is_datasets_available = lambda: False

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    _DEPS_AVAILABLE = True
    _DEPS_IMPORT_ERROR = ""
except Exception as _e:
    _DEPS_AVAILABLE = False
    _DEPS_IMPORT_ERROR = f"transformers and torch are required. Run: pip install transformers torch. Error: {_e}"

from backend.app.models.ner import (
    CONLL_ID2LABEL,
    CONLL_LABEL2ID,
    CONLL_LABEL_NAMES,
    NEREvaluationMetrics,
    NEREntity,
    NERResponse,
)
from backend.app.services.crf_ner_service import compute_ner_metrics, extract_entities


def align_labels_with_tokens(
    labels: List[int], word_ids: List[Optional[int]]
) -> List[int]:
    """Align word-level label IDs with tokenizer subword tokens."""
    aligned_labels: List[int] = []
    current_word: Optional[int] = None

    for word_id in word_ids:
        if word_id is None:
            # Special tokens ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)
        elif word_id != current_word:
            # First subword token of a word
            current_word = word_id
            aligned_labels.append(labels[word_id] if word_id < len(labels) else 0)
        else:
            # Subsequent subwords of a word
            aligned_labels.append(-100)

    return aligned_labels


class BERTNERService:
    """Service for fine-tuning and serving a Transformer token classifier for NER."""

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.label2id: Dict[str, int] = CONLL_LABEL2ID
        self.id2label: Dict[int, str] = CONLL_ID2LABEL
        self.is_trained: bool = False

    def _check_deps(self) -> None:
        if not _DEPS_AVAILABLE:
            raise RuntimeError(_DEPS_IMPORT_ERROR)

    def _init_components(self) -> None:
        self._check_deps()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )

    def train(
        self,
        train_sentences: List[List[str]],
        train_tags: List[List[str]],
        val_sentences: Optional[List[List[str]]] = None,
        val_tags: Optional[List[List[str]]] = None,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Fine-tune BERT for NER on tokenized sentences and tag sequences."""
        self._init_components()

        class TorchNERDataset(torch.utils.data.Dataset):
            def __init__(inner_self, sentences: List[List[str]], tags: List[List[str]], tokenizer: Any, label2id: Dict[str, int], max_length: int = 128):
                inner_self.items = []
                for words, tag_seq in zip(sentences, tags):
                    tokenized = tokenizer(
                        words,
                        is_split_into_words=True,
                        truncation=True,
                        max_length=max_length,
                    )
                    label_ids = [label2id.get(t, 0) for t in tag_seq]
                    word_ids = tokenized.word_ids()
                    aligned = align_labels_with_tokens(label_ids, word_ids)
                    item = {k: v for k, v in tokenized.items()}
                    item["labels"] = aligned
                    inner_self.items.append(item)

            def __len__(inner_self):
                return len(inner_self.items)

            def __getitem__(inner_self, idx):
                return inner_self.items[idx]

        train_encoded = TorchNERDataset(train_sentences, train_tags, self.tokenizer, self.label2id)

        val_encoded = None
        if val_sentences and val_tags:
            val_encoded = TorchNERDataset(val_sentences, val_tags, self.tokenizer, self.label2id)

        output_dir = Path("models/saved/ner/bert_tmp")
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch" if val_encoded else "no",
            save_strategy="epoch" if val_encoded else "no",
            logging_steps=10,
            report_to="none",
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=val_encoded,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        self.is_trained = True

        val_metrics = None
        if val_sentences and val_tags:
            val_metrics = self.evaluate(val_sentences, val_tags)

        return {
            "epochs": epochs,
            "training_time_sec": round(training_time, 2),
            "val_metrics": val_metrics.model_dump() if val_metrics else None,
        }

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predict list of IOB2 tags for a batch of tokenized sentences."""
        self._init_components()

        predicted_sequences: List[List[str]] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        for tokens in sentences:
            inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

            word_ids = inputs.word_ids(batch_index=0)
            word_labels: List[str] = []
            prev_word: Optional[int] = None

            for idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id != prev_word:
                    prev_word = word_id
                    pred_id = int(predictions[idx])
                    word_labels.append(self.id2label.get(pred_id, "O"))

            # Handle edge cases where tokenizer truncates or drops tokens
            while len(word_labels) < len(tokens):
                word_labels.append("O")
            word_labels = word_labels[: len(tokens)]

            predicted_sequences.append(word_labels)

        return predicted_sequences

    def predict_single(self, tokens: List[str]) -> NERResponse:
        """Predict NER tags and extract entity spans for a single sentence."""
        pred_labels = self.predict([tokens])[0]
        entities = extract_entities(tokens, pred_labels)
        return NERResponse(
            tokens=tokens,
            predicted_labels=pred_labels,
            entities=entities,
            model_type="bert_ner",
        )

    def evaluate(
        self, test_sentences: List[List[str]], test_tags: List[List[str]]
    ) -> NEREvaluationMetrics:
        """Evaluate entity-level precision, recall, and F1 score on a test set."""
        pred_sequences = self.predict(test_sentences)
        return compute_ner_metrics(test_tags, pred_sequences, model_type="bert_ner")

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save model and tokenizer weights to directory."""
        self._check_deps()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        meta_path = path / "metadata.json"
        meta = {
            "model_name": self.model_name,
            "label2id": self.label2id,
            "model_type": "bert_ner",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved BERT NER artifact to {path}")
        return path

    def load(self, model_dir: Union[str, Path]) -> "BERTNERService":
        """Load model and tokenizer weights from directory."""
        self._check_deps()
        path = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.is_trained = True

        logger.info(f"Loaded BERT NER artifact from {path}")
        return self
