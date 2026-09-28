"""ResearchGPT Transformer Classifier Service.

Fine-tunes DistilBERT (or any AutoModelForSequenceClassification model) on
a binary / multi-class text classification task using the Hugging Face
Transformers Trainer API.

Supports:
- Training from a HuggingFace model checkpoint
- Inference (single text and batch)
- Standard evaluation metrics (accuracy, macro/weighted P/R/F1)
- Artifact persistence (model checkpoint + metadata JSON)
- CPU and GPU training (auto-detects device)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("transformer_classifier_service")

# ---------------------------------------------------------------------------
# Lazy imports – only available after `pip install transformers torch`
# ---------------------------------------------------------------------------
try:
    import torch
    from datasets import Dataset
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    _DEPS_AVAILABLE = True
except ImportError as _e:  # pragma: no cover
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = str(_e)

from backend.app.models.classification import (
    ClassificationEvaluationMetrics,
    ClassificationResponse,
)


class TransformerClassifierService:
    """Service for fine-tuning and serving a transformer-based text classifier.

    Args:
        model_name: HuggingFace model hub ID (default: distilbert-base-uncased).
        num_labels: Number of output classes.
        max_length: Maximum tokenizer sequence length.
        batch_size: Per-device training batch size.
        num_epochs: Number of fine-tuning epochs.
        learning_rate: AdamW learning rate.
        output_dir: Directory for Trainer checkpoints (temp during training).
        random_state: Reproducibility seed.
    """

    DEFAULT_MODEL = "distilbert-base-uncased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_labels: int = 2,
        max_length: int = 256,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        output_dir: Optional[Union[str, Path]] = None,
        random_state: int = 42,
    ):
        if not _DEPS_AVAILABLE:
            raise ImportError(
                f"transformers and torch are required for TransformerClassifierService. "
                f"Install with: pip install transformers torch. Original error: {_IMPORT_ERROR}"
            )

        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.output_dir = Path(output_dir) if output_dir else Path("models/saved/transformer/checkpoints")
        self.random_state = random_state

        self.tokenizer = None
        self.model = None
        self.label2id: Dict[Any, int] = {}
        self.id2label: Dict[int, Any] = {}
        self._is_trained: bool = False

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_label_maps(self, labels: List[Any]) -> None:
        unique = sorted(set(labels), key=lambda x: (str(type(x)), x))
        self.label2id = {lbl: i for i, lbl in enumerate(unique)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}
        self.num_labels = len(unique)

    def _tokenize(self, examples: Dict) -> Dict:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
        )

    def _make_hf_dataset(self, texts: List[str], labels: Optional[List[Any]] = None) -> Dataset:
        data: Dict[str, List] = {"text": texts}
        if labels is not None:
            data["labels"] = [self.label2id[lbl] for lbl in labels]
        ds = Dataset.from_dict(data)
        ds = ds.map(self._tokenize, batched=True, remove_columns=["text"])
        return ds

    @staticmethod
    def _compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
        return {"accuracy": acc, "f1_weighted": f1}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_texts: List[str],
        train_labels: List[Any],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Any]] = None,
    ) -> "TransformerClassifierService":
        """Fine-tune the transformer on the supplied training data.

        Args:
            train_texts: List of raw text strings.
            train_labels: Corresponding labels (str or int).
            val_texts: Optional validation texts for early stopping.
            val_labels: Corresponding validation labels.

        Returns:
            self (for chaining).
        """
        if not train_texts:
            raise ValueError("train_texts cannot be empty")

        self._build_label_maps(train_labels)
        logger.info(
            "Fine-tuning [%s] | labels=%s | train=%d | val=%s",
            self.model_name,
            list(self.label2id.keys()),
            len(train_texts),
            len(val_texts) if val_texts else "none",
        )

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label={str(k): str(v) for k, v in self.id2label.items()},
            label2id={str(k): int(v) for k, v in self.label2id.items()},
            ignore_mismatched_sizes=True,
        )

        train_ds = self._make_hf_dataset(train_texts, train_labels)
        eval_ds = self._make_hf_dataset(val_texts, val_labels) if val_texts else None

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        use_gpu = torch.cuda.is_available()
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if eval_ds else []

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=0.01,
            eval_strategy="epoch" if eval_ds else "no",
            save_strategy="epoch" if eval_ds else "no",
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="f1_weighted" if eval_ds else None,
            seed=self.random_state,
            use_cpu=not use_gpu,          # replaces deprecated no_cuda
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,  # replaces deprecated tokenizer=
            data_collator=collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks if callbacks else None,
        )

        trainer.train()
        self._is_trained = True
        logger.info("Fine-tuning complete.")
        return self

    def predict(self, texts: List[str]) -> List[Any]:
        """Return predicted labels for a list of texts."""
        self._check_trained()
        logits = self._run_inference(texts)
        pred_ids = np.argmax(logits, axis=-1)
        return [self.id2label[int(i)] for i in pred_ids]

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Return softmax probability distributions over all classes."""
        self._check_trained()
        logits = self._run_inference(texts)
        exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        results = []
        for row in probs:
            results.append({str(self.id2label[i]): float(p) for i, p in enumerate(row)})
        return results

    def predict_single(self, text: str) -> ClassificationResponse:
        """Classify a single text string."""
        pred = self.predict([text])[0]
        probs = self.predict_proba([text])[0]
        confidence = probs.get(str(pred), max(probs.values()))
        return ClassificationResponse(
            text=text,
            predicted_label=pred,
            probabilities=probs,
            confidence=confidence,
            model_type=f"transformer:{self.model_name}",
        )

    def evaluate(
        self,
        texts: List[str],
        true_labels: List[Any],
    ) -> ClassificationEvaluationMetrics:
        """Compute standard classification metrics on a held-out set."""
        self._check_trained()
        preds = self.predict(texts)
        classes = list(self.label2id.keys())
        classes_str = [str(c) for c in classes]

        acc = float(accuracy_score(true_labels, preds))
        p_macro = float(precision_score(true_labels, preds, average="macro", zero_division=0))
        r_macro = float(recall_score(true_labels, preds, average="macro", zero_division=0))
        f1_macro = float(f1_score(true_labels, preds, average="macro", zero_division=0))
        p_weighted = float(precision_score(true_labels, preds, average="weighted", zero_division=0))
        r_weighted = float(recall_score(true_labels, preds, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(true_labels, preds, average="weighted", zero_division=0))

        cm = confusion_matrix(true_labels, preds, labels=classes).tolist()
        report = classification_report(
            true_labels, preds, labels=classes, target_names=classes_str,
            output_dict=True, zero_division=0,
        )

        return ClassificationEvaluationMetrics(
            accuracy=acc,
            precision_macro=p_macro,
            recall_macro=r_macro,
            f1_macro=f1_macro,
            precision_weighted=p_weighted,
            recall_weighted=r_weighted,
            f1_weighted=f1_weighted,
            confusion_matrix=cm,
            classes=classes_str,
            sample_count=len(texts),
            classification_report=report,
        )

    def save(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist the fine-tuned model, tokenizer, and metadata JSON."""
        self._check_trained()
        save_path = Path(output_dir) / (model_name or f"transformer_{self.model_name.replace('/', '_')}")
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        meta: Dict[str, Any] = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "random_state": self.random_state,
            "label2id": {str(k): v for k, v in self.label2id.items()},
            "id2label": {str(k): str(v) for k, v in self.id2label.items()},
        }
        if extra_metadata:
            meta["extra"] = extra_metadata

        meta_file = save_path / "training_meta.json"
        with open(meta_file, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Saved transformer checkpoint to %s", save_path)
        return save_path

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "TransformerClassifierService":
        """Load a fine-tuned checkpoint from disk."""
        if not _DEPS_AVAILABLE:
            raise ImportError("transformers and torch are required.")

        path = Path(model_dir)
        meta_file = path / "training_meta.json"
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        service = cls(
            model_name=meta.get("model_name", cls.DEFAULT_MODEL),
            num_labels=meta.get("num_labels", 2),
            max_length=meta.get("max_length", 256),
            batch_size=meta.get("batch_size", 16),
            num_epochs=meta.get("num_epochs", 3),
            learning_rate=meta.get("learning_rate", 2e-5),
            warmup_ratio=meta.get("warmup_ratio", 0.1),
            random_state=meta.get("random_state", 42),
        )

        service.tokenizer = AutoTokenizer.from_pretrained(str(path))
        service.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        service.label2id = {
            (int(k) if k.isdigit() else k): v
            for k, v in meta.get("label2id", {}).items()
        }
        service.id2label = {
            int(k): (int(v) if str(v).isdigit() else v)
            for k, v in meta.get("id2label", {}).items()
        }
        service._is_trained = True
        return service

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "TransformerClassifierService has not been trained or loaded yet. "
                "Call .train() or .load() first."
            )

    def _run_inference(self, texts: List[str]) -> np.ndarray:
        """Run a forward pass and return raw logits as a numpy array."""
        self.model.eval()
        device = next(self.model.parameters()).device

        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)

        return outputs.logits.cpu().numpy()
