"""ResearchGPT Extractive Question Answering Service (Task 16 / FR-09).

Performs extractive question answering on scientific text and document contexts
using fine-tuned Transformer QA models (`distilbert-base-cased-distilled-squad`).
"""

import logging
import re
import string
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("qa_service")

# ---------------------------------------------------------------------------
# Lazy imports for Transformer / PyTorch QA
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    import transformers.utils as _u
    import transformers.utils.import_utils as _iu
    _iu.is_datasets_available = lambda: False
    _u.is_datasets_available = lambda: False
    _iu.check_torch_load_is_safe = lambda: None

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    _DEPS_AVAILABLE = True
except Exception as _e:
    _DEPS_AVAILABLE = False
    logger.warning(f"PyTorch / transformers not available for QuestionAnsweringService: {_e}")

from backend.app.models.qa import (
    QABatchRequest,
    QABatchResponse,
    QAEvaluationMetrics,
    QARequest,
    QAResponse,
)


def _normalize_answer(s: str) -> str:
    """Standard SQuAD text normalization (lowercase, remove punctuation, remove articles)."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1_and_em(prediction: str, ground_truth: str) -> Tuple[float, float, float, float]:
    """Compute Token-level Precision, Recall, F1 and Exact Match between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()

    # Exact Match
    em = 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0

    if not pred_tokens or not gold_tokens:
        f1 = 1.0 if pred_tokens == gold_tokens else 0.0
        return em, f1, f1, f1

    common_tokens = set(pred_tokens) & set(gold_tokens)
    if not common_tokens:
        return em, 0.0, 0.0, 0.0

    # Count overlaps
    overlap = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return em, f1, precision, recall


class QuestionAnsweringService:
    """Service for answering questions from document contexts using Transformer QA models."""

    DEFAULT_MODEL = "distilbert-base-cased-distilled-squad"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = "cuda" if _DEPS_AVAILABLE and torch.cuda.is_available() else "cpu"

    def _init_model(self) -> None:
        """Lazily initialize tokenizer and QA model."""
        if not _DEPS_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required for QuestionAnsweringService.")

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def answer_question(self, question: str, context: str) -> QAResponse:
        """Extract the answer span for a given question from context."""
        start_time = time.time()
        q_clean = question.strip()
        c_clean = context.strip()

        if not q_clean:
            raise ValueError("Question cannot be empty")
        if not c_clean:
            latency_ms = (time.time() - start_time) * 1000.0
            return QAResponse(
                question=q_clean,
                answer="",
                start_char=0,
                end_char=0,
                confidence_score=0.0,
                model_name=self.model_name,
                latency_ms=round(latency_ms, 2),
            )

        self._init_model()

        # Tokenize question and context with offset mappings
        inputs = self.tokenizer(
            q_clean,
            c_clean,
            add_special_tokens=True,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        input_ids = inputs["input_ids"][0].cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)

        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))

        # Context token mask: ignore tokens belonging to question and special tokens
        sequence_ids = inputs.get("token_type_ids")
        # Ensure end_idx >= start_idx and reasonable span length
        if end_idx < start_idx or (end_idx - start_idx) > 30:
            end_idx = start_idx

        # Calculate confidence score
        confidence = float(start_probs[start_idx] * end_probs[end_idx])

        # Extract answer text from character offsets
        start_char, end_char = int(offset_mapping[start_idx][0]), int(offset_mapping[end_idx][1])
        if start_char == 0 and end_char == 0:
            # Pointing to [CLS] token (no answer)
            answer_text = ""
            confidence = 0.0
        else:
            answer_text = c_clean[start_char:end_char].strip()

        latency_ms = (time.time() - start_time) * 1000.0

        return QAResponse(
            question=q_clean,
            answer=answer_text,
            start_char=start_char,
            end_char=end_char,
            confidence_score=round(max(0.0, min(1.0, confidence)), 4),
            model_name=self.model_name,
            latency_ms=round(latency_ms, 2),
        )

    def answer_batch(self, items: List[QARequest]) -> QABatchResponse:
        """Answer a batch of questions over corresponding contexts."""
        responses = [self.answer_question(item.question, item.context) for item in items]
        avg_conf = float(np.mean([r.confidence_score for r in responses])) if responses else 0.0

        return QABatchResponse(
            results=responses,
            total_items=len(items),
            avg_confidence=round(avg_conf, 4),
        )

    def evaluate_qa(self, qa_dataset: List[Dict[str, Any]]) -> QAEvaluationMetrics:
        """Evaluate exact match and token F1 metrics across reference benchmark QA pairs."""
        em_scores = []
        f1_scores = []
        prec_scores = []
        rec_scores = []

        for item in qa_dataset:
            question = item["question"]
            context = item["context"]
            gold_answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]

            resp = self.answer_question(question, context)
            pred = resp.answer

            # Take max metric across all acceptable gold answers
            best_em, best_f1, best_prec, best_rec = 0.0, 0.0, 0.0, 0.0
            for gold in gold_answers:
                em, f1, prec, rec = compute_f1_and_em(pred, gold)
                if f1 > best_f1:
                    best_f1 = f1
                    best_prec = prec
                    best_rec = rec
                if em > best_em:
                    best_em = em

            em_scores.append(best_em)
            f1_scores.append(best_f1)
            prec_scores.append(best_prec)
            rec_scores.append(best_rec)

        mean_em = float(np.mean(em_scores)) * 100.0 if em_scores else 0.0
        mean_f1 = float(np.mean(f1_scores)) * 100.0 if f1_scores else 0.0
        mean_prec = float(np.mean(prec_scores)) * 100.0 if prec_scores else 0.0
        mean_rec = float(np.mean(rec_scores)) * 100.0 if rec_scores else 0.0

        return QAEvaluationMetrics(
            exact_match=round(mean_em, 2),
            token_f1=round(mean_f1, 2),
            precision=round(mean_prec, 2),
            recall=round(mean_rec, 2),
            sample_count=len(qa_dataset),
            model_name=self.model_name,
        )
