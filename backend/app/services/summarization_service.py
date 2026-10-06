"""ResearchGPT Summarization Service (Task 12).

Implements abstractive document summarization using seq2seq transformer models
(e.g., T5 or BART) with ROUGE-1/2/L evaluation metrics.

Features:
- Extractive pre-truncation / lead text chunking for long research texts
- Generation controls (max/min length, beam search, length penalty, n-gram repeat suppression)
- Batch summarization support
- Evaluation framework using ROUGE-1, ROUGE-2, and ROUGE-L metrics
- Model artifact loading and saving
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("summarization_service")

# ---------------------------------------------------------------------------
# Lazy imports – transformers, torch, rouge_score
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    _DEPS_IMPORT_ERROR = "transformers and torch are required. Install with: pip install transformers torch"

try:
    from rouge_score import rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


from backend.app.models.summarization import (
    SummarizationEvaluationMetrics,
    SummarizationResponse,
)


class SummarizationService:
    """Service for abstractive document summarization using T5 or BART."""

    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.is_loaded: bool = False

    def _check_deps(self) -> None:
        if not _DEPS_AVAILABLE:
            raise RuntimeError(_DEPS_IMPORT_ERROR)

    def load_model(self) -> None:
        """Lazy load tokenizer and seq2seq model checkpoint."""
        self._check_deps()
        if not self.is_loaded:
            logger.info(f"Loading summarization model checkpoint: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.is_loaded = True

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 2.0,
    ) -> SummarizationResponse:
        """Generate an abstractive summary for a single document text."""
        self.load_model()

        start_time = time.time()
        input_words = text.split()
        input_length_words = len(input_words)

        if not text.strip():
            return SummarizationResponse(
                summary="",
                input_length_words=0,
                summary_length_words=0,
                compression_ratio=0.0,
                model_type=self.model_name,
                execution_time_sec=0.0,
            )

        # Format prompt if T5 model family
        prompt_text = text
        if "t5" in self.model_name.lower() and not text.startswith("summarize:"):
            prompt_text = f"summarize: {text}"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                early_stopping=True,
            )

        summary_text = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        ).strip()

        exec_time = time.time() - start_time
        summary_words = summary_text.split()
        summary_length_words = len(summary_words)
        compression_ratio = (
            round(summary_length_words / input_length_words, 4)
            if input_length_words > 0
            else 0.0
        )

        return SummarizationResponse(
            summary=summary_text,
            input_length_words=input_length_words,
            summary_length_words=summary_length_words,
            compression_ratio=compression_ratio,
            model_type=self.model_name,
            execution_time_sec=round(exec_time, 3),
        )

    def summarize_batch(
        self,
        texts: List[str],
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
    ) -> List[SummarizationResponse]:
        """Summarize a batch of input texts."""
        return [
            self.summarize(
                txt, max_length=max_length, min_length=min_length, num_beams=num_beams
            )
            for txt in texts
        ]

    def evaluate(
        self, reference_summaries: List[str], generated_summaries: List[str]
    ) -> SummarizationEvaluationMetrics:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L metrics across reference/generated summary pairs."""
        if not _ROUGE_AVAILABLE:
            raise RuntimeError("rouge_score package is required. Install with: pip install rouge-score")

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        rouge1_f1s: List[float] = []
        rouge2_f1s: List[float] = []
        rougeL_f1s: List[float] = []

        for ref, gen in zip(reference_summaries, generated_summaries):
            scores = scorer.score(ref, gen)
            rouge1_f1s.append(scores["rouge1"].fmeasure)
            rouge2_f1s.append(scores["rouge2"].fmeasure)
            rougeL_f1s.append(scores["rougeL"].fmeasure)

        r1 = float(np.mean(rouge1_f1s)) if rouge1_f1s else 0.0
        r2 = float(np.mean(rouge2_f1s)) if rouge2_f1s else 0.0
        rL = float(np.mean(rougeL_f1s)) if rougeL_f1s else 0.0

        return SummarizationEvaluationMetrics(
            rouge1=round(r1, 4),
            rouge2=round(r2, 4),
            rougeL=round(rL, 4),
            sample_count=len(reference_summaries),
            model_type=self.model_name,
        )

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save model and tokenizer weights to directory."""
        self.load_model()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        meta_path = path / "metadata.json"
        meta = {"model_name": self.model_name, "model_type": "seq2seq_summarization"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved summarization artifact to {path}")
        return path

    def load(self, model_dir: Union[str, Path]) -> "SummarizationService":
        """Load model and tokenizer weights from directory."""
        self._check_deps()
        path = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.is_loaded = True

        logger.info(f"Loaded summarization artifact from {path}")
        return self
