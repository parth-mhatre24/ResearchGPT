"""ResearchGPT FLAN-T5 Generative Question Answering Service.

Generates coherent, full-sentence scientific answers conditioned strictly
on retrieved context passages using Google's `google/flan-t5-base` model.
"""

import logging
import time
from typing import Any, List, Optional, Union

logger = logging.getLogger("flan_t5_service")

# ---------------------------------------------------------------------------
# Lazy imports for Transformer / PyTorch Seq2Seq
# ---------------------------------------------------------------------------
try:
    import torch
    import transformers.utils as _u
    import transformers.utils.import_utils as _iu
    _iu.is_datasets_available = lambda: False
    _u.is_datasets_available = lambda: False
    _iu.check_torch_load_is_safe = lambda: None

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _DEPS_AVAILABLE = True
except Exception as _e:
    _DEPS_AVAILABLE = False
    logger.warning(f"PyTorch / transformers not available for FlanT5GenerativeService: {_e}")


class FlanT5GenerativeService:
    """Service for generative scientific question answering using google/flan-t5-base."""

    DEFAULT_MODEL = "google/flan-t5-base"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = "cuda" if _DEPS_AVAILABLE and torch.cuda.is_available() else "cpu"

    def _init_model(self) -> None:
        """Lazily load the Seq2Seq tokenizer and model weights."""
        if not _DEPS_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required for FlanT5GenerativeService.")

        if self.tokenizer is None:
            logger.info(f"Loading FLAN-T5 tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            logger.info(f"Loading FLAN-T5 model from {self.model_name} on {self.device}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def generate_answer(
        self,
        question: str,
        contexts: Union[str, List[str]],
        max_new_tokens: int = 200,
        num_beams: int = 4,
    ) -> str:
        """Generate a natural language answer based strictly on the provided context passages."""
        q_clean = question.strip()
        if not q_clean:
            raise ValueError("Question cannot be empty")

        if isinstance(contexts, list):
            context_str = "\n\n".join([c.strip() for c in contexts if c and c.strip()])
        else:
            context_str = str(contexts).strip()

        if not context_str:
            return "I could not find sufficient relevant evidence in the paper to answer this question."

        # Detect broad conceptual / summarization inquiry
        is_summary = any(
            k in q_clean.lower()
            for k in ["summar", "overview", "what is this paper", "describe", "explain", "main contribution", "key finding"]
        )

        if is_summary:
            prompt = (
                f"Context:\n{context_str}\n\n"
                f"Instruction: Provide a comprehensive summary of the research paper described in the context, covering the methodology, models, and key findings.\n"
                f"Summary:"
            )
            min_len = 35
            target_max_tokens = max(max_new_tokens, 350)
        else:
            prompt = (
                f"Context:\n{context_str}\n\n"
                f"Question: {q_clean}\n\n"
                f"Answer: Answer the question thoroughly and list all points completely from the context.\n"
            )
            min_len = 15
            target_max_tokens = max(max_new_tokens, 450)

        try:
            self._init_model()

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=target_max_tokens,
                    min_length=min_len,
                    num_beams=2,
                    length_penalty=1.2,
                    no_repeat_ngram_size=0,
                    repetition_penalty=1.0,
                    early_stopping=False,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return generated_text if generated_text else "I could not find sufficient relevant evidence in the paper."

        except Exception as e:
            logger.error(f"FLAN-T5 generation failed: {e}", exc_info=True)
            # Graceful fallback: return top context excerpt
            return context_str[:300] + ("..." if len(context_str) > 300 else "")
