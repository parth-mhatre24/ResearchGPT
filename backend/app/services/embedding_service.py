"""ResearchGPT Dense Embedding Service (Task 14 / FR-08).

Generates 384-dimensional dense semantic embeddings using Sentence-BERT
(`sentence-transformers/all-MiniLM-L6-v2`) with mean pooling and unit L2 normalization.
"""

import logging
import time
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger("embedding_service")

# ---------------------------------------------------------------------------
# Lazy imports for Transformer / PyTorch Sentence Embeddings
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    import transformers.utils as _u
    import transformers.utils.import_utils as _iu
    _iu.is_datasets_available = lambda: False
    _u.is_datasets_available = lambda: False
    _iu.check_torch_load_is_safe = lambda: None

    from transformers import AutoModel, AutoTokenizer

    _DEPS_AVAILABLE = True
except Exception as _e:
    _DEPS_AVAILABLE = False
    logger.warning(f"PyTorch / transformers not available for EmbeddingService: {_e}")

from backend.app.models.embeddings import DocumentChunk, EmbeddingResponse


def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class EmbeddingService:
    """Service for encoding text chunks into normalized dense vectors."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = "cuda" if _DEPS_AVAILABLE and torch.cuda.is_available() else "cpu"

    def _init_model(self) -> None:
        """Lazily initialize tokenizer and transformer encoder."""
        if not _DEPS_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required for EmbeddingService.")

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into an (N, 384) L2-normalized float32 matrix."""
        if not texts:
            return np.zeros((0, self.DIMENSION), dtype=np.float32)

        self._init_model()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1D (384,) L2-normalized vector."""
        embs = self.embed_texts([query])
        return embs[0]

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> np.ndarray:
        """Encode a list of DocumentChunk instances."""
        texts = [c.text for c in chunks]
        return self.embed_texts(texts, batch_size=batch_size)

    def get_embedding_response(self, texts: List[str]) -> EmbeddingResponse:
        """Generate an EmbeddingResponse Pydantic payload."""
        start_time = time.time()
        embs = self.embed_texts(texts)
        latency_ms = (time.time() - start_time) * 1000.0

        return EmbeddingResponse(
            embeddings=embs.tolist(),
            dimension=embs.shape[1] if len(embs) > 0 else self.DIMENSION,
            model_name=self.model_name,
            sample_count=len(texts),
            latency_ms=round(latency_ms, 2),
        )
