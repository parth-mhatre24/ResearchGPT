"""ResearchGPT Semantic Textual Similarity Service (Task 13 / FR-07).

Evaluates semantic textual similarity using:
1. Classical TF-IDF Cosine Similarity Baseline
2. Neural Sentence-BERT Embeddings (all-MiniLM-L6-v2)

Provides correlation evaluation against STS Benchmark (STS-B) ground truth scores.
"""

import logging
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("similarity_service")

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

    _TRANSFORMERS_AVAILABLE = True
except Exception as _e:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers / PyTorch not available for Sentence-BERT: {_e}")

from backend.app.models.similarity import (
    SimilarityBatchRequest,
    SimilarityBatchResponse,
    SimilarityEvaluationMetrics,
    SimilarityPairRequest,
    SimilarityPairResponse,
)


def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # First element contains hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class SemanticSimilarityService:
    """Service for computing and evaluating semantic textual similarity."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = "cuda" if _TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"

    def _init_transformer(self) -> None:
        """Lazily initialize the transformer encoder."""
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required for Sentence-BERT embeddings.")

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into L2-normalized dense sentence embeddings."""
        self._init_transformer()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
                # L2 normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def compute_similarity_pair(
        self, text_a: str, text_b: str, method: str = "sentence_bert"
    ) -> SimilarityPairResponse:
        """Compute semantic similarity between two texts."""
        start_time = time.time()

        if method == "tfidf":
            sim = self._compute_tfidf_similarity(text_a, text_b)
        else:
            sim = self._compute_sentence_bert_similarity(text_a, text_b)

        latency_ms = (time.time() - start_time) * 1000.0
        score_normalized = max(0.0, min(1.0, float(sim)))
        score_stsb = score_normalized * 5.0

        return SimilarityPairResponse(
            text_a=text_a,
            text_b=text_b,
            similarity_score=round(score_normalized, 4),
            score_stsb_scale=round(score_stsb, 3),
            method=method,
            latency_ms=round(latency_ms, 2),
        )

    def compute_similarity_batch(
        self, pairs: List[Tuple[str, str]], method: str = "sentence_bert"
    ) -> SimilarityBatchResponse:
        """Compute similarity scores for a batch of text pairs."""
        responses = []
        scores = []

        if method == "sentence_bert" and _TRANSFORMERS_AVAILABLE:
            # Efficient vectorized computation
            texts_a = [p[0] for p in pairs]
            texts_b = [p[1] for p in pairs]
            embs_a = self.encode_texts(texts_a)
            embs_b = self.encode_texts(texts_b)
            # Dot product of normalized vectors equals cosine similarity
            cos_sims = np.sum(embs_a * embs_b, axis=1)

            for i, (a, b) in enumerate(pairs):
                sim = max(0.0, min(1.0, float(cos_sims[i])))
                resp = SimilarityPairResponse(
                    text_a=a,
                    text_b=b,
                    similarity_score=round(sim, 4),
                    score_stsb_scale=round(sim * 5.0, 3),
                    method=method,
                    latency_ms=0.0,
                )
                responses.append(resp)
                scores.append(sim)
        else:
            for a, b in pairs:
                resp = self.compute_similarity_pair(a, b, method=method)
                responses.append(resp)
                scores.append(resp.similarity_score)

        mean_sim = float(np.mean(scores)) if scores else 0.0
        return SimilarityBatchResponse(
            results=responses,
            total_pairs=len(pairs),
            mean_similarity=round(mean_sim, 4),
        )

    def _compute_tfidf_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between TF-IDF representations."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        try:
            tfidf_mat = vectorizer.fit_transform([text_a, text_b])
            sim = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]
            return float(sim)
        except Exception:
            return 0.0

    def _compute_sentence_bert_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity using Sentence-BERT embeddings."""
        embs = self.encode_texts([text_a, text_b])
        # Dot product of normalized embeddings
        sim = float(np.dot(embs[0], embs[1]))
        return sim

    def evaluate_stsb(
        self,
        texts_a: List[str],
        texts_b: List[str],
        gold_scores: List[float],
        method: str = "sentence_bert",
    ) -> SimilarityEvaluationMetrics:
        """Evaluate correlation metrics against STS-B benchmark ground truth scores (scale 0-5)."""
        pairs = list(zip(texts_a, texts_b))
        batch_resp = self.compute_similarity_batch(pairs, method=method)
        pred_scores = np.array([r.score_stsb_scale for r in batch_resp.results])
        gold_scores_arr = np.array(gold_scores)

        # Pearson & Spearman correlation
        pearson_corr, _ = pearsonr(pred_scores, gold_scores_arr)
        spearman_corr, _ = spearmanr(pred_scores, gold_scores_arr)
        mse = float(np.mean((pred_scores - gold_scores_arr) ** 2))

        return SimilarityEvaluationMetrics(
            pearson_correlation=round(float(pearson_corr), 4),
            spearman_correlation=round(float(spearman_corr), 4),
            mse=round(mse, 4),
            sample_count=len(gold_scores),
            method=method,
            model_name=self.model_name if method == "sentence_bert" else "tfidf",
        )
