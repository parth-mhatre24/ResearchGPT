"""ResearchGPT CRF NER Service (Task 9 – NER Baseline).

Implements a feature-rich sklearn-crfsuite Conditional Random Field (CRF)
for sequence labelling on CoNLL-2003 (IOB2 BIO tags).

Features per token:
- word shape, prefix/suffix, capitalisation flags
- 2-token context window (preceding and following tokens)
- BOS / EOS markers

Provides training, inference, entity-span extraction, standard
entity-level evaluation metrics (P/R/F1), and artifact serialization.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("crf_ner_service")

# ---------------------------------------------------------------------------
# Lazy import – sklearn-crfsuite is optional for non-NER tasks
# ---------------------------------------------------------------------------
try:
    import sklearn_crfsuite  # type: ignore[import-untyped]
    from sklearn_crfsuite import metrics as crf_metrics  # type: ignore[import-untyped]

    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False
    _CRF_IMPORT_ERROR = "sklearn-crfsuite not installed. Run: pip install sklearn-crfsuite"

from backend.app.models.ner import (
    CONLL_ID2LABEL,
    CONLL_LABEL_NAMES,
    NEREvaluationMetrics,
    NEREntity,
    NEREntityMetrics,
    NERResponse,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _word_shape(word: str) -> str:
    """Collapse each character to its shape class (X=upper, x=lower, d=digit, ?=other)."""
    shape = ""
    for ch in word:
        if ch.isupper():
            shape += "X"
        elif ch.islower():
            shape += "x"
        elif ch.isdigit():
            shape += "d"
        else:
            shape += "?"
    # Compress repeated shape chars (XxXx -> XxXx stays; XXX -> X{3})
    return re.sub(r"(.)\1{3,}", lambda m: m.group(1) + "+", shape)


def _token_features(tokens: List[str], i: int) -> Dict[str, Any]:
    """Return a feature dict for token at position i within the sentence."""
    word = tokens[i]
    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:2]": word[:2],
        "word[:3]": word[:3],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.shape": _word_shape(word),
        "word.hasdigit": any(c.isdigit() for c in word),
        "word.hyphon": "-" in word,
    }

    if i > 0:
        prev_word = tokens[i - 1]
        features.update({
            "-1:word.lower": prev_word.lower(),
            "-1:word.istitle": prev_word.istitle(),
            "-1:word.isupper": prev_word.isupper(),
        })
    else:
        features["BOS"] = True  # beginning of sentence

    if i > 1:
        prev2 = tokens[i - 2]
        features.update({
            "-2:word.lower": prev2.lower(),
            "-2:word.istitle": prev2.istitle(),
        })

    if i < len(tokens) - 1:
        next_word = tokens[i + 1]
        features.update({
            "+1:word.lower": next_word.lower(),
            "+1:word.istitle": next_word.istitle(),
            "+1:word.isupper": next_word.isupper(),
        })
    else:
        features["EOS"] = True  # end of sentence

    if i < len(tokens) - 2:
        next2 = tokens[i + 2]
        features.update({
            "+2:word.lower": next2.lower(),
            "+2:word.istitle": next2.istitle(),
        })

    return features


def sentence_to_features(tokens: List[str]) -> List[Dict[str, Any]]:
    """Convert a token list into a list of per-token feature dicts."""
    return [_token_features(tokens, i) for i in range(len(tokens))]


def ids_to_labels(tag_ids: List[int]) -> List[str]:
    """Map integer NER tag IDs to IOB2 label strings."""
    return [CONLL_ID2LABEL.get(t, "O") for t in tag_ids]


# ---------------------------------------------------------------------------
# Entity span extraction
# ---------------------------------------------------------------------------

def extract_entities(tokens: List[str], labels: List[str]) -> List[NEREntity]:
    """Convert a BIO-tagged sequence into a list of NEREntity spans."""
    entities: List[NEREntity] = []
    current: Optional[Dict] = None

    for i, (tok, lbl) in enumerate(zip(tokens, labels)):
        if lbl.startswith("B-"):
            if current is not None:
                entities.append(NEREntity(**current))
            current = {
                "text": tok,
                "label": lbl[2:],
                "start_token": i,
                "end_token": i,
            }
        elif lbl.startswith("I-") and current is not None and current["label"] == lbl[2:]:
            current["text"] += f" {tok}"
            current["end_token"] = i
        else:
            if current is not None:
                entities.append(NEREntity(**current))
                current = None

    if current is not None:
        entities.append(NEREntity(**current))

    return entities


# ---------------------------------------------------------------------------
# Entity-level evaluation helpers
# ---------------------------------------------------------------------------

def _collect_spans(
    tag_sequences: List[List[str]],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Return {entity_type: [(sent_id, start, end), ...]} from a list of label sequences."""
    spans: Dict[str, List] = {}
    for s_id, labels in enumerate(tag_sequences):
        for entity in extract_entities([str(j) for j in range(len(labels))], labels):
            spans.setdefault(entity.label, []).append(
                (s_id, entity.start_token, entity.end_token)
            )
    return spans


def compute_ner_metrics(
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    model_type: str = "crf",
) -> NEREvaluationMetrics:
    """Compute entity-level precision, recall, and F1 across all entity types."""
    # Convert to plain lists to avoid numpy array concatenation issues
    true_seqs = [list(seq) for seq in true_sequences]
    pred_seqs = [list(seq) for seq in pred_sequences]

    entity_types = sorted({
        lbl[2:] for seq in (true_seqs + pred_seqs)
        for lbl in seq if lbl.startswith("B-")
    })

    true_spans = _collect_spans(true_seqs)
    pred_spans = _collect_spans(pred_seqs)

    per_entity: List[NEREntityMetrics] = []
    total_tp = total_fp = total_fn = 0

    for etype in entity_types:
        t_set = set(true_spans.get(etype, []))
        p_set = set(pred_spans.get(etype, []))
        tp = len(t_set & p_set)
        fp = len(p_set - t_set)
        fn = len(t_set - p_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_entity.append(
            NEREntityMetrics(
                entity_type=etype,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                support=len(t_set),
            )
        )

    # Micro (pooled across all entity types)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) else 0.0
    )

    # Macro (average over entity types)
    macro_prec = float(np.mean([e.precision for e in per_entity])) if per_entity else 0.0
    macro_rec = float(np.mean([e.recall for e in per_entity])) if per_entity else 0.0
    macro_f1 = float(np.mean([e.f1 for e in per_entity])) if per_entity else 0.0

    # Token accuracy (flat, includes O)
    flat_true = [lbl for seq in true_seqs for lbl in seq]
    flat_pred = [lbl for seq in pred_seqs for lbl in seq]
    token_acc = sum(t == p for t, p in zip(flat_true, flat_pred)) / len(flat_true) if flat_true else 0.0

    return NEREvaluationMetrics(
        precision_micro=round(micro_prec, 4),
        recall_micro=round(micro_rec, 4),
        f1_micro=round(micro_f1, 4),
        precision_macro=round(macro_prec, 4),
        recall_macro=round(macro_rec, 4),
        f1_macro=round(macro_f1, 4),
        per_entity_metrics=per_entity,
        token_accuracy=round(token_acc, 4),
        sample_count=len(true_sequences),
        model_type=model_type,
    )


# ---------------------------------------------------------------------------
# CRF NER Service
# ---------------------------------------------------------------------------

class CRFNERService:
    """Conditional Random Field NER baseline service.

    Args:
        algorithm: CRF algorithm (lbfgs, l2sgd, pa, ap, arow).
        c1: L1 regularisation (lbfgs only).
        c2: L2 regularisation (lbfgs only).
        max_iterations: Maximum training iterations.
        all_possible_transitions: Whether to include all label transitions.
    """

    def __init__(
        self,
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 200,
        all_possible_transitions: bool = True,
    ):
        if not _CRF_AVAILABLE:
            raise ImportError(_CRF_IMPORT_ERROR)

        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.all_possible_transitions = all_possible_transitions

        self.crf: Optional[sklearn_crfsuite.CRF] = None
        self._is_trained: bool = False
        self._build()

    def _build(self) -> None:
        self.crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=self.all_possible_transitions,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        token_sequences: List[List[str]],
        label_sequences: List[List[str]],
    ) -> "CRFNERService":
        """Fit the CRF on a list of tokenised sentences and their IOB2 label sequences.

        Args:
            token_sequences: List of token lists, one per sentence.
            label_sequences: List of label lists (IOB2 strings), one per sentence.

        Returns:
            self (for chaining).
        """
        if not token_sequences:
            raise ValueError("token_sequences cannot be empty")
        if len(token_sequences) != len(label_sequences):
            raise ValueError(
                f"Length mismatch: {len(token_sequences)} token sequences vs "
                f"{len(label_sequences)} label sequences"
            )

        X = [sentence_to_features(tokens) for tokens in token_sequences]
        y = label_sequences

        logger.info("Training CRF on %d sentences ...", len(token_sequences))
        t0 = time.perf_counter()
        self.crf.fit(X, y)
        elapsed = time.perf_counter() - t0
        self._is_trained = True
        logger.info("CRF trained in %.2fs", elapsed)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_sequence(self, tokens: List[str]) -> List[str]:
        """Return IOB2 label predictions for a single sentence."""
        self._check_trained()
        features = [sentence_to_features(tokens)]
        return list(self.crf.predict(features)[0])

    def predict_batch(self, token_sequences: List[List[str]]) -> List[List[str]]:
        """Return IOB2 label predictions for a batch of sentences."""
        self._check_trained()
        X = [sentence_to_features(tokens) for tokens in token_sequences]
        return [list(seq) for seq in self.crf.predict(X)]

    def predict_single(self, text: str) -> NERResponse:
        """Tokenise text (whitespace split), predict labels, extract entities."""
        tokens = text.strip().split()
        labels = self.predict_sequence(tokens)
        entities = extract_entities(tokens, labels)
        return NERResponse(
            tokens=tokens,
            predicted_labels=labels,
            entities=entities,
            model_type="crf",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        token_sequences: List[List[str]],
        label_sequences: List[List[str]],
    ) -> NEREvaluationMetrics:
        """Evaluate the CRF on gold-standard token/label sequences."""
        pred_sequences = self.predict_batch(token_sequences)
        return compute_ner_metrics(label_sequences, pred_sequences, model_type="crf")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialise CRF model and companion metadata to disk."""
        import joblib

        self._check_trained()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        name = model_name or "crf_ner"
        model_file = out_path / f"{name}.joblib"
        meta_file = out_path / f"{name}_meta.json"

        joblib.dump(self.crf, model_file)

        meta: Dict[str, Any] = {
            "model_type": "crf",
            "algorithm": self.algorithm,
            "c1": self.c1,
            "c2": self.c2,
            "max_iterations": self.max_iterations,
            "all_possible_transitions": self.all_possible_transitions,
        }
        if extra_metadata:
            meta["extra"] = extra_metadata
        with open(meta_file, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Saved CRF NER model to %s", model_file)
        return model_file

    @classmethod
    def load(cls, model_file: Union[str, Path]) -> "CRFNERService":
        """Load a CRF model from a .joblib file plus companion metadata."""
        import joblib

        path = Path(model_file)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        meta_file = path.parent / f"{path.stem}_meta.json"
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        service = cls(
            algorithm=meta.get("algorithm", "lbfgs"),
            c1=meta.get("c1", 0.1),
            c2=meta.get("c2", 0.1),
            max_iterations=meta.get("max_iterations", 200),
            all_possible_transitions=meta.get("all_possible_transitions", True),
        )
        service.crf = joblib.load(path)
        service._is_trained = True
        return service

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained or self.crf is None:
            raise RuntimeError(
                "CRFNERService has not been trained or loaded yet. "
                "Call .train() or .load() first."
            )
