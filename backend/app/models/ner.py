"""Pydantic schemas for Named Entity Recognition (NER) models, requests, and evaluation metrics."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# CoNLL-2003 label set (IOB2 format)
# ---------------------------------------------------------------------------
CONLL_LABEL_NAMES: List[str] = [
    "O",        # 0 – Outside any entity
    "B-PER",    # 1 – Beginning of a person name
    "I-PER",    # 2 – Inside a person name
    "B-ORG",    # 3 – Beginning of an organization
    "I-ORG",    # 4 – Inside an organization
    "B-LOC",    # 5 – Beginning of a location
    "I-LOC",    # 6 – Inside a location
    "B-MISC",   # 7 – Beginning of a miscellaneous entity
    "I-MISC",   # 8 – Inside a miscellaneous entity
]

CONLL_LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(CONLL_LABEL_NAMES)}
CONLL_ID2LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(CONLL_LABEL_NAMES)}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class NERRequest(BaseModel):
    """Request for running NER on a piece of text."""

    text: str = Field(..., description="Raw text on which to run NER")
    model_type: Optional[str] = Field(
        "crf",
        description="Model to use: 'crf' (sklearn-CRF baseline) or 'transformer'",
    )


class NEREntity(BaseModel):
    """A single extracted named entity span."""

    text: str = Field(..., description="Surface form of the entity")
    label: str = Field(..., description="Entity type label, e.g. PER, ORG, LOC, MISC")
    start_token: int = Field(..., description="Start token index (inclusive)")
    end_token: int = Field(..., description="End token index (inclusive)")


class NERResponse(BaseModel):
    """Response returned by the NER service."""

    tokens: List[str] = Field(..., description="Input tokens")
    predicted_labels: List[str] = Field(..., description="IOB2 label per token")
    entities: List[NEREntity] = Field(..., description="Extracted entity spans")
    model_type: str = Field(..., description="Model architecture used")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class NEREntityMetrics(BaseModel):
    """Per-entity-type precision/recall/F1 metrics."""

    entity_type: str
    precision: float
    recall: float
    f1: float
    support: int


class NEREvaluationMetrics(BaseModel):
    """Top-level NER evaluation result (entity-level, micro and macro)."""

    precision_micro: float = Field(..., description="Entity-level micro precision")
    recall_micro: float = Field(..., description="Entity-level micro recall")
    f1_micro: float = Field(..., description="Entity-level micro F1 (primary metric)")
    precision_macro: float = Field(..., description="Entity-level macro precision")
    recall_macro: float = Field(..., description="Entity-level macro recall")
    f1_macro: float = Field(..., description="Entity-level macro F1")
    per_entity_metrics: List[NEREntityMetrics] = Field(
        default_factory=list, description="Breakdown by entity type"
    )
    token_accuracy: float = Field(..., description="Token-level accuracy (including O tags)")
    sample_count: int = Field(..., description="Number of sentences evaluated")
    model_type: str = Field(..., description="Model architecture that produced these metrics")
