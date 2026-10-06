"""Named Entity Recognition API Router (Task 18 / FR-05)."""

import time
from typing import List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.deps import get_bert_ner_service, get_crf_ner_service
from backend.app.models.ner import NEREntity, NERRequest, NERResponse
from backend.app.services.bert_ner_service import BERTNERService
from backend.app.services.crf_ner_service import CRFNERService

router = APIRouter(prefix="/ner", tags=["Named Entity Recognition"])


class NERExtractRequest(BaseModel):
    """Request payload for entity extraction."""

    text: str = Field(..., description="Raw text string or paragraph to extract entities from")
    method: Literal["bert", "crf"] = Field(default="bert", description="Model family to use")


class NERBatchExtractRequest(BaseModel):
    """Batch entity extraction request."""

    texts: List[str] = Field(..., description="List of texts to extract entities from")
    method: Literal["bert", "crf"] = Field(default="bert", description="Model family to use")


class NERBatchExtractResponse(BaseModel):
    """Batch entity extraction response."""

    results: List[NERResponse]
    total_documents: int


@router.post(
    "/extract",
    response_model=NERResponse,
    summary="Extract named entities from text",
    status_code=status.HTTP_200_OK,
)
def extract_entities(
    payload: NERExtractRequest,
    bert_ner: BERTNERService = Depends(get_bert_ner_service),
    crf_ner: CRFNERService = Depends(get_crf_ner_service),
) -> NERResponse:
    """Extract named entities (PER, LOC, ORG, MISC) from input text."""
    start_time = time.time()
    text = payload.text.strip()
    if not text:
        return NERResponse(
            tokens=[],
            predicted_labels=[],
            entities=[],
            model_type=payload.method,
            latency_ms=0.0,
        )

    try:
        tokens = text.split()
        if payload.method == "bert":
            resp = bert_ner.predict_single(tokens)
        else:
            if not crf_ner.is_trained:
                # Fallback mini train for CRF
                crf_ner.train([["Alice", "visited", "Paris", "."]], [["B-PER", "O", "B-LOC", "O"]])
            resp = crf_ner.predict_single(tokens)

        latency_ms = (time.time() - start_time) * 1000.0
        resp.latency_ms = round(latency_ms, 2)
        return resp
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NER extraction failed: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=NERBatchExtractResponse,
    summary="Batch entity extraction",
    status_code=status.HTTP_200_OK,
)
def extract_entities_batch(
    payload: NERBatchExtractRequest,
    bert_ner: BERTNERService = Depends(get_bert_ner_service),
    crf_ner: CRFNERService = Depends(get_crf_ner_service),
) -> NERBatchExtractResponse:
    """Extract entities from multiple texts in batch."""
    results = []
    for text in payload.texts:
        single_req = NERExtractRequest(text=text, method=payload.method)
        resp = extract_entities(single_req, bert_ner=bert_ner, crf_ner=crf_ner)
        results.append(resp)

    return NERBatchExtractResponse(
        results=results,
        total_documents=len(payload.texts),
    )
