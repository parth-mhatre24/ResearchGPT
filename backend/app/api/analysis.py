"""Integrated Paper Analysis API Router (Task 18 / End-to-End Orchestration)."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.deps import (
    get_bert_ner_service,
    get_retrieval_service,
    get_summarization_service,
)
from backend.app.models.ner import NEREntity, NERResponse
from backend.app.models.summarization import (
    SummarizationRequest,
    SummarizationResponse,
)
from backend.app.services.bert_ner_service import BERTNERService
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.summarization_service import SummarizationService

router = APIRouter(prefix="/analysis", tags=["Integrated Paper Analysis"])


class PaperAnalysisRequest(BaseModel):
    """Request payload for comprehensive single-pass research paper analysis."""

    document_id: str = Field(..., description="Unique paper identifier or title")
    text: str = Field(..., description="Full text or abstract of the research paper")
    auto_index: bool = Field(default=True, description="Automatically index chunks into vector search")
    max_summary_length: int = Field(default=150, description="Target maximum summary tokens")


class PaperAnalysisResponse(BaseModel):
    """Integrated research paper analysis digest."""

    document_id: str
    word_count: int
    summary: SummarizationResponse
    entities: List[NEREntity]
    chunks_indexed: int
    analysis_latency_ms: float


@router.post(
    "/paper",
    response_model=PaperAnalysisResponse,
    summary="Generate complete multi-modal research paper analysis digest",
    status_code=status.HTTP_200_OK,
)
def analyze_paper(
    payload: PaperAnalysisRequest,
    summarizer: SummarizationService = Depends(get_summarization_service),
    bert_ner: BERTNERService = Depends(get_bert_ner_service),
    retrieval_service: VectorRetrievalService = Depends(get_retrieval_service),
) -> PaperAnalysisResponse:
    """Analyze a research paper in a single API call: generate summary, extract technical entities, and index for RAG."""
    start_time = time.time()
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text cannot be empty.",
        )

    try:
        # Step 1: Abstractive Summarization
        summ_resp = summarizer.summarize(
            text=text,
            max_length=payload.max_summary_length,
            min_length=20,
            num_beams=4,
        )

        # Step 2: Named Entity Recognition
        tokens = text.split()[:250]  # First 250 tokens for entity extraction
        ner_resp: NERResponse = bert_ner.predict_single(tokens)

        # Step 3: Vector Indexing (optional)
        chunks_indexed = 0
        if payload.auto_index:
            chunks_indexed = retrieval_service.add_document(
                document_id=payload.document_id,
                text=text,
                chunk_size=50,
                chunk_overlap=10,
            )

        latency_ms = (time.time() - start_time) * 1000.0

        return PaperAnalysisResponse(
            document_id=payload.document_id,
            word_count=len(text.split()),
            summary=summ_resp,
            entities=ner_resp.entities,
            chunks_indexed=chunks_indexed,
            analysis_latency_ms=round(latency_ms, 2),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paper analysis failed: {exc}",
        ) from exc
