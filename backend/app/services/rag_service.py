"""ResearchGPT Retrieval-Augmented Generation / RAG Service (Task 17 / FR-10).

Orchestrates semantic vector retrieval (Task 15) and extractive QA (Task 16)
to synthesize grounded answers with precise document provenance citations.
"""

import logging
import time
from typing import List, Optional, Union

from backend.app.models.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    SourceCitation,
)
from backend.app.models.retrieval import RetrievalQuery
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.flan_t5_service import FlanT5GenerativeService
from backend.app.services.qa_service import QuestionAnsweringService

logger = logging.getLogger("rag_service")


class RAGService:
    """Service for end-to-end Retrieval-Augmented Generation (RAG)."""

    def __init__(
        self,
        retrieval_service: Optional[VectorRetrievalService] = None,
        qa_service: Optional[QuestionAnsweringService] = None,
        generative_service: Optional[FlanT5GenerativeService] = None,
    ):
        self.retrieval_service = retrieval_service or VectorRetrievalService()
        self.qa_service = qa_service or QuestionAnsweringService()
        self.generative_service = generative_service or FlanT5GenerativeService()

    def query(
        self,
        query: Union[str, RAGQueryRequest],
        top_k: int = 3,
        similarity_threshold: float = 0.30,
        document_id: Optional[str] = None,
        generation_mode: str = "generative",
    ) -> RAGQueryResponse:
        """Execute grounded RAG pipeline: Query -> Vector Search -> Generative/Extractive QA -> Citations."""
        start_total = time.time()

        if isinstance(query, RAGQueryRequest):
            q_text = query.queryquery if hasattr(query, "queryquery") else query.query
            k = query.top_k
            threshold = query.similarity_threshold
            doc_filter = query.document_id
            mode = getattr(query, "generation_mode", "generative") or "generative"
        else:
            q_text = str(query)
            k = top_k
            threshold = similarity_threshold
            doc_filter = document_id
            mode = generation_mode

        if not q_text.strip():
            raise ValueError("Query cannot be empty")

        # Step 1: Semantic Vector Retrieval (retrieve candidate chunks)
        is_summary_query = any(w in q_text.lower() for w in ["summar", "overview", "what is this paper", "describe", "explain"])
        search_k = max(k, 4) if (is_summary_query or mode.lower() == "generative") else k

        retrieval_req = RetrievalQuery(
            query_text=q_text,
            top_k=search_k,
            score_threshold=None,  # Fetch candidate chunks, evaluate relevance downstream
            document_id=doc_filter,
        )
        retrieval_resp = self.retrieval_service.search(retrieval_req)
        retrieval_latency = retrieval_resp.retrieval_latency_ms

        # Step 2: Handle empty retrieval case
        if retrieval_resp.total_retrieved == 0:
            total_latency = (time.time() - start_total) * 1000.0
            return RAGQueryResponse(
                query=q_text,
                answer="I could not find sufficient relevant evidence in the indexed research papers to answer this question.",
                confidence_score=0.0,
                is_grounded=False,
                citations=[],
                retrieval_latency_ms=round(retrieval_latency, 2),
                qa_latency_ms=0.0,
                total_latency_ms=round(total_latency, 2),
            )

        # Step 3: Build Source Citations
        citations: List[SourceCitation] = []
        for i, chunk in enumerate(retrieval_resp.results, start=1):
            snippet = chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
            citation = SourceCitation(
                citation_index=i,
                document_id=chunk.metadata.document_id,
                page_number=chunk.metadata.page_number,
                section_title=chunk.metadata.section_title,
                chunk_id=chunk.chunk_id,
                similarity_score=chunk.similarity_score,
                text_snippet=snippet,
            )
            citations.append(citation)

        top_chunk = retrieval_resp.results[0]
        top_sim = top_chunk.similarity_score

        # Step 4: Anti-hallucination / Refusal Check
        if top_sim < 0.08:
            total_latency = (time.time() - start_total) * 1000.0
            return RAGQueryResponse(
                query=q_text,
                answer="I could not find sufficient relevant evidence in the indexed research papers to answer this question.",
                confidence_score=0.0,
                is_grounded=False,
                citations=[],
                retrieval_latency_ms=round(retrieval_latency, 2),
                qa_latency_ms=0.0,
                total_latency_ms=round(total_latency, 2),
            )

        # Step 5: Answer Synthesis
        start_qa = time.time()

        if mode.lower() == "generative":
            # Extract text passages from retrieved chunks
            passages = [chunk.text for chunk in retrieval_resp.results]
            generated_answer = self.generative_service.generate_answer(q_text, passages)

            qa_latency = (time.time() - start_qa) * 1000.0
            total_latency = (time.time() - start_total) * 1000.0

            # Confidence based on retrieval relevance
            confidence = min(1.0, max(0.20, top_sim + 0.35)) if top_sim >= 0.15 else max(0.1, top_sim * 2.0)

            return RAGQueryResponse(
                query=q_text,
                answer=generated_answer,
                confidence_score=round(confidence, 4),
                is_grounded=True,
                citations=citations,
                retrieval_latency_ms=round(retrieval_latency, 2),
                qa_latency_ms=round(qa_latency, 2),
                total_latency_ms=round(total_latency, 2),
            )

        else:
            # Extractive Mode (SQuAD span extraction)
            best_answer = ""
            best_score = -1.0
            best_chunk_idx = 0

            for idx, chunk in enumerate(retrieval_resp.results):
                qa_resp = self.qa_service.answer_question(q_text, chunk.text)
                combined_score = (qa_resp.confidence_score * 0.6) + (chunk.similarity_score * 0.4)
                if qa_resp.answer and combined_score > best_score:
                    best_score = combined_score
                    best_answer = qa_resp.answer
                    best_chunk_idx = idx

            qa_latency = (time.time() - start_qa) * 1000.0
            total_latency = (time.time() - start_total) * 1000.0

            if best_answer.strip() and best_score > 0.18:
                final_answer = best_answer
                final_confidence = best_score
            else:
                final_answer = top_chunk.text
                final_confidence = top_sim

            return RAGQueryResponse(
                query=q_text,
                answer=final_answer,
                confidence_score=round(max(0.0, min(1.0, final_confidence)), 4),
                is_grounded=True,
                citations=citations,
                retrieval_latency_ms=round(retrieval_latency, 2),
                qa_latency_ms=round(qa_latency, 2),
                total_latency_ms=round(total_latency, 2),
            )
