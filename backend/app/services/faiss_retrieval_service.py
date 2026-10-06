"""ResearchGPT Vector Indexing and Semantic Retrieval Service (Task 15 / FR-08).

Manages vector representations of document chunks and performs top-k semantic
similarity retrieval for research queries. Supports FAISS / exact Flat IP search
with index serialization and metadata persistence.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("faiss_retrieval_service")

from backend.app.models.embeddings import DocumentChunk
from backend.app.models.retrieval import (
    IndexInfo,
    RetrievalQuery,
    RetrievalResponse,
    RetrievedDocumentChunk,
)
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import EmbeddingService


class VectorRetrievalService:
    """Service for semantic document chunk indexing and nearest-neighbor vector retrieval."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chunking_service: Optional[ChunkingService] = None,
        dimension: int = 384,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunking_service = chunking_service or ChunkingService()
        self.dimension = dimension

        # Vector storage: (N, D) float32 array
        self.vectors: Optional[np.ndarray] = None
        # Metadata storage: list of DocumentChunk instances
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed and index a list of DocumentChunk objects."""
        if not chunks:
            return 0

        # Compute L2-normalized embeddings for new chunks
        new_vectors = self.embedding_service.embed_chunks(chunks)

        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector index. Total vectors: {len(self.chunks)}")
        return len(chunks)

    def add_document(
        self,
        document_id: str,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        page_number: Optional[int] = None,
    ) -> int:
        """Chunk, embed, and index a raw document string."""
        chunk_resp = self.chunking_service.chunk_text(
            text=text,
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page_number,
        )
        return self.add_chunks(chunk_resp.chunks)

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks and vector embeddings associated with a document_id."""
        if not self.chunks:
            return 0

        keep_indices = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.metadata.document_id != document_id
        ]
        deleted_count = len(self.chunks) - len(keep_indices)

        if deleted_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.vectors is not None and len(self.vectors) > 0:
                if len(keep_indices) == 0:
                    self.vectors = np.zeros((0, self.dimension), dtype=np.float32)
                else:
                    self.vectors = self.vectors[keep_indices]
            logger.info(f"Deleted {deleted_count} chunks for document '{document_id}'. Remaining chunks: {len(self.chunks)}")

        return deleted_count

    def search(
        self,
        query: Union[str, RetrievalQuery],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResponse:
        """Retrieve top-k most semantically relevant chunks for a given query."""
        start_time = time.time()

        if isinstance(query, RetrievalQuery):
            q_text = query.query_text
            k = query.top_k
            threshold = query.score_threshold
            doc_filter = query.document_id
        else:
            q_text = str(query)
            k = top_k
            threshold = score_threshold
            doc_filter = document_id

        if not q_text.strip():
            raise ValueError("Query text cannot be empty")

        if self.vectors is None or len(self.chunks) == 0:
            latency_ms = (time.time() - start_time) * 1000.0
            return RetrievalResponse(
                query=q_text,
                total_retrieved=0,
                top_k=k,
                results=[],
                retrieval_latency_ms=round(latency_ms, 2),
            )

        # Encode query to normalized vector
        q_vec = self.embedding_service.embed_query(q_text)

        # Compute cosine similarity / inner product
        scores = np.dot(self.vectors, q_vec)

        # Optional document_id filtering
        valid_indices = list(range(len(self.chunks)))
        if doc_filter:
            valid_indices = [
                idx for idx in valid_indices if self.chunks[idx].metadata.document_id == doc_filter
            ]

        if not valid_indices:
            latency_ms = (time.time() - start_time) * 1000.0
            return RetrievalResponse(
                query=q_text,
                total_retrieved=0,
                top_k=k,
                results=[],
                retrieval_latency_ms=round(latency_ms, 2),
            )

        filtered_scores = scores[valid_indices]
        # Sort candidate indices by descending similarity score
        sorted_local_order = np.argsort(-filtered_scores)

        results: List[RetrievedDocumentChunk] = []
        rank = 1

        for local_idx in sorted_local_order:
            if len(results) >= k:
                break

            global_idx = valid_indices[local_idx]
            sim_score = float(filtered_scores[local_idx])
            # Clamp similarity score to [0.0, 1.0]
            normalized_score = max(0.0, min(1.0, sim_score))

            if threshold is not None and normalized_score < threshold:
                continue

            chunk = self.chunks[global_idx]
            retrieved_item = RetrievedDocumentChunk(
                rank=rank,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity_score=round(normalized_score, 4),
                metadata=chunk.metadata,
            )
            results.append(retrieved_item)
            rank += 1

        latency_ms = (time.time() - start_time) * 1000.0
        return RetrievalResponse(
            query=q_text,
            total_retrieved=len(results),
            top_k=k,
            results=results,
            retrieval_latency_ms=round(latency_ms, 2),
        )

    def get_index_info(self) -> IndexInfo:
        """Return statistical metadata about the current index state."""
        total = len(self.chunks)
        unique_docs = sorted(list({c.metadata.document_id for c in self.chunks}))

        return IndexInfo(
            total_vectors=total,
            dimension=self.dimension,
            index_type="FlatIP_Cosine",
            unique_documents=len(unique_docs),
            document_ids=unique_docs,
        )

    def save_index(self, save_dir: Union[str, Path]) -> Path:
        """Persist the vector index matrix and chunk metadata to disk."""
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save vectors binary
        if self.vectors is not None:
            np.save(out_path / "vectors.npy", self.vectors)

        # Save metadata JSON
        meta_payload = {
            "dimension": self.dimension,
            "total_vectors": len(self.chunks),
            "chunks": [c.model_dump() for c in self.chunks],
        }
        with open(out_path / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh, indent=2)

        logger.info(f"Saved vector index ({len(self.chunks)} chunks) to {out_path}")
        return out_path

    @classmethod
    def load_index(
        cls,
        load_dir: Union[str, Path],
        embedding_service: Optional[EmbeddingService] = None,
    ) -> "VectorRetrievalService":
        """Load an indexed vector database from disk."""
        in_path = Path(load_dir)
        meta_file = in_path / "metadata.json"
        vec_file = in_path / "vectors.npy"

        if not meta_file.exists():
            raise FileNotFoundError(f"Index metadata file not found at {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as fh:
            meta_data = json.load(fh)

        dimension = meta_data.get("dimension", 384)
        service = cls(embedding_service=embedding_service, dimension=dimension)

        if vec_file.exists():
            service.vectors = np.load(vec_file)
        else:
            service.vectors = np.zeros((0, dimension), dtype=np.float32)

        service.chunks = [DocumentChunk(**c) for c in meta_data.get("chunks", [])]
        logger.info(f"Loaded vector index with {len(service.chunks)} chunks from {in_path}")
        return service
