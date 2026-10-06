"""ResearchGPT Document Chunking Service (Task 14 / FR-08).

Partitions raw research papers or extracted text into coherent semantic chunks
with sliding window overlap, sentence boundary awareness, and provenance metadata.
"""

import re
from typing import List, Optional

from backend.app.models.embeddings import (
    ChunkMetadata,
    ChunkingRequest,
    ChunkingResponse,
    DocumentChunk,
)


class ChunkingService:
    """Service for splitting document text into overlapping semantic chunks."""

    def __init__(self, default_chunk_size: int = 256, default_overlap: int = 32):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str = "doc_default",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        page_number: Optional[int] = None,
    ) -> ChunkingResponse:
        """Partition document text into overlapping chunks respecting sentence boundaries."""
        c_size = chunk_size or self.default_chunk_size
        c_overlap = chunk_overlap or self.default_overlap

        if c_overlap >= c_size:
            c_overlap = max(0, c_size - 1)

        cleaned_text = text.strip()
        if not cleaned_text:
            return ChunkingResponse(
                document_id=document_id,
                total_chunks=0,
                chunks=[],
                avg_chunk_length=0.0,
            )

        # Split text into sentences preserving spans
        sentences, sentence_spans = self._split_sentences_with_spans(cleaned_text)

        chunks: List[DocumentChunk] = []
        curr_sentences: List[str] = []
        curr_word_count = 0
        chunk_idx = 0
        sent_start_idx = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            words_in_sent = len(sent.split())
            curr_sentences.append(sent)
            curr_word_count += words_in_sent

            if curr_word_count >= c_size or i == len(sentences) - 1:
                # Form chunk
                chunk_text = " ".join(curr_sentences)
                first_sent_span = sentence_spans[sent_start_idx]
                last_sent_span = sentence_spans[i]

                meta = ChunkMetadata(
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    page_number=page_number,
                    section_title=self._detect_section_title(chunk_text),
                    token_count=len(chunk_text.split()),
                    char_start=first_sent_span[0],
                    char_end=last_sent_span[1],
                )

                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_idx}",
                    text=chunk_text,
                    metadata=meta,
                )
                chunks.append(chunk)
                chunk_idx += 1

                # If reached end, terminate
                if i == len(sentences) - 1:
                    break

                # Sliding window overlap
                overlap_words = 0
                step_back = 0
                for j in range(len(curr_sentences) - 1, -1, -1):
                    w_count = len(curr_sentences[j].split())
                    overlap_words += w_count
                    step_back += 1
                    if overlap_words >= c_overlap:
                        break

                keep_count = max(1, step_back)
                curr_sentences = curr_sentences[-keep_count:]
                curr_word_count = sum(len(s.split()) for s in curr_sentences)
                sent_start_idx = i - keep_count + 1

            i += 1

        avg_len = sum(c.metadata.token_count for c in chunks) / max(1, len(chunks))

        return ChunkingResponse(
            document_id=document_id,
            total_chunks=len(chunks),
            chunks=chunks,
            avg_chunk_length=round(avg_len, 2),
        )

    def _split_sentences_with_spans(self, text: str) -> tuple[List[str], List[tuple[int, int]]]:
        """Split text into sentences while recording starting and ending character indices."""
        sentence_regex = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")
        sentences = []
        spans = []

        for match in sentence_regex.finditer(text):
            s = match.group().strip()
            if s:
                sentences.append(s)
                spans.append((match.start(), match.end()))

        if not sentences:
            sentences = [text]
            spans = [(0, len(text))]

        return sentences, spans

    def _detect_section_title(self, text: str) -> Optional[str]:
        """Detect probable section headings (e.g. '1. Introduction', 'Abstract', '2. Related Work')."""
        first_line = text.strip().split("\n")[0].strip()
        heading_match = re.match(r"^(?:(?:\d+\.?)+\s+)?(Abstract|Introduction|Related Work|Methodology|Experiments|Results|Discussion|Conclusion|References)\b", first_line, re.IGNORECASE)
        if heading_match:
            return heading_match.group(0).strip()
        return None
