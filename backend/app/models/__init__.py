"""Models package for data structures and domain entities."""

from backend.app.models.classification import (
    ClassificationEvaluationMetrics,
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.models.document import (
    DocumentExtractionResponse,
    DocumentUploadResponse,
    PageExtraction,
)
from backend.app.models.embeddings import (
    ChunkMetadata,
    ChunkingRequest,
    ChunkingResponse,
    DocumentChunk,
    EmbeddingRequest,
    EmbeddingResponse,
)
from backend.app.models.ner import (
    NEREvaluationMetrics,
    NEREntity,
    NEREntityMetrics,
    NERRequest,
    NERResponse,
)
from backend.app.models.preprocessing import (
    ClassicalPreprocessingRequest,
    ClassicalPreprocessingResponse,
    TransformerPreprocessingRequest,
    TransformerPreprocessingResponse,
)
from backend.app.models.qa import (
    QABatchRequest,
    QABatchResponse,
    QAEvaluationMetrics,
    QARequest,
    QAResponse,
)
from backend.app.models.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    SourceCitation,
)
from backend.app.models.retrieval import (
    IndexInfo,
    RetrievalQuery,
    RetrievalResponse,
    RetrievedDocumentChunk,
)
from backend.app.models.similarity import (
    SimilarityBatchRequest,
    SimilarityBatchResponse,
    SimilarityEvaluationMetrics,
    SimilarityPairRequest,
    SimilarityPairResponse,
)
from backend.app.models.summarization import (
    SummarizationEvaluationMetrics,
    SummarizationRequest,
    SummarizationResponse,
)

__all__ = [
    "ClassificationEvaluationMetrics",
    "ClassificationRequest",
    "ClassificationResponse",
    "DocumentExtractionResponse",
    "DocumentUploadResponse",
    "PageExtraction",
    "NEREvaluationMetrics",
    "NEREntity",
    "NEREntityMetrics",
    "NERRequest",
    "NERResponse",
    "ClassicalPreprocessingRequest",
    "ClassicalPreprocessingResponse",
    "TransformerPreprocessingRequest",
    "TransformerPreprocessingResponse",
    "SummarizationEvaluationMetrics",
    "SummarizationRequest",
    "SummarizationResponse",
    "ChunkMetadata",
    "ChunkingRequest",
    "ChunkingResponse",
    "DocumentChunk",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "IndexInfo",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievedDocumentChunk",
    "SimilarityBatchRequest",
    "SimilarityBatchResponse",
    "SimilarityEvaluationMetrics",
    "SimilarityPairRequest",
    "SimilarityPairResponse",
    "QABatchRequest",
    "QABatchResponse",
    "QAEvaluationMetrics",
    "QARequest",
    "QAResponse",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "SourceCitation",
]
