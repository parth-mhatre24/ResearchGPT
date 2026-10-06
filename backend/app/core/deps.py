"""Centralized Dependency Injection and Singleton Lifecycle Management for ResearchGPT.

Manages shared singleton instances of NLP models and services to prevent
redundant memory allocations and ensure optimal API latency.
"""

from functools import lru_cache
import logging
from typing import Optional

from backend.app.services.classical_classifier_service import ClassicalClassifierService
from backend.app.services.bert_ner_service import BERTNERService
from backend.app.services.crf_ner_service import CRFNERService
from backend.app.services.summarization_service import SummarizationService
from backend.app.services.similarity_service import SemanticSimilarityService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.flan_t5_service import FlanT5GenerativeService
from backend.app.services.qa_service import QuestionAnsweringService
from backend.app.services.rag_service import RAGService

logger = logging.getLogger("deps")

# ---------------------------------------------------------------------------
# Singleton Cache Providers
# ---------------------------------------------------------------------------

@lru_cache()
def get_chunking_service() -> ChunkingService:
    """Return singleton instance of ChunkingService."""
    return ChunkingService()


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Return singleton instance of EmbeddingService."""
    return EmbeddingService()


@lru_cache()
def get_retrieval_service() -> VectorRetrievalService:
    """Return singleton instance of VectorRetrievalService preloaded with foundational research papers."""
    service = VectorRetrievalService(
        embedding_service=get_embedding_service(),
        chunking_service=get_chunking_service(),
    )
    # Seed foundational research papers for instant out-of-the-box querying
    sample_papers = [
        (
            "vaswani_2017_attention_is_all_you_need",
            "Attention Is All You Need. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, "
            "Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. "
            "Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional "
            "neural networks that include an encoder and a decoder. The best performing models also connect the encoder "
            "and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "Architecture: The Transformer follows this overall architecture using stacked self-attention and point-wise, "
            "fully connected layers for both the encoder and decoder. The encoder is composed of a stack of N = 6 identical layers. "
            "Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. "
            "Literature Survey: Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, "
            "have been firmly established as state of the art approaches in sequence modeling and transduction problems."
        ),
        (
            "devlin_2018_bert",
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. "
            "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. "
            "Abstract: We introduce a new language representation model called BERT, which stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, "
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning "
            "on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one "
            "additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering "
            "and language inference, without substantial task-specific architecture modifications. "
            "Model Architecture: BERT-Base has L=12 layers, H=768 hidden units, A=12 attention heads, with 110 million total parameters. "
            "Pre-training Objective: Masked Language Model (MLM) and Next Sentence Prediction (NSP)."
        )
    ]
    for doc_id, text in sample_papers:
        service.add_document(document_id=doc_id, text=text, chunk_size=40, chunk_overlap=8)
    return service


@lru_cache()
def get_qa_service() -> QuestionAnsweringService:
    """Return singleton instance of QuestionAnsweringService."""
    return QuestionAnsweringService()


@lru_cache()
def get_flan_t5_service() -> FlanT5GenerativeService:
    """Return singleton instance of FlanT5GenerativeService."""
    return FlanT5GenerativeService()


@lru_cache()
def get_rag_service() -> RAGService:
    """Return singleton instance of RAGService."""
    return RAGService(
        retrieval_service=get_retrieval_service(),
        qa_service=get_qa_service(),
        generative_service=get_flan_t5_service(),
    )


@lru_cache()
def get_summarization_service() -> SummarizationService:
    """Return singleton instance of SummarizationService."""
    return SummarizationService()


@lru_cache()
def get_bert_ner_service() -> BERTNERService:
    """Return singleton instance of BERTNERService."""
    return BERTNERService()


@lru_cache()
def get_crf_ner_service() -> CRFNERService:
    """Return singleton instance of CRFNERService."""
    return CRFNERService()


@lru_cache()
def get_similarity_service() -> SemanticSimilarityService:
    """Return singleton instance of SemanticSimilarityService."""
    return SemanticSimilarityService()


@lru_cache()
def get_classical_classifier_service() -> ClassicalClassifierService:
    """Return singleton instance of ClassicalClassifierService."""
    return ClassicalClassifierService(model_type="logistic_regression")
