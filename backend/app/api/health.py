"""Health and System Status API Router (Task 18)."""

from fastapi import APIRouter
from backend.app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=200, summary="Basic liveness check")
def health_check():
    """Return 200 OK if service is alive."""
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


@router.get("/health/detailed", status_code=200, summary="Deep readiness and environment check")
def detailed_health_check():
    """Return system environment, PyTorch availability, and active service capabilities."""
    import sys
    try:
        import torch
        torch_avail = True
        cuda_avail = torch.cuda.is_available()
        torch_version = torch.__version__
    except Exception:
        torch_avail = False
        cuda_avail = False
        torch_version = "unavailable"

    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "python_version": sys.version.split()[0],
        "torch_available": torch_avail,
        "cuda_available": cuda_avail,
        "torch_version": torch_version,
        "services": {
            "pdf_extraction": True,
            "preprocessing": True,
            "classical_classification": True,
            "transformer_classification": torch_avail,
            "bilstm_crf_ner": torch_avail,
            "bert_ner": torch_avail,
            "summarization": torch_avail,
            "semantic_similarity": torch_avail,
            "vector_retrieval": torch_avail,
            "question_answering": torch_avail,
            "rag_pipeline": torch_avail,
        },
    }
