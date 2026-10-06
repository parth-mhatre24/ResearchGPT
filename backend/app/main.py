"""ResearchGPT Main FastAPI Application (Task 18 / Production API Layer).

Unified production API connecting all modular NLP services:
- PDF Document Ingestion & Extraction
- Classical & Transformer Preprocessing
- Text Classification (Classical & Transformer)
- Named Entity Recognition (BiLSTM-CRF & BERT)
- Abstractive Document Summarization (T5)
- Semantic Textual Similarity (Sentence-BERT & TF-IDF)
- Vector Indexing & Nearest-Neighbor Semantic Retrieval
- Extractive Question Answering (DistilBERT SQuAD)
- Grounded Retrieval-Augmented Generation (RAG) with Provenance Citations
- Integrated Single-Pass Research Paper Analysis
"""

import logging
import time
import uuid
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.analysis import router as analysis_router
from backend.app.api.classification import router as classification_router
from backend.app.api.documents import router as documents_router
from backend.app.api.health import router as health_router
from backend.app.api.ner import router as ner_router
from backend.app.api.preprocessing import router as preprocessing_router
from backend.app.api.qa import router as qa_router
from backend.app.api.rag import router as rag_router
from backend.app.api.retrieval import router as retrieval_router
from backend.app.api.similarity import router as similarity_router
from backend.app.api.summarization import router as summarization_router
from backend.app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("researchgpt_api")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "ResearchGPT — An intelligent, multi-modal NLP research paper assistant "
        "providing automated ingestion, extraction, classification, technical entity recognition, "
        "abstractive summarization, vector retrieval, and grounded question answering with source citations."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development and web frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Process Timing & Request Tracking Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    """Inject X-Request-ID and X-Process-Time-Ms headers into every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled server error for request [{request_id}]: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    process_time = (time.time() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# ---------------------------------------------------------------------------
# Register API Routers under /api/v1
# ---------------------------------------------------------------------------
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(preprocessing_router, prefix=settings.API_V1_STR)
app.include_router(classification_router, prefix=settings.API_V1_STR)
app.include_router(ner_router, prefix=settings.API_V1_STR)
app.include_router(summarization_router, prefix=settings.API_V1_STR)
app.include_router(similarity_router, prefix=settings.API_V1_STR)
app.include_router(retrieval_router, prefix=settings.API_V1_STR)
app.include_router(qa_router, prefix=settings.API_V1_STR)
app.include_router(rag_router, prefix=settings.API_V1_STR)
app.include_router(analysis_router, prefix=settings.API_V1_STR)


from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Mount Frontend Static Directory
# ---------------------------------------------------------------------------
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    css_dir = frontend_dir / "css"
    js_dir = frontend_dir / "js"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")


# ---------------------------------------------------------------------------
# Root Discovery & Frontend Serving Endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["Discovery"])
def root(request: Request):
    """Service discovery endpoint or interactive web UI depending on Accept header."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and (frontend_dir / "index.html").exists():
        return FileResponse(frontend_dir / "index.html")

    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "version": settings.VERSION,
        "ui_url": "/ui/",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": f"{settings.API_V1_STR}/openapi.json",
        "endpoints": {
            "health": f"{settings.API_V1_STR}/health",
            "health_detailed": f"{settings.API_V1_STR}/health/detailed",
            "documents_upload": f"{settings.API_V1_STR}/documents/upload",
            "preprocessing": f"{settings.API_V1_STR}/preprocessing/classical",
            "classification": f"{settings.API_V1_STR}/classification/predict",
            "ner": f"{settings.API_V1_STR}/ner/extract",
            "summarization": f"{settings.API_V1_STR}/summarization/summarize",
            "similarity": f"{settings.API_V1_STR}/similarity/compare",
            "retrieval": f"{settings.API_V1_STR}/retrieval/search",
            "qa": f"{settings.API_V1_STR}/qa/answer",
            "rag": f"{settings.API_V1_STR}/rag/query",
            "paper_analysis": f"{settings.API_V1_STR}/analysis/paper",
        },
    }
