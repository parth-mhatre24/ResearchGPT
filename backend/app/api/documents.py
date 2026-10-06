from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, HTTPException, Path as ApiPath, UploadFile, status

from backend.app.core.deps import get_retrieval_service
from backend.app.models.document import DocumentExtractionResponse, DocumentUploadResponse
from backend.app.services.document_service import delete_uploaded_file, list_uploaded_files, save_pdf_upload
from backend.app.services.faiss_retrieval_service import VectorRetrievalService
from backend.app.services.pdf_extraction_service import extract_document_text_by_id

router = APIRouter()


@router.get(
    "/documents",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="List all uploaded research papers and their indexing status",
)
def get_uploaded_documents(
    retrieval_service: VectorRetrievalService = Depends(get_retrieval_service),
) -> List[Dict[str, Any]]:
    """Retrieve list of all uploaded PDF papers with chunk counts in vector storage."""
    files = list_uploaded_files()
    index_info = retrieval_service.get_index_info()

    # Calculate chunk counts per document
    chunk_counts: Dict[str, int] = {}
    for c in retrieval_service.chunks:
        doc_id = c.metadata.document_id
        chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1

    results = []
    # Include files on disk
    seen_ids = set()
    for f in files:
        doc_id = f["document_id"]
        seen_ids.add(doc_id)
        results.append({
            "document_id": doc_id,
            "filename": f["filename"],
            "original_filename": f["original_filename"],
            "file_size_bytes": f["file_size_bytes"],
            "created_at": f["created_at"],
            "chunks_indexed": chunk_counts.get(doc_id, 0),
            "is_indexed": doc_id in chunk_counts,
        })

    # Also include seeded documents in memory (e.g. Vaswani, BERT)
    for doc_id, count in chunk_counts.items():
        if doc_id not in seen_ids:
            results.append({
                "document_id": doc_id,
                "filename": f"{doc_id}.txt",
                "original_filename": doc_id.replace("_", " ").title(),
                "file_size_bytes": count * 500,
                "created_at": "Pre-indexed",
                "chunks_indexed": count,
                "is_indexed": True,
            })

    return results


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a research paper PDF",
    description="Validates and securely stores an uploaded research paper PDF file.",
)
def upload_document(file: UploadFile = File(...)):
    return save_pdf_upload(file)


@router.post(
    "/documents/{document_id}/extract",
    response_model=DocumentExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract text from uploaded PDF document",
    description="Extracts page-aware text content and metadata from an uploaded PDF document.",
)
def extract_document_text(
    document_id: str = ApiPath(..., description="Unique document ID returned during upload")
):
    return extract_document_text_by_id(document_id)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an uploaded research paper and purge its vectors",
)
def delete_document(
    document_id: str = ApiPath(..., description="Unique document ID to delete"),
    retrieval_service: VectorRetrievalService = Depends(get_retrieval_service),
) -> Dict[str, Any]:
    """Delete the PDF file from disk and remove all indexed vector chunks from the retrieval store."""
    file_deleted = delete_uploaded_file(document_id)
    chunks_deleted = retrieval_service.delete_document(document_id)

    if not file_deleted and chunks_deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found on disk or in vector store.",
        )

    return {
        "document_id": document_id,
        "file_deleted": file_deleted,
        "chunks_purged": chunks_deleted,
        "message": f"Successfully deleted document '{document_id}' and purged {chunks_deleted} vectors.",
    }
