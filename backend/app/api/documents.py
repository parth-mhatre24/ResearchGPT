from fastapi import APIRouter, File, Path as ApiPath, UploadFile, status

from app.models.document import DocumentExtractionResponse, DocumentUploadResponse
from app.services.document_service import save_pdf_upload
from app.services.pdf_extraction_service import extract_document_text_by_id

router = APIRouter()


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
