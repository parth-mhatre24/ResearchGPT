from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, status
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.app.core.config import settings
from backend.app.models.document import DocumentExtractionResponse, PageExtraction


def locate_uploaded_pdf(document_id: str) -> Path:
    """Locate stored PDF file by document_id in upload directory."""
    if not settings.UPLOAD_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload directory does not exist.",
        )

    matches = list(settings.UPLOAD_DIR.glob(f"{document_id}_*.pdf"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return matches[0]


def extract_text_from_pdf_path(pdf_path: Path, document_id: str) -> DocumentExtractionResponse:
    """Extract page-aware text content from a PDF file path."""
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file at path '{pdf_path}' not found.",
        )

    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid PDF file: {str(err)}",
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to open PDF file: {str(err)}",
        )

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Encrypted or password-protected PDF files are not supported.",
            )

    pages: list[PageExtraction] = []
    total_chars = 0

    for idx, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        clean_text = raw_text.strip()
        char_count = len(clean_text)
        total_chars += char_count

        pages.append(
            PageExtraction(
                page_number=idx,
                text=clean_text,
                char_count=char_count,
            )
        )

    return DocumentExtractionResponse(
        document_id=document_id,
        filename=pdf_path.name,
        total_pages=len(reader.pages),
        total_characters=total_chars,
        pages=pages,
        extracted_at=datetime.now(timezone.utc),
    )


def extract_document_text_by_id(document_id: str) -> DocumentExtractionResponse:
    """Locate and extract text for an uploaded document ID."""
    pdf_path = locate_uploaded_pdf(document_id)
    return extract_text_from_pdf_path(pdf_path, document_id)
