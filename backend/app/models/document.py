from datetime import datetime
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique UUID identifier for the uploaded document")
    filename: str = Field(..., description="Sanitized stored filename")
    original_filename: str = Field(..., description="Original name of the uploaded PDF file")
    file_size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="Validated content type")
    created_at: datetime = Field(..., description="Upload timestamp")
    storage_path: str = Field(..., description="Relative storage path")


class PageExtraction(BaseModel):
    page_number: int = Field(..., description="1-based page index")
    text: str = Field(..., description="Extracted text content of the page")
    char_count: int = Field(..., description="Number of characters extracted from page")


class DocumentExtractionResponse(BaseModel):
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Stored document filename")
    total_pages: int = Field(..., description="Total pages in the PDF document")
    total_characters: int = Field(..., description="Total characters extracted across all pages")
    pages: list[PageExtraction] = Field(..., description="List of per-page text extractions")
    extracted_at: datetime = Field(..., description="Extraction completion timestamp")
