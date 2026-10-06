import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from backend.app.core.config import settings
from backend.app.models.document import DocumentUploadResponse


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal and unsafe characters."""
    basename = Path(filename).name
    clean_name = re.sub(r"[^\w\.-]", "_", basename)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"document_{clean_name}"
    return clean_name


def save_pdf_upload(file: UploadFile) -> DocumentUploadResponse:
    """Validate and securely store an uploaded PDF file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    original_name = file.filename
    clean_name = sanitize_filename(original_name)

    # Validate file extension
    ext = Path(clean_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{ext}'. Only PDF files are allowed.",
        )

    # Read content for validation
    try:
        content = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )
    finally:
        file.file.seek(0)

    file_size = len(content)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_size} bytes) exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes.",
        )

    # Validate PDF magic header bytes
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content header does not match valid PDF format (%PDF-).",
        )

    # Ensure upload directory exists
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique ID and target path
    doc_id = str(uuid.uuid4())
    stored_filename = f"{doc_id}_{clean_name}"
    destination_path = settings.UPLOAD_DIR / stored_filename

    # Save to disk
    with open(destination_path, "wb") as f:
        f.write(content)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=stored_filename,
        original_filename=original_name,
        file_size_bytes=file_size,
        content_type=file.content_type or "application/pdf",
        created_at=datetime.now(timezone.utc),
        storage_path=str(destination_path),
    )


def list_uploaded_files() -> list[dict]:
    """List all stored PDF files in upload directory."""
    if not settings.UPLOAD_DIR.exists():
        return []

    files = []
    for p in settings.UPLOAD_DIR.glob("*_*.pdf"):
        parts = p.name.split("_", 1)
        doc_id = parts[0]
        orig_name = parts[1] if len(parts) > 1 else p.name
        files.append({
            "document_id": doc_id,
            "filename": p.name,
            "original_filename": orig_name,
            "file_size_bytes": p.stat().st_size,
            "created_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        })
    return sorted(files, key=lambda x: x["created_at"], reverse=True)


def delete_uploaded_file(document_id: str) -> bool:
    """Delete uploaded PDF file from disk for a given document_id."""
    if not settings.UPLOAD_DIR.exists():
        return False

    matches = list(settings.UPLOAD_DIR.glob(f"{document_id}_*.pdf"))
    deleted = False
    for m in matches:
        try:
            m.unlink()
            deleted = True
        except Exception:
            pass
    return deleted
