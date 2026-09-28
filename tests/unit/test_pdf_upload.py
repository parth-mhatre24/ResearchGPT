import io
import os
import sys
from pathlib import Path

# Add backend directory to sys.path for test discovery
backend_dir = str(Path(__file__).resolve().parents[2] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.services.document_service import sanitize_filename

client = TestClient(app)

SAMPLE_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("my research paper (1).pdf") == "my_research_paper__1_.pdf"
    assert sanitize_filename("test.pdf") == "test.pdf"


def test_valid_pdf_upload():
    file_bytes = io.BytesIO(SAMPLE_PDF_CONTENT)
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("sample_paper.pdf", file_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["original_filename"] == "sample_paper.pdf"
    assert data["file_size_bytes"] == len(SAMPLE_PDF_CONTENT)
    assert "document_id" in data
    assert os.path.exists(data["storage_path"])

    # Clean up uploaded test file
    if os.path.exists(data["storage_path"]):
        os.remove(data["storage_path"])


def test_upload_invalid_extension():
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("sample.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are allowed" in response.json()["detail"]


def test_upload_empty_pdf():
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_upload_invalid_pdf_content():
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("fake.pdf", io.BytesIO(b"NOT A REAL PDF HEADER"), "application/pdf")},
    )
    assert response.status_code == 400
    assert "does not match valid PDF format" in response.json()["detail"]


def test_upload_exceeds_max_size(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 10)
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("sample.pdf", io.BytesIO(SAMPLE_PDF_CONTENT), "application/pdf")},
    )
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"]
