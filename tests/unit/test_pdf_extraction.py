import io
import os
import sys
from pathlib import Path

# Add backend directory to sys.path for test discovery
backend_dir = str(Path(__file__).resolve().parents[2] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)

# Minimal 2-page valid PDF with extractable text streams
SAMPLE_MULTIPAGE_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 56 >>
stream
BT
/F1 12 Tf
100 700 Td
(ResearchGPT Section 1 Introduction) Tj
ET
endstream
endobj
6 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >>
endobj
7 0 obj
<< /Length 58 >>
stream
BT
/F1 12 Tf
100 700 Td
(ResearchGPT Section 2 Architecture) Tj
ET
endstream
endobj
xref
0 8
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000122 00000 n 
0000000251 00000 n 
0000000325 00000 n 
0000000432 00000 n 
0000000561 00000 n 
trailer
<< /Size 8 /Root 1 0 R >>
startxref
670
%%EOF"""


def test_pdf_extraction_workflow():
    # 1. Upload sample multipage PDF
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("paper.pdf", io.BytesIO(SAMPLE_MULTIPAGE_PDF), "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["document_id"]
    storage_path = doc_data["storage_path"]

    try:
        # 2. Extract text from uploaded document
        extract_res = client.post(f"{settings.API_V1_STR}/documents/{doc_id}/extract")
        assert extract_res.status_code == 200
        ext_data = extract_res.json()

        assert ext_data["document_id"] == doc_id
        assert ext_data["total_pages"] == 2
        assert len(ext_data["pages"]) == 2

        # Verify page 1
        page1 = ext_data["pages"][0]
        assert page1["page_number"] == 1
        assert "ResearchGPT Section 1 Introduction" in page1["text"]

        # Verify page 2
        page2 = ext_data["pages"][1]
        assert page2["page_number"] == 2
        assert "ResearchGPT Section 2 Architecture" in page2["text"]

        assert ext_data["total_characters"] == len(page1["text"]) + len(page2["text"])

    finally:
        if os.path.exists(storage_path):
            os.remove(storage_path)


def test_extract_nonexistent_document():
    response = client.post(f"{settings.API_V1_STR}/documents/nonexistent-doc-id-12345/extract")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_extract_corrupted_pdf_file():
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    corrupted_doc_id = "corrupted-test-doc-id"
    corrupted_path = settings.UPLOAD_DIR / f"{corrupted_doc_id}_bad.pdf"

    # Create fake invalid PDF header file
    with open(corrupted_path, "wb") as f:
        f.write(b"%PDF-1.4 Fake header but garbage content $$$$$$$")

    try:
        response = client.post(f"{settings.API_V1_STR}/documents/{corrupted_doc_id}/extract")
        assert response.status_code == 400
        assert "corrupted" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()
    finally:
        if corrupted_path.exists():
            os.remove(corrupted_path)
