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

client = TestClient(app)

SAMPLE_TEXT = (
    "ResearchGPT processes research papers using natural language processing algorithms. "
    "The system extracts technical entities and classifies document domains."
)


def test_classical_preprocessing_pipeline():
    response = client.post(
        f"{settings.API_V1_STR}/preprocessing/classical",
        json={
            "text": SAMPLE_TEXT,
            "remove_stopwords": True,
            "lemmatize": True,
            "lowercase": True,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["original_text"] == SAMPLE_TEXT
    assert len(data["sentences"]) == 2
    assert "algorithms" not in data["tokens"]  # Should be lemmatized to 'algorithm'
    assert "algorithm" in data["tokens"]
    assert "the" not in data["tokens"]  # Stopword removed
    assert data["stopwords_removed_count"] > 0
    assert len(data["cleaned_text"]) > 0


def test_transformer_preprocessing_pipeline():
    raw_input = "Page 1 of 5\n\nResearchGPT   preserves   full context.\n   Line wraps and  punctuation!  "
    response = client.post(
        f"{settings.API_V1_STR}/preprocessing/transformer",
        json={
            "text": raw_input,
            "normalize_whitespace": True,
            "strip_headers": True,
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Header "Page 1 of 5" should be stripped
    assert "Page 1 of 5" not in data["cleaned_text"]
    # Punctuation and casing preserved
    assert "ResearchGPT preserves full context." in data["cleaned_text"]
    assert "punctuation!" in data["cleaned_text"]
    assert data["line_count"] == 2


def test_preprocessing_edge_cases():
    # Empty string
    empty_res = client.post(
        f"{settings.API_V1_STR}/preprocessing/classical",
        json={"text": "   \n\t  "},
    )
    assert empty_res.status_code == 200
    assert empty_res.json()["processed_token_count"] == 0
    assert empty_res.json()["cleaned_text"] == ""

    # Special characters and numbers
    special_res = client.post(
        f"{settings.API_V1_STR}/preprocessing/transformer",
        json={"text": "BERT & DistilBERT models @ 99.5% accuracy! #NLP"},
    )
    assert special_res.status_code == 200
    assert "99.5%" in special_res.json()["cleaned_text"]


def test_document_preprocessing_endpoints():
    pdf_bytes = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 64 >>
stream
BT
/F1 12 Tf
100 700 Td
(ResearchGPT Preprocessing Pipelines test paper) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000318 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
433
%%EOF"""

    # Upload document
    upload_res = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("prep_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]
    storage_path = upload_res.json()["storage_path"]

    try:
        # Test document classical preprocessing
        class_res = client.post(f"{settings.API_V1_STR}/preprocessing/document/{doc_id}/classical")
        assert class_res.status_code == 200
        assert "researchgpt" in class_res.json()["tokens"]

        # Test document transformer preprocessing
        trans_res = client.post(f"{settings.API_V1_STR}/preprocessing/document/{doc_id}/transformer")
        assert trans_res.status_code == 200
        assert "ResearchGPT Preprocessing Pipelines" in trans_res.json()["cleaned_text"]

    finally:
        if os.path.exists(storage_path):
            os.remove(storage_path)
