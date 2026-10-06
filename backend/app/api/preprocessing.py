from fastapi import APIRouter, Path as ApiPath, status

from backend.app.models.preprocessing import (
    ClassicalPreprocessingRequest,
    ClassicalPreprocessingResponse,
    TransformerPreprocessingRequest,
    TransformerPreprocessingResponse,
)
from backend.app.services.pdf_extraction_service import extract_document_text_by_id
from backend.app.services.preprocessing_service import preprocess_classical, preprocess_transformer

router = APIRouter()


@router.post(
    "/preprocessing/classical",
    response_model=ClassicalPreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Classical NLP Preprocessing",
    description="Applies sentence segmentation, tokenization, lowercasing, stopword removal, and lemmatization.",
)
def classical_preprocessing(req: ClassicalPreprocessingRequest):
    return preprocess_classical(req)


@router.post(
    "/preprocessing/transformer",
    response_model=TransformerPreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Transformer Compatible Preprocessing",
    description="Applies light structure-preserving text normalization and whitespace cleanup.",
)
def transformer_preprocessing(req: TransformerPreprocessingRequest):
    return preprocess_transformer(req)


@router.post(
    "/preprocessing/document/{document_id}/classical",
    response_model=ClassicalPreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Preprocess uploaded document using classical NLP",
    description="Extracts document text by document_id and runs classical NLP preprocessing.",
)
def classical_preprocess_document(
    document_id: str = ApiPath(..., description="Document ID of an uploaded PDF")
):
    extracted = extract_document_text_by_id(document_id)
    full_text = "\n".join(page.text for page in extracted.pages)
    req = ClassicalPreprocessingRequest(text=full_text)
    return preprocess_classical(req)


@router.post(
    "/preprocessing/document/{document_id}/transformer",
    response_model=TransformerPreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Preprocess uploaded document using transformer pipeline",
    description="Extracts document text by document_id and runs transformer text cleaning.",
)
def transformer_preprocess_document(
    document_id: str = ApiPath(..., description="Document ID of an uploaded PDF")
):
    extracted = extract_document_text_by_id(document_id)
    full_text = "\n".join(page.text for page in extracted.pages)
    req = TransformerPreprocessingRequest(text=full_text)
    return preprocess_transformer(req)
