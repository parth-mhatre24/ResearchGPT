import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from app.models.preprocessing import (
    ClassicalPreprocessingRequest,
    ClassicalPreprocessingResponse,
    TransformerPreprocessingRequest,
    TransformerPreprocessingResponse,
)

# Ensure required NLTK resources are available
_NLTK_INITIALIZED = False


def _ensure_nltk_data():
    global _NLTK_INITIALIZED
    if not _NLTK_INITIALIZED:
        for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        _NLTK_INITIALIZED = True


def preprocess_classical(req: ClassicalPreprocessingRequest) -> ClassicalPreprocessingResponse:
    """Execute classical NLP preprocessing pipeline."""
    _ensure_nltk_data()
    raw_text = req.text or ""
    if not raw_text.strip():
        return ClassicalPreprocessingResponse(
            original_text=raw_text,
            sentences=[],
            tokens=[],
            cleaned_text="",
            original_char_count=len(raw_text),
            processed_token_count=0,
            stopwords_removed_count=0,
        )

    # Sentence segmentation
    try:
        sentences = sent_tokenize(raw_text)
    except Exception:
        sentences = [s.strip() for s in raw_text.split(".") if s.strip()]

    # Tokenization
    try:
        raw_tokens = word_tokenize(raw_text)
    except Exception:
        raw_tokens = re.findall(r"\b\w+\b", raw_text)

    stop_words = set(stopwords.words("english")) if req.remove_stopwords else set()
    lemmatizer = WordNetLemmatizer() if req.lemmatize else None

    processed_tokens: list[str] = []
    stopwords_removed = 0

    for token in raw_tokens:
        # Keep alphanumeric tokens
        if not token.isalnum():
            continue

        tok = token.lower() if req.lowercase else token

        if req.remove_stopwords and tok.lower() in stop_words:
            stopwords_removed += 1
            continue

        if lemmatizer:
            tok = lemmatizer.lemmatize(tok)

        processed_tokens.append(tok)

    cleaned_text = " ".join(processed_tokens)

    return ClassicalPreprocessingResponse(
        original_text=raw_text,
        sentences=sentences,
        tokens=processed_tokens,
        cleaned_text=cleaned_text,
        original_char_count=len(raw_text),
        processed_token_count=len(processed_tokens),
        stopwords_removed_count=stopwords_removed,
    )


def preprocess_transformer(req: TransformerPreprocessingRequest) -> TransformerPreprocessingResponse:
    """Execute light, non-destructive transformer preprocessing pipeline."""
    raw_text = req.text or ""
    if not raw_text.strip():
        return TransformerPreprocessingResponse(
            original_text=raw_text,
            cleaned_text="",
            original_char_count=len(raw_text),
            cleaned_char_count=0,
            line_count=0,
        )

    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if req.strip_headers:
            # Strip standard header/footer patterns (e.g. Page 1, Page 1 of 5)
            if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", stripped, re.IGNORECASE):
                continue

        if req.normalize_whitespace:
            stripped = re.sub(r"[ \t]+", " ", stripped)

        cleaned_lines.append(stripped)

    cleaned_text = "\n".join(cleaned_lines)

    return TransformerPreprocessingResponse(
        original_text=raw_text,
        cleaned_text=cleaned_text,
        original_char_count=len(raw_text),
        cleaned_char_count=len(cleaned_text),
        line_count=len(cleaned_lines),
    )
