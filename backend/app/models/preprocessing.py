from pydantic import BaseModel, Field


class ClassicalPreprocessingRequest(BaseModel):
    text: str = Field(..., description="Raw text input to preprocess")
    remove_stopwords: bool = Field(True, description="Whether to remove stopwords")
    lemmatize: bool = Field(True, description="Whether to apply WordNet lemmatization")
    lowercase: bool = Field(True, description="Whether to lowercase tokens")


class ClassicalPreprocessingResponse(BaseModel):
    original_text: str = Field(..., description="Original raw text input")
    sentences: list[str] = Field(..., description="Segmented sentences")
    tokens: list[str] = Field(..., description="Extracted & processed tokens")
    cleaned_text: str = Field(..., description="Rejoined cleaned token string")
    original_char_count: int = Field(..., description="Character count of original text")
    processed_token_count: int = Field(..., description="Total token count after processing")
    stopwords_removed_count: int = Field(0, description="Count of stopwords removed")


class TransformerPreprocessingRequest(BaseModel):
    text: str = Field(..., description="Raw text input to clean")
    normalize_whitespace: bool = Field(True, description="Collapse extra whitespace and linebreaks")
    strip_headers: bool = Field(False, description="Optionally strip page header/footer pattern lines")


class TransformerPreprocessingResponse(BaseModel):
    original_text: str = Field(..., description="Original raw text input")
    cleaned_text: str = Field(..., description="Lightly cleaned, structure-preserving text")
    original_char_count: int = Field(..., description="Character count of original text")
    cleaned_char_count: int = Field(..., description="Character count of cleaned text")
    line_count: int = Field(..., description="Number of preserved lines")
