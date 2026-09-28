from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "ResearchGPT"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Document upload settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: set[str] = {".pdf"}
    ALLOWED_CONTENT_TYPES: set[str] = {"application/pdf"}

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
