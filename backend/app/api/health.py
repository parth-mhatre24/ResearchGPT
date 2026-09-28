from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", status_code=200)
def health_check():
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }
