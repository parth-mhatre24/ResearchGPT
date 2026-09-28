from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.preprocessing import router as preprocessing_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

app.include_router(health_router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(documents_router, prefix=settings.API_V1_STR, tags=["Documents"])
app.include_router(preprocessing_router, prefix=settings.API_V1_STR, tags=["Preprocessing"])


@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }
