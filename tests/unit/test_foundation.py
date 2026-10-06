import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path for test discovery
backend_dir = str(Path(__file__).resolve().parents[2] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)


def test_settings():
    assert settings.PROJECT_NAME == "ResearchGPT"
    assert settings.VERSION == "0.1.0"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "ResearchGPT" in data["message"]


def test_health_endpoint():
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["project"] == "ResearchGPT"
