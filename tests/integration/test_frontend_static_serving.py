"""Integration tests for Frontend Static Asset Serving (Task 19)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_frontend_ui_route():
    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ResearchGPT" in resp.text
    assert "Grounded RAG" in resp.text


def test_frontend_css_static_serving():
    resp = client.get("/css/style.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]
    assert "--bg-primary" in resp.text
    assert "glass-card" in resp.text


def test_frontend_js_modules_static_serving():
    resp_app = client.get("/js/app.js")
    assert resp_app.status_code == 200
    assert "javascript" in resp_app.headers["content-type"]
    assert "ApiClient" in resp_app.text

    resp_api = client.get("/js/api.js")
    assert resp_api.status_code == 200
    assert "javascript" in resp_api.headers["content-type"]
    assert "export class ApiClient" in resp_api.text


def test_content_negotiated_root_route():
    # 1. API client asking for JSON (or wildcard)
    api_resp = client.get("/", headers={"accept": "application/json"})
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert "message" in data
    assert "endpoints" in data

    # 2. Browser asking for HTML
    browser_resp = client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert browser_resp.status_code == 200
    assert "text/html" in browser_resp.headers["content-type"]
    assert "<!DOCTYPE html>" in browser_resp.text
