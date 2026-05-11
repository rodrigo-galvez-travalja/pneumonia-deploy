import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_png(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "classes" in data


def test_predict_valid(client):
    image_bytes = _make_png()
    response = client.post(
        "/predict",
        files={"file": ("xray.png", io.BytesIO(image_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert len(data["probabilities"]) == 3


def test_predict_invalid_mime(client):
    response = client.post(
        "/predict",
        files={"file": ("file.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 422


def test_predict_empty_file(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 422


def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"total_predictions" in response.content
