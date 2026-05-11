import io
import os

import pytest
from PIL import Image

from app.services.wrapper import PneumoniaWrapper

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pth")


@pytest.fixture(scope="module")
def wrapper():
    return PneumoniaWrapper(MODEL_PATH)


def _make_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(80, 80, 80)).save(buf, format="JPEG")
    return buf.getvalue()


def test_wrapper_loads(wrapper):
    assert wrapper.is_loaded


def test_predict_returns_expected_keys(wrapper):
    result = wrapper.predict(_make_jpeg())
    assert "prediction" in result
    assert "label" in result
    assert "confidence" in result
    assert "inference_time_ms" in result
    assert "probabilities" in result


def test_predict_probabilities_sum_to_one(wrapper):
    result = wrapper.predict(_make_jpeg())
    total = sum(p["probability"] for p in result["probabilities"])
    assert abs(total - 1.0) < 1e-4


def test_predict_confidence_in_range(wrapper):
    result = wrapper.predict(_make_jpeg())
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_three_classes(wrapper):
    result = wrapper.predict(_make_jpeg())
    assert len(result["probabilities"]) == 3


def test_predict_invalid_bytes(wrapper):
    with pytest.raises(ValueError):
        wrapper.predict(b"not an image")
