"""
API endpoint tests using FastAPI TestClient.
Run: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_test_image(tmp_path):
    """Create a small synthetic test image."""
    img = tmp_path / "test.jpg"
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 160), 60, (0, 200, 0), -1)
    cv2.imwrite(str(img), frame)
    return img


@pytest.fixture()
def app():
    from src.api import create_app
    return create_app(weights_path="weights/best.onnx")


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["backend"] in {"opencv", "onnxruntime"}

    def test_health_includes_model_name(self, client):
        r = client.get("/health")
        data = r.json()
        assert "best.onnx" in data["model"]


class TestInfoEndpoint:
    def test_info_returns_classes(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert "classes" in data
        assert data["classes"]["0"] == "recyclable"

    def test_openapi_contains_detect_response_schema(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        detect_schema = spec["paths"]["/detect"]["post"]["responses"]["200"]
        assert "content" in detect_schema
        schema_ref = detect_schema["content"]["application/json"]["schema"]["$ref"]
        assert schema_ref.endswith("DetectResponse")


class TestDetectEndpoint:
    def test_detect_image_json(self, client, tmp_path):
        img = _make_test_image(tmp_path)
        with open(img, "rb") as f:
            r = client.post("/detect", files={"file": ("test.jpg", f, "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
        assert "detections" in data
        assert "count" in data
        assert "inference_ms" in data
        assert data["inference_ms"] > 0

    def test_detect_rejects_non_image(self, client, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("not an image")
        with open(txt, "rb") as f:
            r = client.post("/detect", files={"file": ("test.txt", f, "text/plain")})
        assert r.status_code == 400

    def test_detect_returns_annotated_image(self, client, tmp_path):
        img = _make_test_image(tmp_path)
        with open(img, "rb") as f:
            r = client.post(
                "/detect?return_image=true",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert len(r.content) > 0

    def test_detect_with_conf_threshold(self, client, tmp_path):
        img = _make_test_image(tmp_path)
        with open(img, "rb") as f:
            r = client.post(
                "/detect?conf_threshold=0.99",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 200
        data = r.json()
        # at 0.99 confidence, unlikely to have detections on synthetic image
        assert isinstance(data["count"], int)


class TestBatchEndpoint:
    def test_batch_detect(self, client, tmp_path):
        img1 = _make_test_image(tmp_path)
        img2 = _make_test_image(tmp_path)
        files = [
            ("files", ("test1.jpg", open(img1, "rb"), "image/jpeg")),
            ("files", ("test2.jpg", open(img2, "rb"), "image/jpeg")),
        ]
        r = client.post("/detect/batch", files=files)
        for f in files:
            f[1][1].close()
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_rejects_non_image(self, client, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("not an image")
        with open(txt, "rb") as f:
            r = client.post(
                "/detect/batch",
                files=[("files", ("test.txt", f, "text/plain"))],
            )
        assert r.status_code == 200
        data = r.json()
        assert data["results"][0]["error"] == "Not an image"
