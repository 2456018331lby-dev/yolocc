"""
Comprehensive inference pipeline tests.
Run: pytest tests/test_inference.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import Detection, YOLODetector
from src.cli import main as cli_main


# ── Detection dataclass ─────────────────────────────────────────────────────

class TestDetection:
    """Test the Detection dataclass."""

    def test_creation(self):
        det = Detection(
            class_id=0, class_name="recyclable",
            confidence=0.95, x1=10, y1=20, x2=100, y2=200,
        )
        assert det.class_id == 0
        assert det.class_name == "recyclable"
        assert det.confidence == 0.95

    def test_bbox_property(self):
        det = Detection(class_id=0, class_name="test", confidence=0.9, x1=10, y1=20, x2=100, y2=200)
        assert det.bbox == (10, 20, 100, 200)

    def test_area_property(self):
        det = Detection(class_id=0, class_name="test", confidence=0.9, x1=10, y1=20, x2=110, y2=220)
        assert det.area == 100 * 200

    def test_area_negative_clamped(self):
        det = Detection(class_id=0, class_name="test", confidence=0.9, x1=100, y1=200, x2=10, y2=20)
        assert det.area == 0

    def test_to_dict(self):
        det = Detection(class_id=1, class_name="hazardous", confidence=0.876, x1=0, y1=0, x2=50, y2=50)
        d = det.to_dict()
        assert d["class_id"] == 1
        assert d["class_name"] == "hazardous"
        assert d["confidence"] == 0.876
        assert "x1" in d and "y2" in d


# ── YOLODetector backend detection ──────────────────────────────────────────

class TestBackendDetection:
    """Test backend auto-detection."""

    def test_pt_backend(self):
        det = YOLODetector.__new__(YOLODetector)
        assert det._detect_backend("model.pt") == "ultralytics"

    def test_onnx_backend_auto_selects_available_backend(self):
        det = YOLODetector.__new__(YOLODetector)
        backend = det._detect_backend("model.onnx")
        assert backend in {"opencv", "onnxruntime"}

    def test_unsupported_raises(self):
        det = YOLODetector.__new__(YOLODetector)
        with pytest.raises(ValueError, match="Unsupported"):
            det._detect_backend("model.tflite")

    def test_path_with_dots(self):
        det = YOLODetector.__new__(YOLODetector)
        assert det._detect_backend("/path/to/my.model.onnx") in {"opencv", "onnxruntime"}

    @patch.dict(sys.modules, {"onnxruntime": None})
    def test_onnx_backend_falls_back_to_opencv_when_onnxruntime_missing(self):
        det = YOLODetector.__new__(YOLODetector)
        with patch("builtins.__import__", side_effect=ImportError("onnxruntime missing")):
            assert det._select_onnx_backend() == "opencv"

    def test_explicit_onnxruntime_backend(self):
        det = YOLODetector("weights/best.onnx", backend="onnxruntime")
        assert det.backend == "onnxruntime"


class TestAutoImgsz:
    """Test that ONNX models auto-detect input image size from model metadata."""

    def test_auto_detects_320_onnx(self):
        det = YOLODetector("weights/smoke_best.onnx", imgsz=640)
        assert det.imgsz == 320

    def test_pt_keeps_explicit_imgsz(self):
        det = YOLODetector("weights/smoke_best.pt", imgsz=320)
        assert det.imgsz == 320


# ── Post-processing ─────────────────────────────────────────────────────────

class TestPostProcess:
    """Test OpenCV DNN post-processing."""

    def _make_detector(self):
        det = YOLODetector.__new__(YOLODetector)
        det.imgsz = 640
        det.conf = 0.5
        det.iou = 0.45
        det.classes = None
        det.max_det = 300
        det.class_names = {0: "recyclable", 1: "hazardous", 2: "kitchen", 3: "other"}
        return det

    def test_empty_output(self):
        det = self._make_detector()
        output = np.zeros((1, 8, 100), dtype=np.float32)
        output[0, 4:, :] = 0.1  # All below threshold
        meta = {"orig_shape": (640, 640), "scale": 1.0, "new_shape": (640, 640), "pad": (0, 0)}
        results = det._postprocess_opencv(output, meta)
        assert results == []

    def test_single_detection(self):
        det = self._make_detector()
        det.conf = 0.1  # Lower threshold

        # Create output with one detection
        output = np.zeros((1, 8, 8400), dtype=np.float32)
        # Set box: cx=320, cy=320, w=100, h=100
        output[0, 0, 0] = 320.0  # cx
        output[0, 1, 0] = 320.0  # cy
        output[0, 2, 0] = 100.0  # w
        output[0, 3, 0] = 100.0  # h
        output[0, 4, 0] = 0.9   # class 0 score
        output[0, 5:, 0] = 0.01  # other classes

        meta = {"orig_shape": (640, 640), "scale": 1.0, "new_shape": (640, 640), "pad": (0, 0)}
        results = det._postprocess_opencv(output, meta)
        assert len(results) == 1
        assert results[0].class_name == "recyclable"
        assert results[0].confidence > 0.1

    def test_class_filter(self):
        det = self._make_detector()
        det.conf = 0.1
        det.classes = [1]  # Only hazardous

        output = np.zeros((1, 8, 8400), dtype=np.float32)
        output[0, 0, 0] = 320.0
        output[0, 1, 0] = 320.0
        output[0, 2, 0] = 100.0
        output[0, 3, 0] = 100.0
        output[0, 4, 0] = 0.9   # class 0 — should be filtered
        output[0, 5, 0] = 0.01

        meta = {"orig_shape": (640, 640), "scale": 1.0, "new_shape": (640, 640), "pad": (0, 0)}
        results = det._postprocess_opencv(output, meta)
        assert len(results) == 0  # class 0 filtered out


# ── Drawing ─────────────────────────────────────────────────────────────────

class TestDrawing:
    """Test detection drawing on frames."""

    def _make_detector(self):
        det = YOLODetector.__new__(YOLODetector)
        det.class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        return det

    def test_draw_single_detection(self):
        det = self._make_detector()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="recyclable", confidence=0.95, x1=100, y1=100, x2=300, y2=300),
        ]
        result = det.draw(frame, detections)
        assert result.shape == (640, 640, 3)
        assert result.sum() > 0  # Something was drawn

    def test_draw_preserves_shape(self):
        det = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.draw(frame, [])
        assert result.shape == (480, 640, 3)

    def test_draw_multiple(self):
        det = self._make_detector()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="a", confidence=0.9, x1=10, y1=10, x2=100, y2=100),
            Detection(class_id=1, class_name="b", confidence=0.8, x1=200, y1=200, x2=400, y2=400),
        ]
        result = det.draw(frame, detections, show_fps=30.5)
        assert result.sum() > 0


# ── Preprocessing ───────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_blob_shape(self):
        det = YOLODetector.__new__(YOLODetector)
        det.imgsz = 640
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, meta = det._preprocess(frame)
        assert blob.shape == (1, 3, 640, 640)
        assert meta["orig_shape"] == (480, 640)

    def test_different_sizes(self):
        det = YOLODetector.__new__(YOLODetector)
        det.imgsz = 320
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        blob, meta = det._preprocess(frame)
        assert blob.shape == (1, 3, 320, 320)
        assert meta["orig_shape"] == (1080, 1920)

    def test_letterbox_preserves_aspect_ratio(self):
        det = YOLODetector.__new__(YOLODetector)
        det.imgsz = 640
        frame = np.zeros((300, 600, 3), dtype=np.uint8)  # 2:1 image
        blob, meta = det._preprocess(frame)
        assert blob.shape == (1, 3, 640, 640)
        # Should scale to 640x320, pad 160px top/bottom
        assert meta["scale"] == pytest.approx(640 / 600)
        assert meta["new_shape"] == (320, 640)
        assert meta["pad"] == (0, 160)


class TestDetectFrameOverrides:
    def _make_detector(self):
        det = YOLODetector.__new__(YOLODetector)
        det.backend = "opencv"
        det.conf = 0.5
        det.iou = 0.45
        det.classes = None
        det.max_det = 300
        det.imgsz = 640
        det.class_names = {0: "recyclable", 1: "hazardous", 2: "kitchen", 3: "other"}
        meta = {"orig_shape": (640, 640), "scale": 1.0, "new_shape": (640, 640), "pad": (0, 0)}
        det._preprocess = MagicMock(return_value=(np.zeros((1, 3, 640, 640), dtype=np.float32), meta))
        det._net = MagicMock()
        output = np.zeros((1, 8, 10), dtype=np.float32)
        output[0, 0, 0] = 320.0
        output[0, 1, 0] = 320.0
        output[0, 2, 0] = 100.0
        output[0, 3, 0] = 100.0
        output[0, 4, 0] = 0.9
        det._net.forward.return_value = output
        return det

    def test_conf_override_does_not_persist(self):
        det = self._make_detector()
        results = det.detect_frame(np.zeros((640, 640, 3), dtype=np.uint8), conf=0.1)
        assert len(results) == 1
        assert det.conf == 0.5


class TestCliDetect:
    def test_detect_prints_dataclass_detection(self, tmp_path):
        from click.testing import CliRunner

        image = tmp_path / "input.jpg"
        save = tmp_path / "out.jpg"
        image.write_bytes(b"fake")
        detection = Detection(0, "recyclable", 0.95, 1, 2, 3, 4)

        with patch("src.inference.YOLODetector") as mock_detector:
            instance = mock_detector.return_value
            instance.detect_image.return_value = [detection]
            result = CliRunner().invoke(
                cli_main,
                ["detect", "--source", str(image), "--weights", "weights/fake.onnx", "--save", str(save)],
            )

        assert result.exit_code == 0
        assert "recyclable" in result.output
        assert "[1,2,3,4]" in result.output

    def test_serve_passes_imgsz_to_api_factory(self):
        from click.testing import CliRunner

        with patch("src.api.create_app") as create_app, patch("uvicorn.run") as uvicorn_run:
            create_app.return_value = object()
            result = CliRunner().invoke(
                cli_main,
                ["serve", "--weights", "weights/best.onnx", "--imgsz", "320", "--host", "127.0.0.1", "--port", "8765"],
            )

        assert result.exit_code == 0
        create_app.assert_called_once_with(weights_path="weights/best.onnx", imgsz=320, backend=None)
        uvicorn_run.assert_called_once()

    def test_detect_passes_backend(self, tmp_path):
        from click.testing import CliRunner

        image = tmp_path / "input.jpg"
        image.write_bytes(b"fake")
        with patch("src.inference.YOLODetector") as mock_detector:
            instance = mock_detector.return_value
            instance.detect_image.return_value = []
            result = CliRunner().invoke(
                cli_main,
                ["detect", "--source", str(image), "--weights", "weights/best.onnx", "--backend", "onnxruntime", "--no-show"],
            )

        assert result.exit_code == 0
        kwargs = mock_detector.call_args.kwargs
        assert kwargs["backend"] == "onnxruntime"


# ── Dataset generation ──────────────────────────────────────────────────────

class TestDatasetGeneration:

    def test_generate_creates_structure(self, tmpdir):
        from scripts.prepare_dataset import generate_synthetic_dataset
        out = Path(str(tmpdir)) / "dataset"
        generate_synthetic_dataset(str(out), num_images=10, seed=123)

        assert (out / "images" / "train").exists()
        assert (out / "labels" / "train").exists()
        assert (out / "images" / "val").exists()
        assert (out / "labels" / "val").exists()
        assert (out / "images" / "test").exists()
        assert (out / "labels" / "test").exists()

    def test_generate_correct_counts(self, tmpdir):
        from scripts.prepare_dataset import generate_synthetic_dataset
        out = Path(str(tmpdir)) / "dataset"
        generate_synthetic_dataset(str(out), num_images=20, seed=42)

        train_imgs = list((out / "images" / "train").glob("*.jpg"))
        val_imgs = list((out / "images" / "val").glob("*.jpg"))
        test_imgs = list((out / "images" / "test").glob("*.jpg"))

        assert len(train_imgs) == 14  # 70%
        assert len(val_imgs) == 3     # 15%
        assert len(test_imgs) == 3    # 15%

    def test_label_format_valid(self, tmpdir):
        from scripts.prepare_dataset import generate_synthetic_dataset
        out = Path(str(tmpdir)) / "dataset"
        generate_synthetic_dataset(str(out), num_images=5, seed=1)

        for lbl_file in (out / "labels" / "train").glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5
                    cls_id = int(parts[0])
                    assert 0 <= cls_id <= 3
                    for v in parts[1:]:
                        fv = float(v)
                        assert 0 <= fv <= 1

    def test_validate_dataset_rejects_bad_class_and_box_size(self, tmpdir):
        from scripts.prepare_dataset import generate_synthetic_dataset, validate_dataset

        out = Path(str(tmpdir)) / "dataset"
        generate_synthetic_dataset(str(out), num_images=3, seed=1)
        label = next((out / "labels" / "train").glob("*.txt"))
        label.write_text("9 0.5 0.5 0.0 0.2\n", encoding="utf-8")

        stats = validate_dataset(str(out))
        assert stats["errors"] >= 2

    def test_images_are_valid(self, tmpdir):
        from scripts.prepare_dataset import generate_synthetic_dataset
        out = Path(str(tmpdir)) / "dataset"
        generate_synthetic_dataset(str(out), num_images=3, seed=1)

        for img_file in (out / "images" / "train").glob("*.jpg"):
            img = cv2.imread(str(img_file))
            assert img is not None
            assert img.shape[0] > 0 and img.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
