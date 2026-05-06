"""
Unified Inference Engine — Industrial Grade
Supports: PyTorch (.pt), OpenCV DNN (.onnx), ONNX Runtime (.onnx)
Features:
  - Automatic backend selection with fallback
  - Class filtering
  - FPS counter overlay
  - Batch-ready architecture
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import get_class_colors, get_class_names, load_dataset_config
from src.logger import get_logger

log = get_logger("inference")


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
        }


class YOLODetector:
    """
    Unified YOLO detector supporting PT and ONNX backends.

    Usage:
        det = YOLODetector("best.onnx", conf=0.25)
        results = det.detect_frame(frame)
        annotated = det.draw(frame, results)
    """

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        data_config: str = "configs/garbage.yaml",
        backend: Optional[str] = None,
    ):
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        self.max_det = max_det
        self.backend_override = backend

        # Load class info
        try:
            self.class_names = get_class_names(data_config)
            self.class_colors = get_class_colors(data_config)
        except Exception:
            self.class_names = {
                0: "recyclable", 1: "hazardous",
                2: "kitchen", 3: "other",
            }
            self.class_colors = {
                0: (0, 255, 0), 1: (0, 0, 255),
                2: (255, 0, 0), 3: (128, 128, 128),
            }

        self.backend = self._detect_backend(weights)
        self._net = None
        self._model = None
        self._ort_session = None
        self._ort_input_name = None
        self._load()

    def _detect_backend(self, weights: str) -> str:
        backend_override = getattr(self, "backend_override", None)
        if backend_override is not None:
            return backend_override
        ext = Path(weights).suffix.lower()
        if ext == ".onnx":
            return self._select_onnx_backend()
        elif ext == ".pt":
            return "ultralytics"
        raise ValueError(f"Unsupported weight format: {ext}. Use .pt or .onnx")

    def _select_onnx_backend(self) -> str:
        """Choose the best available ONNX backend.

        Policy:
        1. Prefer ONNX Runtime when importable (better portability/perf consistency)
        2. Fall back to OpenCV DNN otherwise
        """
        try:
            import onnxruntime  # noqa: F401
            return "onnxruntime"
        except Exception:
            return "opencv"

    def _load(self):
        if self.backend == "ultralytics":
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            log.info(f"Loaded Ultralytics model: {self.weights}")
        elif self.backend == "onnxruntime":
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if "cuda" in self.device.lower():
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(self.weights, providers=providers)
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            self._auto_detect_onnx_imgsz()
            log.info(f"Loaded ONNX model via ONNX Runtime: {self.weights} providers={providers}")
        elif self.backend == "opencv":
            self._net = cv2.dnn.readNetFromONNX(self.weights)
            if "cuda" in self.device.lower():
                try:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    log.info("Using CUDA backend")
                except Exception:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    log.warning("CUDA unavailable, falling back to CPU")
            else:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Auto-detect input image size from ONNX model metadata
            self._auto_detect_onnx_imgsz()

            log.info(f"Loaded ONNX model via OpenCV DNN: {self.weights}")

    def _auto_detect_onnx_imgsz(self):
        """Auto-detect fixed square ONNX input size from model metadata."""
        try:
            import onnx
            onnx_model = onnx.load(self.weights)
            onnx_h = onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
            onnx_w = onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
            if onnx_h == onnx_w and onnx_h > 0:
                if onnx_h != self.imgsz:
                    log.info(f"Auto-detected ONNX input size {onnx_h} (was {self.imgsz})")
                self.imgsz = onnx_h
        except Exception as e:
            log.warning(f"Could not auto-detect ONNX input size, using {self.imgsz}: {e}")

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Letterbox preprocess to preserve aspect ratio like Ultralytics YOLO.

        Returns:
            blob: (1, 3, imgsz, imgsz) float32 normalized input
            meta: {
                "orig_shape": (h, w),
                "scale": scale_factor,
                "new_shape": (new_h, new_w),
                "pad": (pad_x, pad_y),
            }
        """
        h, w = frame.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        # Resize while preserving aspect ratio
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding (centered)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2

        # Letterbox canvas with YOLO default gray (114)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.imgsz, self.imgsz),
            swapRB=True, crop=False,
        )
        meta = {
            "orig_shape": (h, w),
            "scale": scale,
            "new_shape": (new_h, new_w),
            "pad": (pad_x, pad_y),
        }
        return blob, meta

    def _postprocess_opencv(self, output: np.ndarray, meta: Dict) -> List[Detection]:
        """Post-process YOLOv8 OpenCV DNN output with letterbox coordinate restore."""
        h, w = meta["orig_shape"]
        scale = meta["scale"]
        pad_x, pad_y = meta["pad"]
        # YOLOv8 output: (1, 4+nc, num_preds) -> transpose to (num_preds, 4+nc)
        preds = output[0].T  # (8400, 84)

        cx = preds[:, 0]
        cy = preds[:, 1]
        bw = preds[:, 2]
        bh = preds[:, 3]
        scores = preds[:, 4:]

        cls_ids = np.argmax(scores, axis=1)
        cls_scores = scores[np.arange(len(scores)), cls_ids]

        # Confidence filter
        mask = cls_scores > self.conf
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        cls_ids, cls_scores = cls_ids[mask], cls_scores[mask]

        # Class filter
        if self.classes is not None:
            cls_mask = np.isin(cls_ids, self.classes)
            cx, cy, bw, bh = cx[cls_mask], cy[cls_mask], bw[cls_mask], bh[cls_mask]
            cls_ids, cls_scores = cls_ids[cls_mask], cls_scores[cls_mask]

        if len(cx) == 0:
            return []

        # Convert to x1y1x2y2 in original coords (undo letterbox padding + scale)
        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, w - 1).astype(int)
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, h - 1).astype(int)
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, w - 1).astype(int)
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, h - 1).astype(int)

        # NMS
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        confs = cls_scores.tolist()
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confs, self.conf, self.iou)

        results = []
        if len(indices) > 0:
            for i in indices.flatten()[:self.max_det]:
                cid = int(cls_ids[i])
                results.append(Detection(
                    class_id=cid,
                    class_name=self.class_names.get(cid, f"class_{cid}"),
                    confidence=float(cls_scores[i]),
                    x1=int(x1[i]), y1=int(y1[i]),
                    x2=int(x2[i]), y2=int(y2[i]),
                ))
        return results

    def detect_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Detection]:
        """Run detection on a single BGR frame.

        Optional ``conf``/``iou`` overrides are applied only for this call.
        This avoids mutating detector state in API requests and keeps inference
        safe when multiple requests are handled concurrently.
        """
        original_conf, original_iou = self.conf, self.iou
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou

        try:
            if self.backend == "ultralytics":
                kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False}
                if self.classes is not None:
                    kwargs["classes"] = self.classes
                results = self._model(frame, **kwargs)

                detections = []
                for r in results:
                    for box in r.boxes:
                        cid = int(box.cls[0])
                        detections.append(Detection(
                            class_id=cid,
                            class_name=self.class_names.get(cid, f"class_{cid}"),
                            confidence=float(box.conf[0]),
                            x1=int(box.xyxy[0][0]),
                            y1=int(box.xyxy[0][1]),
                            x2=int(box.xyxy[0][2]),
                            y2=int(box.xyxy[0][3]),
                        ))
                return detections[:self.max_det]

            if self.backend == "onnxruntime":
                blob, meta = self._preprocess(frame)
                output = self._ort_session.run(None, {self._ort_input_name: blob})[0]
                return self._postprocess_opencv(output, meta)

            if self.backend == "opencv":
                blob, meta = self._preprocess(frame)
                self._net.setInput(blob)
                output = self._net.forward()
                return self._postprocess_opencv(output, meta)

            return []
        finally:
            self.conf, self.iou = original_conf, original_iou

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_fps: float = 0.0,
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Draw detections on frame with labels."""
        for det in detections:
            color = self.class_colors.get(det.class_id, (255, 255, 255))
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label background
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 4),
                (x1 + tw, y1),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )

        # FPS overlay
        if show_fps > 0:
            cv2.putText(
                frame, f"FPS: {show_fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

        # Detection count
        cv2.putText(
            frame, f"Objects: {len(detections)}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255), 2, cv2.LINE_AA,
        )

        return frame

    def detect_image(self, image_path: str, save_path: Optional[str] = None) -> List[Detection]:
        """Run detection on a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        detections = self.detect_frame(frame)
        annotated = self.draw(frame.copy(), detections)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, annotated)
            log.info(f"Saved result: {save_path}")

        return detections

    def detect_video(
        self,
        source: Union[str, int],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> List[Detection]:
        """Run detection on video source (file path or webcam index)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        all_detections: List[Detection] = []
        frame_count = 0
        total_time = 0.0

        log.info(f"Processing video: {source} ({width}x{height} @ {fps}fps)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                detections = self.detect_frame(frame)
                t1 = time.time()

                total_time += t1 - t0
                frame_count += 1
                all_detections.extend(detections)

                current_fps = 1.0 / max(t1 - t0, 1e-6)
                annotated = self.draw(frame, detections, show_fps=current_fps)

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow("YOLO Object Detection", annotated)
                    key = cv2.waitKey(1)
                    if key in (ord("q"), 27):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        avg_fps = frame_count / max(total_time, 1e-6)
        log.info(f"Processed {frame_count} frames, avg FPS: {avg_fps:.1f}")

        return all_detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    det = YOLODetector(
        weights=args.weights, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, device=args.device,
    )

    src = args.source
    try:
        src = int(args.source)
    except ValueError:
        pass

    if isinstance(src, int):
        det.detect_video(src, save_path=args.save, show=not args.no_show)
    elif Path(src).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        results = det.detect_image(src, save_path=args.save)
        for d in results:
            print(f"  {d.class_name:15s} {d.confidence:.3f}  [{d.x1},{d.y1},{d.x2},{d.y2}]")
    else:
        det.detect_video(src, save_path=args.save, show=not args.no_show)
