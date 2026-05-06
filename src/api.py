"""
REST API for YOLO inference — Industrial Grade
Features:
  - FastAPI with auto-docs (/docs)
  - Image upload + URL inference
  - Health check endpoint
  - Structured JSON responses
  - CORS support
"""

import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from src.logger import get_logger

log = get_logger("api")


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class ImageSizeResponse(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    detections: List[DetectionResponse]
    count: int
    inference_ms: float
    image_size: ImageSizeResponse


class BatchDetectItemResponse(BaseModel):
    filename: Optional[str] = None
    detections: List[DetectionResponse] = []
    count: Optional[int] = None
    inference_ms: Optional[float] = None
    error: Optional[str] = None


class BatchDetectResponse(BaseModel):
    results: List[BatchDetectItemResponse]
    total: int


def create_app(
    weights_path: str = "weights/best.onnx",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    max_upload_mb: int = 10,
    cors_origins: Optional[List[str]] = None,
    backend: Optional[str] = None,
):
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    from src.inference import YOLODetector

    app = FastAPI(
        title="YOLO Garbage Detection API",
        version="2.0.0",
        description="Real-time garbage classification using YOLOv8",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load detector
    detector = YOLODetector(
        weights=weights_path, imgsz=imgsz, conf=conf, iou=iou, backend=backend,
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": Path(weights_path).name,
            "backend": detector.backend,
        }

    @app.get("/info")
    async def info():
        """Model information."""
        return {
            "model": Path(weights_path).name,
            "backend": detector.backend,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "classes": detector.class_names,
        }

    @app.post("/detect", response_model=DetectResponse)
    async def detect(
        file: UploadFile = File(...),
        return_image: bool = Query(False, description="Return annotated image"),
        conf_threshold: Optional[float] = Query(None, description="Override confidence threshold"),
    ):
        """
        Detect objects in uploaded image.

        Returns JSON with detections. Set return_image=true to get annotated image.
        """
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        if len(contents) > max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large (max {max_upload_mb}MB)")

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run inference with per-request threshold without mutating shared detector state.
        t0 = time.time()
        detections = detector.detect_frame(frame, conf=conf_threshold)
        elapsed = time.time() - t0

        # Build response
        det_dicts = [d.to_dict() for d in detections]

        if return_image:
            annotated = detector.draw(frame.copy(), detections)
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg",
                headers={
                    "X-Detections": str(len(detections)),
                    "X-Inference-Time-Ms": f"{elapsed*1000:.1f}",
                },
            )

        return {
            "detections": det_dicts,
            "count": len(det_dicts),
            "inference_ms": round(elapsed * 1000, 1),
            "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        }

    @app.post("/detect/batch", response_model=BatchDetectResponse)
    async def detect_batch(
        files: List[UploadFile] = File(...),
    ):
        """Batch detect objects in multiple images."""
        results = []
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                results.append({"filename": file.filename, "error": "Not an image"})
                continue

            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                results.append({"filename": file.filename, "error": "Invalid image"})
                continue

            t0 = time.time()
            detections = detector.detect_frame(frame)
            elapsed = time.time() - t0

            results.append({
                "filename": file.filename,
                "detections": [d.to_dict() for d in detections],
                "count": len(detections),
                "inference_ms": round(elapsed * 1000, 1),
            })

        return {"results": results, "total": len(results)}

    return app


# Standalone runner
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best.onnx")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Install uvicorn: pip install uvicorn[standard]")
        exit(1)

    app = create_app(weights_path=args.weights, imgsz=args.imgsz)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
