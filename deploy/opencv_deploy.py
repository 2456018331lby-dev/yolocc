"""
OpenCV DNN Deployment - Lightweight edge inference
Supports: Webcam, Image, Video file
Usage:
    python deploy/opencv_deploy.py --weights weights/best.onnx --source 0
    python deploy/opencv_deploy.py --weights weights/best.onnx --source image.jpg --save result.jpg
    python deploy/opencv_deploy.py --weights weights/best.onnx --source video.mp4 --save output.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# Garbage classification config
CLASS_NAMES = {0: "recyclable", 1: "hazardous", 2: "kitchen", 3: "other"}
CLASS_COLORS = {
    0: (0, 255, 0),      # green
    1: (0, 0, 255),      # red
    2: (255, 0, 0),      # blue
    3: (128, 128, 128),  # gray
}
CLASS_CN = {
    0: "Ke Hui Shou",
    1: "You Hai",
    2: "Chu Yu",
    3: "Qi Ta",
}


class OpenCVDeploy:
    """Lightweight OpenCV DNN deployment for YOLOv8."""

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.net = cv2.dnn.readNetFromONNX(weights)

        # Try CUDA
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.backend = "CUDA"
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = "CPU"

        print(f"[INFO] Model loaded: {weights} (backend: {self.backend})")

    def preprocess(self, frame: np.ndarray) -> tuple:
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.imgsz, self.imgsz),
            swapRB=True, crop=False
        )
        return blob, frame.shape[:2]

    def postprocess(self, output: np.ndarray, orig_shape: tuple) -> list:
        h, w = orig_shape
        outputs = output[0].T  # (8400, 84)

        boxes, confidences, class_ids = [], [], []

        for det in outputs:
            cx, cy, bw, bh = det[:4]
            scores = det[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])

            if conf > self.conf:
                x1 = max(0, int((cx - bw / 2) * w / self.imgsz))
                y1 = max(0, int((cy - bh / 2) * h / self.imgsz))
                x2 = min(w - 1, int((cx + bw / 2) * w / self.imgsz))
                y2 = min(h - 1, int((cy + bh / 2) * h / self.imgsz))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(conf)
                class_ids.append(cls_id)

        results = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.iou)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw, bh = boxes[i]
                    results.append({
                        "cls": class_ids[i],
                        "name": CLASS_NAMES.get(class_ids[i], "?"),
                        "conf": confidences[i],
                        "box": (x, y, x + bw, y + bh),
                    })
        return results

    def detect(self, frame: np.ndarray) -> list:
        blob, orig_shape = self.preprocess(frame)
        self.net.setInput(blob)
        output = self.net.forward()
        return self.postprocess(output, orig_shape)

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            cls_id = det["cls"]
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            x1, y1, x2, y2 = det["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det['name']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def run(self, source, save_path: str = None, show: bool = True):
        """Run detection on video/webcam."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_count = 0
        fps_list = []

        print(f"[INFO] Processing: {source} ({w}x{h} @ {fps}fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            dets = self.detect(frame)
            t1 = time.time()

            frame = self.draw(frame, dets)
            current_fps = 1.0 / max(t1 - t0, 1e-6)
            fps_list.append(current_fps)

            # FPS overlay
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Detection count overlay
            cv2.putText(frame, f"Objects: {len(dets)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("YOLO Garbage Detection", frame)
                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    break

            frame_count += 1

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = sum(fps_list) / max(len(fps_list), 1)
        print(f"\n[DONE] {frame_count} frames, avg FPS: {avg_fps:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV DNN Deployment")
    parser.add_argument("--weights", type=str, required=True, help="ONNX weights path")
    parser.add_argument("--source", type=str, default="0", help="Input source")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--save", type=str, default=None, help="Save output path")
    parser.add_argument("--no-show", action="store_true", help="Don't show window")
    return parser.parse_args()


def main():
    args = parse_args()
    deploy = OpenCVDeploy(args.weights, args.imgsz, args.conf, args.iou)

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    if isinstance(source, int):
        deploy.run(source, save_path=args.save, show=not args.no_show)
    elif Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {source}")
        dets = deploy.detect(frame)
        frame = deploy.draw(frame, dets)

        if args.save:
            cv2.imwrite(args.save, frame)
            print(f"[INFO] Saved: {args.save}")

        for d in dets:
            print(f"  {d['name']:15s} {d['conf']:.3f}  {d['box']}")

        if not args.no_show:
            cv2.imshow("Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        deploy.run(source, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
