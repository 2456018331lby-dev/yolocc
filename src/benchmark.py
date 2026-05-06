"""
Model Benchmarking — Industrial Grade
Features:
  - Multiple backend comparison (PT, ONNX, ONNX Runtime)
  - Percentile latency (P50, P95, P99)
  - Memory footprint tracking
  - JSON export
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.logger import get_logger, log_kv, log_section, log_success, log_table

log = get_logger("benchmark")


def _benchmark_pytorch(weights: str, imgsz: int, n_runs: int, warmup: int, device: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights)

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_arr = np.array(times)
    return {
        "backend": "PyTorch",
        "mean_ms": round(float(np.mean(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "p50_ms": round(float(np.percentile(times_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(times_arr, 99)), 2),
        "fps": round(1000.0 / float(np.mean(times_arr)), 1),
        "model_size_mb": round(os.path.getsize(weights) / (1024 * 1024), 2),
    }


def _benchmark_onnx(weights: str, imgsz: int, n_runs: int, warmup: int, device: str) -> dict:
    net = cv2.dnn.readNetFromONNX(weights)

    if "cuda" in device.lower():
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(dummy, 1 / 255.0, (imgsz, imgsz), swapRB=True)

    for _ in range(warmup):
        net.setInput(blob)
        net.forward()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        net.setInput(blob)
        net.forward()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_arr = np.array(times)
    return {
        "backend": "ONNX (OpenCV DNN)",
        "mean_ms": round(float(np.mean(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "p50_ms": round(float(np.percentile(times_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(times_arr, 99)), 2),
        "fps": round(1000.0 / float(np.mean(times_arr)), 1),
        "model_size_mb": round(os.path.getsize(weights) / (1024 * 1024), 2),
    }


def _benchmark_onnxruntime(weights: str, imgsz: int, n_runs: int, warmup: int) -> dict:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(weights, providers=providers)
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_arr = np.array(times)
    return {
        "backend": "ONNX Runtime",
        "mean_ms": round(float(np.mean(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "p50_ms": round(float(np.percentile(times_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(times_arr, 99)), 2),
        "fps": round(1000.0 / float(np.mean(times_arr)), 1),
        "model_size_mb": round(os.path.getsize(weights) / (1024 * 1024), 2),
    }


def run_benchmark(
    weights_list: List[str],
    imgsz: int = 640,
    n_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
    output: str = "results/benchmark_results.json",
) -> List[dict]:
    """
    Benchmark multiple models and output comparison table.

    Returns:
        List of result dicts.
    """
    log_section("Model Benchmark", log)
    log_kv("Image Size", imgsz, log)
    log_kv("Runs", n_runs, log)
    log_kv("Warmup", warmup, log)
    log_kv("Device", device, log)
    log.info("─" * 60)

    results = []

    for w in weights_list:
        ext = Path(w).suffix.lower()
        log.info(f"\n  Benchmarking: {w}")

        try:
            if ext == ".pt":
                r = _benchmark_pytorch(w, imgsz, n_runs, warmup, device)
                results.append(r)
            elif ext == ".onnx":
                r = _benchmark_onnx(w, imgsz, n_runs, warmup, device)
                results.append(r)
                # Also try ONNX Runtime
                try:
                    r2 = _benchmark_onnxruntime(w, imgsz, n_runs, warmup)
                    results.append(r2)
                except ImportError:
                    log.info("    ONNX Runtime not installed, skipping")
            else:
                log.warning(f"  Skipping unsupported format: {ext}")
        except Exception as e:
            log.error(f"  Error: {e}")

    if results:
        # Print table
        headers = ["Backend", "Mean", "P50", "P95", "P99", "FPS", "Size(MB)"]
        rows = []
        for r in results:
            rows.append([
                r["backend"],
                f"{r['mean_ms']:.2f}ms",
                f"{r['p50_ms']:.2f}ms",
                f"{r['p95_ms']:.2f}ms",
                f"{r['p99_ms']:.2f}ms",
                f"{r['fps']:.1f}",
                f"{r['model_size_mb']:.2f}",
            ])

        log.info("")
        log_table(headers, rows, log)

        # Save JSON
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log.info(f"\n  Results saved to: {out_path}")

    log_success("Benchmark complete.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_benchmark(args.weights, imgsz=args.imgsz, n_runs=args.runs, device=args.device)
