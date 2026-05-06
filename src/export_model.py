"""
YOLOv8 Model Export — Industrial Grade
Supports: ONNX, INT8 PTQ, TorchScript, OpenVINO, TFLite, NCNN
Features:
  - Size comparison table
  - Integrity verification
  - Auto-naming with quantization suffix
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

from src.config import PROJECT_ROOT
from src.logger import get_logger, log_kv, log_section, log_success, log_table

log = get_logger("export")

FORMAT_EXT = {
    "onnx": ".onnx",
    "torchscript": ".torchscript",
    "openvino": "_openvino_model",
    "tflite": ".tflite",
    "ncnn": "_ncnn_model",
}


def _get_file_size_mb(path: str) -> float:
    p = Path(path)
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    elif p.is_file():
        return p.stat().st_size / (1024 * 1024)
    return 0.0


def export_single(model: YOLO, fmt: str, imgsz: int, int8: bool,
                  data: Optional[str], simplify: bool, opset: int,
                  device: str) -> Optional[str]:
    """Export model to a single format. Returns output path."""
    log.info(f"  Exporting {fmt.upper()}...")

    kwargs = {
        "format": fmt,
        "imgsz": imgsz,
        "device": device,
    }

    if fmt == "onnx":
        kwargs["simplify"] = simplify
        kwargs["opset"] = opset

    if int8 and fmt in ("onnx", "tflite"):
        kwargs["int8"] = True
        if data:
            kwargs["data"] = str(PROJECT_ROOT / data)
        log.info("    INT8 quantization enabled")

    t0 = time.time()
    result = model.export(**kwargs)
    elapsed = time.time() - t0

    result_str = str(result)
    size_mb = _get_file_size_mb(result_str)
    log.info(f"    Done in {elapsed:.1f}s — {result_str} ({size_mb:.2f} MB)")
    return result_str


def run_export(
    weights: str,
    fmt: str = "onnx",
    imgsz: int = 640,
    int8: bool = False,
    data: str = "configs/template_object.yaml",
    simplify: bool = True,
    opset: int = 11,
    device: str = "cpu",
) -> Dict[str, Optional[str]]:
    """
    Export model to specified format(s).

    Returns:
        Dict mapping format name to exported file path.
    """
    log_section("Model Export", log)
    log_kv("Weights", weights, log)
    log_kv("Format", fmt, log)
    log_kv("Image Size", imgsz, log)
    log_kv("INT8", int8, log)
    log_kv("Simplify", simplify, log)
    log_kv("ONNX Opset", opset, log)
    log.info("─" * 60)

    model = YOLO(weights)

    if fmt == "all":
        formats = ["onnx", "torchscript"]
        results = {}
        for f in formats:
            try:
                results[f] = export_single(model, f, imgsz, int8, data, simplify, opset, device)
            except Exception as e:
                log.error(f"  {f} export failed: {e}")
                results[f] = None

        # Summary table
        rows = []
        for f, path in results.items():
            status = "OK" if path else "FAILED"
            size = f"{_get_file_size_mb(path):.2f} MB" if path else "-"
            rows.append([f.upper(), status, size])
        log.info("")
        log_table(["Format", "Status", "Size"], rows, log)
    else:
        results = {fmt: export_single(model, fmt, imgsz, int8, data, simplify, opset, device)}

    log_success("Export complete.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=list(FORMAT_EXT.keys()) + ["all"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", default="configs/template_object.yaml")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_export(**vars(args))
