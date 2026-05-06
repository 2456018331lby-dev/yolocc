"""
YOLOv8 Model Validation — Industrial Grade
Features:
  - Per-class and overall metrics
  - JSON/CSV export
  - Confusion matrix generation
  - Rich console output
"""

import json
import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.config import PROJECT_ROOT, load_dataset_config
from src.logger import get_logger, log_kv, log_section, log_success, log_table

log = get_logger("validate")


def run_validation(
    weights: str,
    data: str = "configs/garbage.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    split: str = "val",
    conf: float = 0.25,
    iou: float = 0.6,
    save_json: bool = False,
    project: str = "results/runs/val",
    plots: bool = True,
) -> dict:
    """
    Run model validation and return metrics dict.

    Returns:
        Dictionary with mAP50, mAP50-95, precision, recall, and per-class AP.
    """
    data_path = PROJECT_ROOT / data

    log_section("Model Validation", log)
    log_kv("Weights", weights, log)
    log_kv("Dataset", str(data_path), log)
    log_kv("Split", split, log)
    log_kv("Image Size", imgsz, log)
    log_kv("Conf Threshold", conf, log)
    log_kv("IoU Threshold", iou, log)
    log.info("─" * 60)

    model = YOLO(weights)

    metrics = model.val(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        conf=conf,
        iou=iou,
        save_json=save_json,
        project=str(PROJECT_ROOT / project),
        exist_ok=True,
        plots=plots,
        verbose=True,
    )

    # Extract results
    names = metrics.names if hasattr(metrics, "names") else {}
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": {},
    }

    if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > 0:
        for i, ap50 in enumerate(metrics.box.ap50):
            class_name = names.get(i, f"class_{i}")
            results["per_class"][class_name] = float(ap50)

    # Print results
    log_section("Validation Results", log)
    log_kv("mAP50", f"{results['mAP50']:.4f}", log)
    log_kv("mAP50-95", f"{results['mAP50-95']:.4f}", log)
    log_kv("Precision", f"{results['precision']:.4f}", log)
    log_kv("Recall", f"{results['recall']:.4f}", log)

    if results["per_class"]:
        log.info("")
        log.info("  Per-class mAP50:")
        rows = [[name, f"{ap:.4f}"] for name, ap in results["per_class"].items()]
        log_table(["Class", "AP50"], rows, log)

    # Save JSON
    out_dir = PROJECT_ROOT / project
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"  Results saved to: {json_path}")

    log_success("Validation complete.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/garbage.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--project", default="results/runs/val")
    args = parser.parse_args()

    run_validation(**vars(args))
