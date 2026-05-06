"""
YOLOv8 Training Pipeline — Industrial Grade
Features:
  - Pydantic config validation
  - Rich logging with file output
  - Metric tracking & checkpoint management
  - Early stopping with patience
  - TensorBoard integration
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from src.config import PROJECT_ROOT, TrainingConfig, load_yaml
from src.logger import get_logger, log_kv, log_section, log_success

log = get_logger("train")


def run_training(
    cfg_path: str = "configs/train_cfg.yaml",
    overrides: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    project: str = "results/runs/train",
    name: str = "object_detect",
) -> str:
    """
    Execute the full training pipeline.

    Args:
        cfg_path: Path to training config YAML.
        overrides: CLI overrides that take precedence over config.
        resume: Whether to resume from last checkpoint.
        project: Output project directory.
        name: Experiment name.

    Returns:
        Path to the training output directory.
    """
    overrides = overrides or {}
    cfg_file = PROJECT_ROOT / cfg_path

    # Load and validate config
    if cfg_file.exists():
        raw = load_yaml(cfg_file)
        # Remove non-training keys
        for k in ("export_format", "export_imgsz", "int8_quantize"):
            raw.pop(k, None)
        # Merge overrides
        raw.update(overrides)
        cfg = TrainingConfig(**raw)
    else:
        cfg = TrainingConfig(**overrides)

    # Resolve data path
    data_path = PROJECT_ROOT / cfg.data
    if not data_path.exists():
        log.warning(f"Dataset config not found: {data_path}")
    else:
        cfg.data = str(data_path)

    # Resolve pretrained weights
    weights = cfg.pretrained or cfg.model

    log_section("YOLOv8 Training Pipeline", log)
    log_kv("Model", weights, log)
    log_kv("Dataset", cfg.data, log)
    log_kv("Epochs", cfg.epochs, log)
    log_kv("Image Size", cfg.imgsz, log)
    log_kv("Batch Size", cfg.batch, log)
    log_kv("Device", cfg.device, log)
    log_kv("Optimizer", cfg.optimizer, log)
    log_kv("Learning Rate", cfg.lr0, log)
    log_kv("Patience", cfg.patience, log)
    log_kv("Resume", resume, log)
    log.info("─" * 60)

    # Initialize model
    model = YOLO(weights)
    log.info(f"Loaded model: {weights}")

    # Train
    t_start = time.time()
    results = model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        patience=cfg.patience,
        optimizer=cfg.optimizer.value if hasattr(cfg.optimizer, "value") else cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        close_mosaic=cfg.close_mosaic,
        hsv_h=cfg.augmentation.hsv_h,
        hsv_s=cfg.augmentation.hsv_s,
        hsv_v=cfg.augmentation.hsv_v,
        degrees=cfg.augmentation.degrees,
        translate=cfg.augmentation.translate,
        scale=cfg.augmentation.scale,
        shear=cfg.augmentation.shear,
        perspective=cfg.augmentation.perspective,
        flipud=cfg.augmentation.flipud,
        fliplr=cfg.augmentation.fliplr,
        mosaic=cfg.augmentation.mosaic,
        mixup=cfg.augmentation.mixup,
        copy_paste=cfg.augmentation.copy_paste,
        erasing=cfg.augmentation.erasing,
        project=str(PROJECT_ROOT / project),
        name=name,
        exist_ok=cfg.exist_ok,
        resume=resume,
        verbose=True,
        plots=True,
    )
    elapsed = time.time() - t_start

    # Summary
    save_dir = getattr(results, "save_dir", PROJECT_ROOT / project / name)
    log_section("Training Complete", log)
    log_kv("Duration", f"{elapsed:.0f}s ({elapsed/3600:.1f}h)", log)
    log_kv("Output", str(save_dir), log)
    log_kv("Best Weights", f"{save_dir}/weights/best.pt", log)
    log_kv("Last Weights", f"{save_dir}/weights/last.pt", log)

    if hasattr(results, "results_dict"):
        rd = results.results_dict
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in rd:
                log_kv(key.split("/")[-1], f"{rd[key]:.4f}", log)

    log_success("Training pipeline finished successfully.")
    return str(save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/train_cfg.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project", default="results/runs/train")
    parser.add_argument("--name", default="object_detect")
    args = parser.parse_args()

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.data:
        overrides["data"] = args.data
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch"] = args.batch
    if args.imgsz:
        overrides["imgsz"] = args.imgsz
    if args.device is not None:
        overrides["device"] = args.device

    run_training(
        cfg_path=args.cfg,
        overrides=overrides,
        resume=args.resume,
        project=args.project,
        name=args.name,
    )
