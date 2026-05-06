"""
Centralized configuration management with Pydantic validation.
Supports YAML files, environment variables, and CLI overrides.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelVariant(str, Enum):
    YOLOV8N = "yolov8n.pt"
    YOLOV8S = "yolov8s.pt"
    YOLOV8M = "yolov8m.pt"
    YOLOV8L = "yolov8l.pt"
    YOLOV8X = "yolov8x.pt"


class ExportFormat(str, Enum):
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    OPENVINO = "openvino"
    TFLITE = "tflite"
    NCNN = "ncnn"


class OptimizerChoice(str, Enum):
    SGD = "SGD"
    ADAM = "Adam"
    ADAMW = "AdamW"
    AUTO = "auto"


# ── Sub-configs ──────────────────────────────────────────────────────────────

class AugmentationConfig(BaseModel):
    """Data augmentation hyperparameters."""
    hsv_h: float = Field(0.015, ge=0, le=1, description="Hue augmentation range")
    hsv_s: float = Field(0.7, ge=0, le=1, description="Saturation augmentation range")
    hsv_v: float = Field(0.4, ge=0, le=1, description="Value augmentation range")
    degrees: float = Field(0.0, ge=0, le=180, description="Rotation degrees (+/-)")
    translate: float = Field(0.1, ge=0, le=1, description="Translation fraction")
    scale: float = Field(0.5, ge=0, description="Scale gain (+/-)")
    shear: float = Field(0.0, ge=0, le=180, description="Shear degrees (+/-)")
    perspective: float = Field(0.0, ge=0, le=0.001, description="Perspective transform")
    flipud: float = Field(0.0, ge=0, le=1, description="Vertical flip probability")
    fliplr: float = Field(0.5, ge=0, le=1, description="Horizontal flip probability")
    mosaic: float = Field(1.0, ge=0, le=1, description="Mosaic augmentation probability")
    mixup: float = Field(0.0, ge=0, le=1, description="MixUp augmentation probability")
    copy_paste: float = Field(0.0, ge=0, le=1, description="Segment copy-paste probability")
    erasing: float = Field(0.4, ge=0, le=1, description="Random erasing probability")


class TrainingConfig(BaseModel):
    """Training configuration."""
    model: str = Field("yolov8n.pt", description="Base model path or name")
    data: str = Field("configs/garbage.yaml", description="Dataset config path")
    epochs: int = Field(100, ge=1, le=10000, description="Total training epochs")
    imgsz: int = Field(640, ge=32, le=2048, description="Input image size")
    batch: int = Field(16, ge=1, le=512, description="Batch size")
    device: str = Field("0", description="Device: 0, 1, cpu, mps")

    @field_validator("device", mode="before")
    @classmethod
    def coerce_device(cls, v):
        return str(v)
    workers: int = Field(8, ge=0, le=32, description="DataLoader workers")
    patience: int = Field(20, ge=0, description="Early stopping patience")
    optimizer: OptimizerChoice = Field(OptimizerChoice.AUTO, description="Optimizer")
    lr0: float = Field(0.01, gt=0, le=1, description="Initial learning rate")
    lrf: float = Field(0.01, gt=0, le=1, description="Final LR factor")
    momentum: float = Field(0.937, ge=0, le=1, description="SGD momentum")
    weight_decay: float = Field(0.0005, ge=0, le=0.1, description="Weight decay")
    warmup_epochs: float = Field(3.0, ge=0, description="Warmup epochs")
    warmup_momentum: float = Field(0.8, ge=0, le=1, description="Warmup momentum")
    warmup_bias_lr: float = Field(0.1, ge=0, le=1, description="Warmup bias LR")
    close_mosaic: int = Field(10, ge=0, description="Disable mosaic last N epochs")
    resume: bool = Field(False, description="Resume from last checkpoint")
    project: str = Field("results/runs/train", description="Project output dir")
    name: str = Field("garbage_detect", description="Experiment name")
    pretrained: Optional[str] = Field(None, description="Pretrained weights path")
    exist_ok: bool = Field(True, description="Overwrite existing experiment")
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @field_validator("data")
    @classmethod
    def resolve_data_path(cls, v: str) -> str:
        p = Path(v)
        if p.exists():
            return str(p.resolve())
        return v


class ExportConfig(BaseModel):
    """Model export configuration."""
    weights: str = Field(..., description="Model weights path")
    format: ExportFormat = Field(ExportFormat.ONNX, description="Export format")
    imgsz: int = Field(640, ge=32, le=2048, description="Export image size")
    int8: bool = Field(False, description="INT8 quantization")
    data: str = Field("configs/garbage.yaml", description="Dataset for INT8 calibration")
    simplify: bool = Field(True, description="Simplify ONNX graph")
    opset: int = Field(11, ge=9, le=20, description="ONNX opset version")
    device: str = Field("cpu", description="Export device")
    output_dir: str = Field("weights/export", description="Output directory")


class InferenceConfig(BaseModel):
    """Inference configuration."""
    source: str = Field("0", description="Input: image/video path or webcam index")
    weights: str = Field(..., description="Model weights path")
    imgsz: int = Field(640, ge=32, le=2048, description="Input image size")
    conf: float = Field(0.25, ge=0, le=1, description="Confidence threshold")
    iou: float = Field(0.45, ge=0, le=1, description="NMS IoU threshold")
    device: str = Field("cpu", description="Device")
    save: Optional[str] = Field(None, description="Save output path")
    show: bool = Field(True, description="Display window")
    classes: Optional[List[int]] = Field(None, description="Filter by class IDs")
    max_det: int = Field(300, ge=1, le=1000, description="Max detections per image")


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""
    weights: List[str] = Field(..., description="Model weight files to benchmark")
    imgsz: int = Field(640, ge=32, le=2048, description="Image size")
    runs: int = Field(100, ge=10, le=10000, description="Number of inference runs")
    warmup: int = Field(10, ge=0, le=100, description="Warmup runs")
    device: str = Field("cpu", description="Device")
    output: str = Field("results/benchmark_results.json", description="Results output path")


class ServerConfig(BaseModel):
    """API server configuration."""
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1024, le=65535, description="Server port")
    workers: int = Field(1, ge=1, le=8, description="Server workers")
    weights: str = Field("weights/best.onnx", description="Model weights path")
    imgsz: int = Field(640, description="Input image size")
    conf: float = Field(0.25, description="Confidence threshold")
    iou: float = Field(0.45, description="NMS IoU threshold")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    max_upload_mb: int = Field(10, ge=1, le=100, description="Max upload size MB")


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    path: str = Field("./data/dataset", description="Dataset root path")
    train: str = Field("images/train", description="Train images subpath")
    val: str = Field("images/val", description="Val images subpath")
    test: str = Field("images/test", description="Test images subpath")
    names: Dict[int, str] = Field(
        {0: "recyclable", 1: "hazardous", 2: "kitchen", 3: "other"},
        description="Class ID to name mapping"
    )
    colors: Dict[int, List[int]] = Field(
        {0: [0, 255, 0], 1: [0, 0, 255], 2: [255, 0, 0], 3: [128, 128, 128]},
        description="Class ID to BGR color mapping"
    )
    nc: Optional[int] = Field(None, description="Number of classes (auto from names)")


# ── Root config ──────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Application root configuration."""
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig, validate_default=False)
    inference: InferenceConfig = Field(default_factory=InferenceConfig, validate_default=False)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig, validate_default=False)
    server: ServerConfig = Field(default_factory=ServerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


# ── Loaders ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent


def load_yaml(path: Union[str, Path], encoding: str = "utf-8") -> dict:
    """Load YAML file with encoding fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding=encoding) as f:
        return yaml.safe_load(f) or {}


def load_dataset_config(path: Union[str, Path] = "configs/garbage.yaml") -> DatasetConfig:
    """Load and validate dataset config from YAML."""
    raw = load_yaml(PROJECT_ROOT / path)
    return DatasetConfig(**raw)


def load_training_config(path: Union[str, Path] = "configs/train_cfg.yaml") -> TrainingConfig:
    """Load and validate training config from YAML."""
    raw = load_yaml(PROJECT_ROOT / path)
    # Separate augmentation keys
    aug_keys = set(AugmentationConfig.model_fields.keys())
    aug_raw = {k: raw.pop(k) for k in list(raw.keys()) if k in aug_keys}
    if aug_raw:
        raw["augmentation"] = aug_raw
    return TrainingConfig(**raw)


def get_class_names(data_config: Union[str, Path] = "configs/garbage.yaml") -> Dict[int, str]:
    """Get class names dict from dataset config."""
    cfg = load_dataset_config(data_config)
    return cfg.names


def get_class_colors(data_config: Union[str, Path] = "configs/garbage.yaml") -> Dict[int, tuple]:
    """Get class colors dict from dataset config (as BGR tuples)."""
    cfg = load_dataset_config(data_config)
    return {k: tuple(v) for k, v in cfg.colors.items()}
