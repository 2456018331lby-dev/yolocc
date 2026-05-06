# YOLOv8 容器杂物检测工程

这是一个可复用的 YOLOv8 检测工程模板：包含数据准备、训练、验证、导出、推理、API、网页演示、Docker、测试与维护文档。

当前仓库默认模型是基于 Stanford TrashNet 真实数据训练的 4 类垃圾检测器，但工程结构本身可复用到新的检测任务。

维护入口：`MAINTENANCE.md`
模型详情：`weights/model_card.md`

## 快速开始

```bash
python -m pip install -e ".[all]"
python -m pytest tests -q
python -m src.cli --help
```

## 当前能力

- `.pt` / `.onnx` 推理
- `.onnx` 自动优先 ONNX Runtime，失败回退 OpenCV DNN
- CLI / API / Streamlit / Docker
- 全量测试通过

## 推荐阅读顺序

1. `MAINTENANCE.md`
2. `DATA_GUIDE.md`
3. `weights/model_card.md`
4. `README.md`（本文件）

## Project Structure

```text
yolocc/
├── src/                        # Core source code
│   ├── __init__.py             # Package init
│   ├── __main__.py             # Module runner
│   ├── cli.py                  # Click CLI entry point
│   ├── config.py               # Pydantic configuration
│   ├── logger.py               # Rich logging
│   ├── train.py                # Training pipeline
│   ├── validate.py             # Model evaluation
│   ├── export_model.py         # ONNX/INT8/TorchScript export
│   ├── inference.py            # Unified inference engine
│   ├── benchmark.py            # Performance benchmarking
│   ├── api.py                  # FastAPI REST server
│   └── visualize.py            # Matplotlib plotting
├── configs/                    # Configuration files
│   ├── garbage.yaml            # Current 4-class example dataset config
│   ├── template_object.yaml    # New 6-class generic template config
│   └── train_cfg.yaml          # Training hyperparameters
├── deploy/                     # Deployment files
│   ├── app.py                  # Streamlit web demo
│   └── opencv_deploy.py        # OpenCV DNN lightweight deploy
├── scripts/                    # Utility scripts
│   ├── prepare_dataset.py      # Dataset generation/split/validation
│   └── run_pipeline.py         # Full pipeline runner
├── tests/                      # Unit tests
│   ├── test_config.py          # Config validation tests
│   ├── test_inference.py       # Inference pipeline tests
│   └── test_export.py          # Export/config tests
├── docker/                     # Docker support
│   ├── Dockerfile              # Multi-stage build
│   └── docker-compose.yml      # API + Demo + Training services
├── pyproject.toml              # Package metadata & tool config
├── requirements.txt            # Runtime dependencies
├── Makefile                    # Common commands
├── MAINTENANCE.md              # Maintainer / AI handoff document
└── weights/                    # Model weights (large files should not be committed)
```

## Quick Start

### Install

```bash
cd /mnt/c/Users/24560/Desktop/study/ccdemo/yolocc

# Full install: runtime + dev + API + UI
python -m pip install -e ".[all]"

# Runtime only
python -m pip install -e .

# Or use requirements.txt
python -m pip install -r requirements.txt
```

### Validate current synthetic dataset

```bash
python scripts/prepare_dataset.py validate --data-dir data/dataset
# or
make data-validate
```

### CLI Usage

```bash
# Training
# Default template path is now the generic object template
yolocc train --cfg configs/train_cfg.yaml --data configs/template_object.yaml --name object_detect_v1

# Fast CPU smoke training
yolocc train --cfg configs/train_cfg.yaml --data configs/template_object.yaml --epochs 1 --batch 2 --device cpu

# Validation
yolocc validate --weights results/runs/train/object_detect_v1/weights/best.pt --data configs/template_object.yaml --device cpu

# Export ONNX
yolocc export --weights results/runs/train/object_detect_v1/weights/best.pt --format onnx --data configs/template_object.yaml

# Detection with exported ONNX
yolocc detect --source data/dataset/images/test/0000.jpg --weights weights/best.onnx --save results/detect_0000.jpg --no-show

# Benchmark
yolocc benchmark --weights results/runs/train/object_detect_v1/weights/best.pt --weights weights/best.onnx

# REST API
yolocc serve --weights weights/best.onnx --port 8000

# Dataset tools
yolocc data generate --num 500
yolocc data validate
```

### Python API

```python
from src.inference import YOLODetector

det = YOLODetector("weights/best.onnx", conf=0.25)
detections = det.detect_image("data/dataset/images/test/0000.jpg", save_path="results/result.jpg")
for d in detections:
    print(f"{d.class_name}: {d.confidence:.3f} {d.bbox}")
```

### Make Commands

```bash
make install          # Install with all deps
make test             # Run tests
make lint             # Lint code
make data             # Generate synthetic dataset
make data-validate    # Validate dataset
make train            # Train model
make export           # Export ONNX
make benchmark        # Run benchmarks
make serve            # Start API server
make demo             # Start Streamlit demo
make pipeline         # Run full pipeline
make docker-build     # Build Docker image
```

## REST API

```bash
# Start server after weights/best.onnx exists.
yolocc serve --weights weights/best.onnx

# Health check
curl http://localhost:8000/health

# Detect objects
curl -X POST http://localhost:8000/detect -F "file=@data/dataset/images/test/0000.jpg"

# Get annotated image
curl -X POST "http://localhost:8000/detect?return_image=true" \
  -F "file=@data/dataset/images/test/0000.jpg" \
  -o results/api_result.jpg

# API docs
# http://localhost:8000/docs
```

## Configuration

All configs use Pydantic validation. Edit `configs/train_cfg.yaml`:

```yaml
model: yolov8n.pt
data: configs/template_object.yaml
epochs: 100
imgsz: 640
batch: 16
device: 0

# Augmentation
mosaic: 1.0
fliplr: 0.5
hsv_h: 0.015
```

Override via CLI:

```bash
yolocc train --epochs 50 --batch 8 --device cpu
```

## Dataset Classes

| ID | Class | Chinese | Synthetic placeholder |
|----|-------|---------|-----------------------|
| 0 | recyclable | 可回收物 | circle |
| 1 | hazardous | 有害垃圾 | triangle |
| 2 | kitchen | 厨余垃圾 | rectangle |
| 3 | other | 其他垃圾 | pentagon |

Current `data/dataset` images are synthetic placeholders. Replace or supplement them with real labeled garbage images before claiming real-world detection performance.

### Using Real Data

A built-in script downloads Stanford TrashNet (2,527 images, 6 classes) and converts it to YOLO detection format with automatic 6→4 class mapping:

```bash
# Download TrashNet and convert to YOLO format
python scripts/download_real_dataset.py --source trashnet --output data/real

# With auto-labeling using pretrained YOLOv8 (better bounding boxes)
python scripts/download_real_dataset.py --source trashnet --output data/real --auto-label

# Validate
python scripts/prepare_dataset.py validate --data-dir data/real

# Train on real data
yolocc train --cfg configs/train_cfg.yaml --data configs/template_object.yaml --epochs 100 --device 0
```

Class mapping (TrashNet → yolocc):
- cardboard, glass, metal, paper, plastic → **recyclable** (0)
- trash → **other** (3)

Note: TrashNet only provides recyclable and other. For all 4 classes (hazardous, kitchen), you need additional data sources or manual annotation.

### Manual Data Preparation

Organize your images and labels in YOLO format:

```
data/dataset/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Each label file has one line per object: `class_id cx cy width height` (normalized 0-1).

| class_id | class | examples |
|----------|-------|----------|
| 0 | recyclable (可回收物) | bottles, cans, cardboard, paper, plastic |
| 1 | hazardous (有害垃圾) | batteries, lightbulbs, medicine, chemicals |
| 2 | kitchen (厨余垃圾) | food waste, fruit peels, leftovers |
| 3 | other (其他垃圾) | cigarettes, ceramics, diapers |

## Deployment Options

| Method | Platform | Language | Status |
|--------|----------|----------|--------|
| OpenCV DNN | PC/Embedded | Python | Implemented; auto-detects ONNX input size, uses letterbox |
| Ultralytics `.pt` | PC/GPU/CPU | Python | Implemented |
| ONNX Runtime | PC/GPU/CPU | Python | Implemented; default `.onnx` backend when available, override via `--backend onnxruntime` |
| NCNN | ARM | C++ | Planned; README no longer claims files exist |
| REST API | Any | HTTP | Implemented |
| Streamlit | Browser | Python | Implemented |

## Performance Reference

| Model | Dataset | mAP50 | mAP50-95 | Inference | Size |
|-------|---------|-------|----------|-----------|------|
| YOLOv8n | TrashNet (real, 30ep) | 0.436 | 0.382 | 2.7ms (RTX 4060) | 6.0 MB .pt / 12 MB .onnx |
| YOLOv8n | synthetic smoke (1ep) | 0.560 | 0.407 | ~8ms (CPU) | 5.9 MB .pt / 12 MB .onnx |

The real model is better for actual garbage detection despite lower mAP, because synthetic metrics are meaningless. TrashNet only has recyclable and other classes directly; hazardous and kitchen come from COCO object heuristics during auto-labeling. For production, manual annotation is recommended.

## Tests

```bash
python -m pytest tests -q
```

Current verified result in this environment:

```text
71 passed
```

Coverage:

- `tests/test_config.py` — Pydantic config validation, YAML loading
- `tests/test_export.py` — YAML/pyproject metadata tests
- `tests/test_inference.py` — Detection dataclass, backend detection, post-process, drawing, preprocessing, dataset generation/validation, CLI detect smoke, API `--imgsz` passthrough, ONNX auto imgsz detection, ONNX Runtime backend
- `tests/test_api.py` — FastAPI TestClient: /health, /info, /detect (json + image + conf override), /detect/batch, error handling
- `tests/test_download_dataset.py` — TrashNet 6→4 class mapping, YOLO label generation, COCO→garbage heuristic mapping, folder-based class extraction

## Resume Description Template

Use only after training on a real/approved dataset and recording measured metrics:

> Built a YOLOv8-based garbage object detection pipeline with dataset validation, training, export, API serving, and web demo. Exported ONNX model for OpenCV DNN deployment and benchmarked latency on target hardware.

Do not claim 92%+ mAP or 30FPS unless those results are reproduced and documented for the exact model/data/hardware.
