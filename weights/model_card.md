# Model Card — Real Garbage Detection Model

## Overview

YOLOv8n garbage detection model trained on real-world trash images (Stanford TrashNet + auto-labeling).

## Training Details

- **Date**: 2026-05-03
- **Base model**: yolov8n.pt (Ultralytics pretrained COCO)
- **Dataset**: Stanford TrashNet (2,527 images) → auto-labeled with YOLOv8n pretrained
- **Split**: train 1766 / val 377 / test 384
- **Image size**: 640
- **Batch**: 16
- **Epochs**: 30
- **Patience**: 15 (early stopping enabled)
- **Device**: NVIDIA RTX 4060 Laptop GPU
- **Training time**: ~11 minutes
- **CLI command**:
  ```bash
  yolo detect train model=yolov8n.pt data=configs/garbage.yaml epochs=30 batch=16 imgsz=640 device=0 patience=15
  ```

## Test Set Metrics

### YOLOv8n (current production default)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.488 | 0.465 | 0.436 | 0.382 |
| recyclable | 0.652 | 0.864 | 0.831 | 0.766 |
| hazardous | 0.691 | 0.222 | 0.345 | 0.290 |
| kitchen | 0.348 | 0.261 | 0.190 | 0.141 |
| other | 0.263 | 0.512 | 0.377 | 0.330 |

**Speed**: 2.7ms inference per image (RTX 4060)

### YOLOv8s (comparison run)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.493 | 0.474 | 0.417 | 0.364 |
| recyclable | 0.609 | 0.878 | 0.813 | 0.748 |
| hazardous | 0.781 | 0.222 | 0.371 | 0.314 |
| kitchen | 0.327 | 0.191 | 0.160 | 0.127 |
| other | 0.256 | 0.605 | 0.324 | 0.268 |

**Speed**: 4.2ms inference per image (RTX 4060)

### Conclusion

`yolov8s` did **not** outperform `yolov8n` on this dataset. Overall mAP50 dropped from 0.436 → 0.417 and inference got slower (2.7ms → 4.2ms).

This strongly indicates the current bottleneck is **data quality and class imbalance**, not model capacity. Therefore:
- `weights/best.pt` / `weights/best.onnx` should remain the **YOLOv8n** model
- Future improvement should focus on manual annotation cleanup and more hazardous/kitchen data
- Bigger models are not justified yet on this auto-labeled dataset

## Artifacts

| File | Size | Notes |
|------|------|-------|
| `weights/real_best.pt` | 6.0 MB | PyTorch weights |
| `weights/real_best.onnx` | 12 MB | ONNX export (640px) |
| `weights/best.pt` | 6.0 MB | Copy of real_best.pt (production default) |
| `weights/best.onnx` | 12 MB | Copy of real_best.onnx (production default) |

## Class Distribution (all splits)

| Class | Boxes |
|-------|-------|
| recyclable | 2,252 |
| hazardous | 53 |
| kitchen | 109 |
| other | 275 |

## Limitations

- **Auto-labeled**: bounding boxes come from pretrained YOLOv8n, not human annotation. Box quality is approximate.
- **Class imbalance**: recyclable dominates (84% of boxes). hazardous and kitchen are underrepresented.
- **TrashNet only**: recyclable = {cardboard, glass, metal, paper, plastic}, trash → other. hazardous and kitchen are from auto-label heuristics (COCO classes like scissors→hazardous, banana→kitchen).
- **Not production-ready**: mAP50 0.44 is reasonable for auto-labeled data but not sufficient for deployment without manual annotation cleanup.

## How to Improve

1. **Manual annotation**: Fix auto-label boxes with LabelImg or CVAT
2. **More data**: Add hazardous/kitchen images (batteries, food waste, etc.)
3. **More epochs**: 100+ epochs with larger batch
4. **Larger model**: YOLOv8s or YOLOv8m
5. **INT8 quantization**: For edge deployment
