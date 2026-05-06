"""
CLI entry point for the yolocc toolkit.
Usage:
    yolocc train --cfg configs/train_cfg.yaml
    yolocc validate --weights best.pt
    yolocc export --weights best.pt --format onnx --int8
    yolocc detect --source 0 --weights best.onnx
    yolocc benchmark --weights best.pt best.onnx
    yolocc serve --weights best.onnx --port 8000
    yolocc data generate --num 500
    yolocc data validate
"""

import sys
from pathlib import Path

import click
from rich.console import Console

console = Console(stderr=True)


@click.group()
@click.version_option(version="2.0.0", prog_name="yolocc")
def main():
    """YOLO object detection toolkit."""
    pass


# ── Train ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--cfg", default="configs/train_cfg.yaml", help="Training config YAML")
@click.option("--model", default=None, help="Override base model")
@click.option("--data", default=None, help="Override dataset config")
@click.option("--epochs", default=None, type=int, help="Override epochs")
@click.option("--batch", default=None, type=int, help="Override batch size")
@click.option("--imgsz", default=None, type=int, help="Override image size")
@click.option("--device", default=None, help="Override device")
@click.option("--resume", is_flag=True, help="Resume training")
@click.option("--project", default="results/runs/train", help="Output project dir")
@click.option("--name", default="object_detect", help="Experiment name")
def train(cfg, model, data, epochs, batch, imgsz, device, resume, project, name):
    """Train YOLOv8 model for object detection."""
    from src.train import run_training
    from src.logger import add_file_handler, get_logger

    log = get_logger("train")
    add_file_handler(Path(project) / name / "train.log")

    overrides = {}
    if model:
        overrides["model"] = model
    if data:
        overrides["data"] = data
    if epochs:
        overrides["epochs"] = epochs
    if batch:
        overrides["batch"] = batch
    if imgsz:
        overrides["imgsz"] = imgsz
    if device is not None:
        overrides["device"] = device

    try:
        run_training(
            cfg_path=cfg,
            overrides=overrides,
            resume=resume,
            project=project,
            name=name,
        )
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


# ── Validate ─────────────────────────────────────────────────────────────────

@main.command()
@click.option("--weights", required=True, help="Model weights path")
@click.option("--data", default="configs/template_object.yaml", help="Dataset config")
@click.option("--imgsz", default=640, type=int, help="Image size")
@click.option("--batch", default=16, type=int, help="Batch size")
@click.option("--device", default="0", help="Device")
@click.option("--split", default="val", help="Dataset split")
@click.option("--conf", default=0.25, type=float, help="Confidence threshold")
@click.option("--iou", default=0.6, type=float, help="NMS IoU threshold")
@click.option("--save-json", is_flag=True, help="Save COCO-format JSON")
@click.option("--project", default="results/runs/val", help="Output dir")
def validate(weights, data, imgsz, batch, device, split, conf, iou, save_json, project):
    """Validate YOLOv8 model on dataset."""
    from src.validate import run_validation

    try:
        run_validation(
            weights=weights, data=data, imgsz=imgsz, batch=batch,
            device=device, split=split, conf=conf, iou=iou,
            save_json=save_json, project=project,
        )
    except Exception as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        sys.exit(1)


# ── Export ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--weights", required=True, help="Model weights path")
@click.option("--format", "fmt", default="onnx",
              type=click.Choice(["onnx", "torchscript", "openvino", "tflite", "ncnn", "all"]),
              help="Export format")
@click.option("--imgsz", default=640, type=int, help="Image size")
@click.option("--int8", "int8_flag", is_flag=True, help="INT8 quantization")
@click.option("--data", default="configs/template_object.yaml", help="Dataset for INT8 calibration")
@click.option("--simplify", is_flag=True, default=True, help="Simplify ONNX")
@click.option("--opset", default=11, type=int, help="ONNX opset version")
def export(weights, fmt, imgsz, int8_flag, data, simplify, opset):
    """Export model to deployment format."""
    from src.export_model import run_export

    try:
        run_export(
            weights=weights, fmt=fmt, imgsz=imgsz,
            int8=int8_flag, data=data, simplify=simplify, opset=opset,
        )
    except Exception as e:
        console.print(f"[error]Export failed: {e}[/error]")
        sys.exit(1)


# ── Detect ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--source", default="0", help="Input: image/video path or webcam index")
@click.option("--weights", required=True, help="Model weights path")
@click.option("--imgsz", default=640, type=int, help="Image size")
@click.option("--conf", default=0.25, type=float, help="Confidence threshold")
@click.option("--iou", default=0.45, type=float, help="NMS IoU threshold")
@click.option("--device", default="cpu", help="Device")
@click.option("--backend", default=None, type=click.Choice(["opencv", "onnxruntime", "ultralytics"]), help="Force inference backend")
@click.option("--save", default=None, help="Save output path")
@click.option("--no-show", is_flag=True, help="Don't display window")
def detect(source, weights, imgsz, conf, iou, device, backend, save, no_show):
    """Run inference on image/video/webcam."""
    from src.inference import YOLODetector

    try:
        det = YOLODetector(weights=weights, imgsz=imgsz, conf=conf, iou=iou, device=device, backend=backend)

        src = source
        try:
            src = int(source)
        except ValueError:
            pass

        if isinstance(src, int):
            det.detect_video(src, save_path=save, show=not no_show)
        elif Path(src).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            detections = det.detect_image(src, save_path=save)
            for d in detections:
                console.print(
                    f"  {d.class_name:15s} {d.confidence:.3f}  "
                    f"[{d.x1},{d.y1},{d.x2},{d.y2}]"
                )
        else:
            det.detect_video(src, save_path=save, show=not no_show)
    except Exception as e:
        console.print(f"[error]Detection failed: {e}[/error]")
        sys.exit(1)


# ── Benchmark ────────────────────────────────────────────────────────────────

@main.command()
@click.option("--weights", required=True, multiple=True, help="Model weight files")
@click.option("--imgsz", default=640, type=int, help="Image size")
@click.option("--runs", default=100, type=int, help="Inference runs")
@click.option("--device", default="cpu", help="Device")
def benchmark(weights, imgsz, runs, device):
    """Benchmark model inference speed."""
    from src.benchmark import run_benchmark

    try:
        run_benchmark(list(weights), imgsz=imgsz, n_runs=runs, device=device)
    except Exception as e:
        console.print(f"[error]Benchmark failed: {e}[/error]")
        sys.exit(1)


# ── Serve ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--weights", default="weights/best.onnx", help="Model weights")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--workers", default=1, type=int, help="Server workers")
@click.option("--imgsz", default=640, type=int, help="Inference image size")
@click.option("--backend", default=None, type=click.Choice(["opencv", "onnxruntime", "ultralytics"]), help="Force inference backend")
def serve(weights, host, port, workers, imgsz, backend):
    """Start REST API inference server."""
    try:
        import uvicorn
        from src.api import create_app

        app = create_app(weights_path=weights, imgsz=imgsz, backend=backend)
        uvicorn.run(app, host=host, port=port, workers=workers)
    except ImportError:
        console.print("[error]Install API deps: pip install yolocc[api][/error]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[error]Server failed: {e}[/error]")
        sys.exit(1)


# ── Data ─────────────────────────────────────────────────────────────────────

@main.group()
def data():
    """Dataset preparation tools."""
    pass


@data.command("generate")
@click.option("--output", default="data/dataset", help="Output directory")
@click.option("--num", default=500, type=int, help="Number of images")
@click.option("--imgsz", default=640, type=int, help="Image size")
def data_generate(output, num, imgsz):
    """Generate synthetic test dataset."""
    from scripts.prepare_dataset import generate_synthetic_dataset

    generate_synthetic_dataset(output, num_images=num, imgsz=imgsz)
    console.print(f"[success]Generated {num} images at {output}[/success]")


@data.command("split")
@click.option("--data-dir", default="data/raw", help="Raw data directory")
@click.option("--ratio", default=[0.8, 0.1, 0.1], type=float, nargs=3, help="Split ratios")
def data_split(data_dir, ratio):
    """Split dataset into train/val/test."""
    from scripts.prepare_dataset import split_dataset

    split_dataset(data_dir, tuple(ratio))


@data.command("validate")
@click.option("--data-dir", default="data/dataset", help="Dataset directory")
def data_validate(data_dir):
    """Validate dataset format and integrity."""
    from scripts.prepare_dataset import validate_dataset

    validate_dataset(data_dir)


# ── Visualize ────────────────────────────────────────────────────────────────

@main.group()
def viz():
    """Visualization tools."""
    pass


@viz.command("training")
@click.option("--dir", default="results/runs/train/object_detect", help="Run directory")
def viz_training(dir):
    """Plot training curves."""
    from src.visualize import plot_training_curves

    plot_training_curves(dir)


@viz.command("benchmark")
@click.option("--json", "json_path", default="results/benchmark_results.json", help="Benchmark JSON")
def viz_benchmark(json_path):
    """Plot benchmark comparison."""
    from src.visualize import plot_benchmark

    plot_benchmark(json_path)


@viz.command("distribution")
@click.option("--dir", default="data/dataset/labels/train", help="Labels directory")
def viz_distribution(dir):
    """Plot class distribution."""
    from src.visualize import plot_class_distribution

    plot_class_distribution(dir)


if __name__ == "__main__":
    main()
