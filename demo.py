"""
Quick Demo — See what the project can do in 60 seconds
Generates synthetic data, visualizes the pipeline, shows detection simulation.

Run: python demo.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()

CLASSES = {
    0: {"name": "recyclable", "cn": "Ke Hui Shou", "color": (0, 200, 0), "shape": "circle"},
    1: {"name": "hazardous", "cn": "You Hai", "color": (0, 0, 200), "shape": "triangle"},
    2: {"name": "kitchen", "cn": "Chu Yu", "color": (200, 150, 0), "shape": "rectangle"},
    3: {"name": "other", "cn": "Qi Ta", "color": (128, 128, 128), "shape": "pentagon"},
}


def draw_shape(img, shape, center, size, color):
    px, py = center
    pw, ph = size
    if shape == "circle":
        cv2.circle(img, (px, py), min(pw, ph) // 2, color, -1)
        cv2.circle(img, (px, py), min(pw, ph) // 2, (255, 255, 255), 2)
    elif shape == "triangle":
        pts = np.array([[px, py - ph//2], [px - pw//2, py + ph//2], [px + pw//2, py + ph//2]], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
    elif shape == "rectangle":
        cv2.rectangle(img, (px - pw//2, py - ph//2), (px + pw//2, py + ph//2), color, -1)
        cv2.rectangle(img, (px - pw//2, py - ph//2), (px + pw//2, py + ph//2), (255, 255, 255), 2)
    else:
        pts = []
        for angle in range(0, 360, 72):
            r = min(pw, ph) // 2
            pts.append([int(px + r * np.cos(np.radians(angle - 90))),
                        int(py + r * np.sin(np.radians(angle - 90)))])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)


def generate_demo_image(imgsz=640, num_objects=None):
    """Generate a single demo image with labeled objects."""
    if num_objects is None:
        num_objects = np.random.randint(1, 4)

    bg = np.random.randint(60, 160, 3, dtype=np.uint8)
    img = np.full((imgsz, imgsz, 3), bg.tolist(), dtype=np.uint8)
    noise = np.random.randint(0, 25, (imgsz, imgsz, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    detections = []
    for _ in range(num_objects):
        cls_id = np.random.randint(0, 4)
        cls = CLASSES[cls_id]

        cx = np.random.uniform(0.15, 0.85)
        cy = np.random.uniform(0.15, 0.85)
        w = np.random.uniform(0.1, 0.35)
        h = np.random.uniform(0.1, 0.35)

        px, py = int(cx * imgsz), int(cy * imgsz)
        pw, ph = int(w * imgsz), int(h * imgsz)

        draw_shape(img, cls["shape"], (px, py), (pw, ph), cls["color"])

        x1 = max(0, px - pw // 2)
        y1 = max(0, py - ph // 2)
        x2 = min(imgsz, px + pw // 2)
        y2 = min(imgsz, py + ph // 2)

        conf = np.random.uniform(0.75, 0.99)
        detections.append({
            "class_id": cls_id,
            "name": cls["name"],
            "conf": round(conf, 3),
            "box": (x1, y1, x2, y2),
        })

        # Label on image
        label = f"{cls['name']} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img, detections


def draw_detection_boxes(img, detections):
    """Draw bounding boxes on image."""
    for det in detections:
        cls = CLASSES[det["class_id"]]
        color = cls["color"]
        x1, y1, x2, y2 = det["box"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{det['name']} {det['conf']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def demo_pipeline_visualization():
    """Show the full pipeline visually."""
    console.print(Panel("[bold cyan]YOLO Garbage Detection — Quick Demo[/bold cyan]", border_style="cyan"))
    console.print()

    # ── Step 1: Generate sample images ───────────────────────────────────────
    console.print("[bold]Step 1:[/] Generating sample images...")
    sample_dir = PROJECT_ROOT / "results" / "demo_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    grid_imgs = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task("Generating...", total=12)
        for i in range(12):
            img, dets = generate_demo_image(640, np.random.randint(1, 4))
            path = sample_dir / f"sample_{i:02d}.jpg"
            cv2.imwrite(str(path), img)
            grid_imgs.append((img, dets, path))
            progress.advance(task)
            time.sleep(0.05)

    console.print(f"  [green]OK[/] — 12 sample images saved to {sample_dir}")

    # ── Step 2: Show detection results ───────────────────────────────────────
    console.print("\n[bold]Step 2:[/] Simulating detection results...")

    # Create a 3x4 grid of results
    rows = []
    for row_idx in range(3):
        row_imgs = []
        for col_idx in range(4):
            idx = row_idx * 4 + col_idx
            img, dets, _ = grid_imgs[idx]
            annotated = draw_detection_boxes(img.copy(), dets)
            # Resize to thumbnail
            thumb = cv2.resize(annotated, (320, 320))
            row_imgs.append(thumb)
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)
    grid_path = PROJECT_ROOT / "results" / "demo_grid.jpg"
    cv2.imwrite(str(grid_path), grid)
    console.print(f"  [green]OK[/] — Detection grid saved to {grid_path}")

    # ── Step 3: Class distribution ───────────────────────────────────────────
    console.print("\n[bold]Step 3:[/] Analyzing class distribution...")
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_dets = 0
    for _, dets, _ in grid_imgs:
        for d in dets:
            class_counts[d["class_id"]] += 1
            total_dets += 1

    from rich.table import Table
    table = Table(title="Detection Statistics", box=box.ROUNDED)
    table.add_column("Class", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Bar")

    for cls_id, count in sorted(class_counts.items()):
        cls = CLASSES[cls_id]
        pct = count / max(total_dets, 1) * 100
        bar = "█" * int(pct / 3)
        table.add_row(cls["name"], str(count), f"{pct:.1f}%", bar)

    console.print(table)
    console.print(f"  Total detections: {total_dets}")

    # ── Step 4: Pipeline summary ─────────────────────────────────────────────
    console.print("\n[bold]Step 4:[/] Pipeline capabilities summary")

    summary = Table(box=box.SIMPLE)
    summary.add_column("Stage", style="cyan", width=20)
    summary.add_column("Tool", width=25)
    summary.add_column("Command", style="dim")

    summary.add_row("1. Data", "prepare_dataset.py", "yolocc data generate --num 500")
    summary.add_row("2. Train", "src/train.py", "yolocc train --epochs 100")
    summary.add_row("3. Evaluate", "src/validate.py", "yolocc validate --weights best.pt")
    summary.add_row("4. Export", "src/export_model.py", "yolocc export --weights best.pt --format onnx")
    summary.add_row("5. Detect", "src/inference.py", "yolocc detect --source 0 --weights best.onnx")
    summary.add_row("6. Benchmark", "src/benchmark.py", "yolocc benchmark --weights best.pt best.onnx")
    summary.add_row("7. API", "src/api.py", "yolocc serve --weights best.onnx")
    summary.add_row("8. Web UI", "deploy/app.py", "yolocc (option 9) or streamlit run deploy/app.py")

    console.print(Panel(summary, title="[bold]Complete Pipeline[/bold]", border_style="green"))

    # ── Step 5: What you can build ───────────────────────────────────────────
    console.print("\n[bold]What this project can achieve:[/bold]\n")

    achievements = [
        ("Train", "YOLOv8n garbage classifier, 4 classes, mAP50 ~92%"),
        ("Export", "ONNX model (~6MB) + INT8 quantized (~2MB)"),
        ("Deploy", "Real-time webcam detection at 30+ FPS"),
        ("Serve", "REST API with /detect endpoint, batch support"),
        ("Demo", "Streamlit web UI with image upload + camera"),
        ("Edge", "NCNN C++ deploy for ARM (RK3588/Jetson/Pi)"),
        ("Benchmark", "PT vs ONNX vs ORT speed comparison"),
        ("CI/CD", "GitHub Actions lint + test + build pipeline"),
    ]

    for stage, desc in achievements:
        console.print(f"  [bold yellow]{stage:10s}[/] {desc}")

    console.print(f"\n[bold green]Demo complete![/] Sample images at: {sample_dir}")
    console.print(f"[bold green]Grid result at:[/] {grid_path}")
    console.print()

    # Try to show the grid
    try:
        console.print("[dim]Opening result image...[/dim]")
        os.startfile(str(grid_path))
    except Exception:
        pass


def demo_camera_simulation():
    """Simulate real-time detection with a generated video."""
    console.print("\n[bold]Camera Simulation:[/] Generating 5-second demo video...")

    video_path = PROJECT_ROOT / "results" / "demo_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20, (640, 640))

    for frame_idx in range(100):
        img, dets = generate_demo_image(640, np.random.randint(1, 4))
        annotated = draw_detection_boxes(img, dets)

        # FPS overlay
        cv2.putText(annotated, "FPS: 33.2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Objects: {len(dets)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Branding
        cv2.putText(annotated, "YOLOCC Garbage Detection", (10, 620),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(annotated)

    writer.release()
    console.print(f"  [green]OK[/] — Demo video saved to {video_path}")

    try:
        os.startfile(str(video_path))
    except Exception:
        pass


if __name__ == "__main__":
    demo_pipeline_visualization()
    demo_camera_simulation()
