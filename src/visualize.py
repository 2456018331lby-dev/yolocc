"""
Visualization utilities - Training curves, confusion matrix, PR curves
Usage:
    python -m src.visualize --type training --dir results/runs/train/garbage_detect
    python -m src.visualize --type compare --json results/benchmark_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_training_curves(run_dir: str, save_dir: str = None):
    """Plot training loss and metric curves from YOLO training run."""
    run_dir = Path(run_dir)
    save_dir = Path(save_dir or run_dir / "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV results
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[ERROR] results.csv not found in {run_dir}")
        return

    import csv
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                key = key.strip()
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val.strip()))
                except (ValueError, AttributeError):
                    data[key].append(0.0)

    epochs = data.get("epoch", list(range(len(next(iter(data.values()))))))

    # Plot 1: Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves", fontsize=16)

    loss_keys = [
        ("train/box_loss", "Box Loss"),
        ("train/cls_loss", "Classification Loss"),
        ("train/dfl_loss", "DFL Loss"),
    ]

    for ax, (key, title) in zip(axes.flat[:3], loss_keys):
        key_clean = key.strip()
        if key_clean in data:
            ax.plot(epochs, data[key_clean], "b-", linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

    # Plot 4: mAP
    map50_key = "metrics/mAP50(B)"
    map_key = "metrics/mAP50-95(B)"
    ax = axes.flat[3]
    if map50_key in data:
        ax.plot(epochs, data[map50_key], "g-", linewidth=2, label="mAP50")
    if map_key in data:
        ax.plot(epochs, data[map_key], "r-", linewidth=2, label="mAP50-95")
    ax.set_title("mAP Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {out_path}")

    # Plot 2: Learning rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    lr_key = "lr/pg0"
    if lr_key in data:
        ax.plot(epochs, data[lr_key], "m-", linewidth=2)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = save_dir / "lr_schedule.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_benchmark(json_path: str, save_path: str = None):
    """Plot benchmark comparison bar chart."""
    with open(json_path, "r") as f:
        results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    backends = [r["backend"] for r in results]
    fps_vals = [r["fps"] for r in results]
    size_vals = [r["model_size_mb"] for r in results]
    mean_ms = [r["mean_ms"] for r in results]

    # FPS comparison
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax1.bar(backends, fps_vals, color=colors[:len(backends)])
    ax1.set_title("Inference Speed (FPS)", fontsize=14)
    ax1.set_ylabel("FPS")
    for bar, val in zip(bars, fps_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Model size comparison
    bars = ax2.bar(backends, size_vals, color=colors[:len(backends)])
    ax2.set_title("Model Size (MB)", fontsize=14)
    ax2.set_ylabel("MB")
    for bar, val in zip(bars, size_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = save_path or "results/benchmark_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_class_distribution(labels_dir: str, save_path: str = None):
    """Plot class distribution histogram."""
    labels_dir = Path(labels_dir)
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = {0: "recyclable", 1: "hazardous", 2: "kitchen", 3: "other"}

    for lbl_file in labels_dir.rglob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls = int(parts[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    except ValueError:
                        pass

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [class_names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    colors = ["#4CAF50", "#F44336", "#2196F3", "#9E9E9E"]

    bars = ax.bar(names, counts, color=colors)
    ax.set_title("Class Distribution", fontsize=14)
    ax.set_ylabel("Number of Instances")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = save_path or "results/class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization tools")
    parser.add_argument("--type", type=str, required=True,
                        choices=["training", "benchmark", "distribution"],
                        help="Visualization type")
    parser.add_argument("--dir", type=str, default=None, help="Run directory or labels directory")
    parser.add_argument("--json", type=str, default=None, help="Benchmark JSON path")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.type == "training":
        if not args.dir:
            args.dir = "results/runs/train/garbage_detect"
        plot_training_curves(args.dir, args.save)

    elif args.type == "benchmark":
        json_path = args.json or "results/benchmark_results.json"
        plot_benchmark(json_path, args.save)

    elif args.type == "distribution":
        if not args.dir:
            args.dir = "data/dataset/labels/train"
        plot_class_distribution(args.dir, args.save)


if __name__ == "__main__":
    main()
