"""
Dataset Preparation — Industrial Grade
Supports:
  1. Generate synthetic test dataset (shapes as placeholders)
  2. Split flat dataset into train/val/test
  3. Validate YOLO label format + integrity checks
  4. Statistics report

Usage:
    python scripts/prepare_dataset.py generate --num 500
    python scripts/prepare_dataset.py split --data-dir data/raw
    python scripts/prepare_dataset.py validate --data-dir data/dataset
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.logger import get_logger, log_kv, log_section, log_success, log_table

log = get_logger("dataset")

# Class definitions
CLASSES = {
    0: {"name": "recyclable", "color": (0, 200, 0), "shape": "circle"},
    1: {"name": "hazardous", "color": (0, 0, 200), "shape": "triangle"},
    2: {"name": "kitchen", "color": (200, 150, 0), "shape": "rectangle"},
    3: {"name": "other", "color": (128, 128, 128), "shape": "pentagon"},
}


def _draw_shape(img: np.ndarray, shape: str, center: Tuple[int, int],
                size: Tuple[int, int], color: Tuple[int, int, int]):
    """Draw a shape on the image."""
    px, py = center
    pw, ph = size

    if shape == "circle":
        cv2.circle(img, (px, py), min(pw, ph) // 2, color, -1)
        cv2.circle(img, (px, py), min(pw, ph) // 2, (255, 255, 255), 2)
    elif shape == "triangle":
        pts = np.array([
            [px, py - ph // 2],
            [px - pw // 2, py + ph // 2],
            [px + pw // 2, py + ph // 2],
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
    elif shape == "rectangle":
        cv2.rectangle(img, (px - pw // 2, py - ph // 2), (px + pw // 2, py + ph // 2), color, -1)
        cv2.rectangle(img, (px - pw // 2, py - ph // 2), (px + pw // 2, py + ph // 2), (255, 255, 255), 2)
    else:  # pentagon
        pts = []
        for angle in range(0, 360, 72):
            r = min(pw, ph) // 2
            ax = int(px + r * np.cos(np.radians(angle - 90)))
            ay = int(py + r * np.sin(np.radians(angle - 90)))
            pts.append([ax, ay])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)


def generate_synthetic_dataset(
    output_dir: str,
    num_images: int = 500,
    imgsz: int = 640,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Generate synthetic garbage detection dataset.

    Returns:
        Dict with split counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    n_train = int(num_images * 0.7)
    n_val = int(num_images * 0.15)
    n_test = num_images - n_train - n_val
    splits = {"train": n_train, "val": n_val, "test": n_test}

    log_section("Generating Synthetic Dataset", log)
    log_kv("Total images", num_images, log)
    log_kv("Image size", imgsz, log)
    log_kv("Output", str(output_dir), log)

    for split, count in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            # Random background
            bg = np.random.randint(80, 180, 3, dtype=np.uint8)
            img = np.full((imgsz, imgsz, 3), bg.tolist(), dtype=np.uint8)
            noise = np.random.randint(0, 30, (imgsz, imgsz, 3), dtype=np.uint8)
            img = cv2.add(img, noise)

            # 1-3 objects
            num_objects = random.randint(1, 3)
            labels = []

            for _ in range(num_objects):
                cls_id = random.randint(0, 3)
                cls = CLASSES[cls_id]

                cx = random.uniform(0.2, 0.8)
                cy = random.uniform(0.2, 0.8)
                w = random.uniform(0.1, 0.3)
                h = random.uniform(0.1, 0.3)

                px, py = int(cx * imgsz), int(cy * imgsz)
                pw, ph = int(w * imgsz), int(h * imgsz)

                _draw_shape(img, cls["shape"], (px, py), (pw, ph), cls["color"])

                # Label text
                cv2.putText(
                    img, cls["name"],
                    (px - pw // 2, py - ph // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                )

                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            img_name = f"{i:04d}.jpg"
            lbl_name = f"{i:04d}.txt"
            cv2.imwrite(str(img_dir / img_name), img)
            with open(lbl_dir / lbl_name, "w") as f:
                f.write("\n".join(labels))

        log.info(f"  {split}: {count} images generated")

    log_success(f"Dataset generated at: {output_dir}")
    return splits


def split_dataset(
    data_dir: str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> None:
    """Split a flat dataset into train/val/test."""
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = data_dir.parent / "dataset"

    images = sorted(
        list((data_dir / "images").glob("*.jpg"))
        + list((data_dir / "images").glob("*.png"))
    )

    if not images:
        log.error(f"No images found in {data_dir / 'images'}")
        return

    random.shuffle(images)
    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    log_section("Splitting Dataset", log)
    log_kv("Source", str(data_dir), log)
    log_kv("Output", str(output_dir), log)
    log_kv("Total images", n, log)

    for split, imgs in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        missing = 0
        for img_path in imgs:
            lbl_path = data_dir / "labels" / (img_path.stem + ".txt")
            shutil.copy2(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            else:
                missing += 1

        log.info(f"  {split}: {len(imgs)} images" + (f" ({missing} labels missing)" if missing else ""))

    log_success("Dataset split complete.")


def validate_dataset(data_dir: str) -> Dict:
    """Validate YOLO dataset format and return statistics."""
    data_dir = Path(data_dir)
    errors: List[str] = []
    class_counter: Counter = Counter()
    total_images = 0
    total_labels = 0
    total_boxes = 0

    log_section("Validating Dataset", log)

    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        if not img_dir.exists():
            log.warning(f"  Missing directory: {img_dir}")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        total_images += len(images)

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                errors.append(f"Missing label: {lbl_path}")
                continue

            total_labels += 1
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Corrupt image: {img_path}")
                continue

            with open(lbl_path, "r") as f:
                for line_num, line in enumerate(f.readlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"{lbl_path}:{line_num} — Expected 5 values, got {len(parts)}")
                        continue

                    try:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                    except ValueError:
                        errors.append(f"{lbl_path}:{line_num} — Invalid number format")
                        continue

                    if cls_id not in CLASSES:
                        errors.append(f"{lbl_path}:{line_num} — Unknown class id: {cls_id}")

                    if not all(0 <= v <= 1 for v in coords):
                        errors.append(f"{lbl_path}:{line_num} — Coordinates out of [0,1] range")

                    _, _, box_w, box_h = coords
                    if box_w <= 0 or box_h <= 0:
                        errors.append(f"{lbl_path}:{line_num} — Box width/height must be > 0")

                    total_boxes += 1
                    class_counter[cls_id] += 1

    # Report
    log_kv("Total images", total_images, log)
    log_kv("Total labels", total_labels, log)
    log_kv("Total boxes", total_boxes, log)

    if class_counter:
        log.info("\n  Class distribution:")
        rows = []
        for cls_id in sorted(class_counter.keys()):
            name = CLASSES.get(cls_id, {}).get("name", f"class_{cls_id}")
            rows.append([name, str(class_counter[cls_id])])
        log_table(["Class", "Count"], rows, log)

    if errors:
        log.warning(f"\n  Errors ({len(errors)}):")
        for e in errors[:20]:
            log.warning(f"    - {e}")
        if len(errors) > 20:
            log.warning(f"    ... and {len(errors) - 20} more")
    else:
        log_success("No errors found.")

    stats = {
        "total_images": total_images,
        "total_labels": total_labels,
        "total_boxes": total_boxes,
        "class_counts": dict(class_counter),
        "errors": len(errors),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    sub = parser.add_subparsers(dest="command")

    # Generate
    gen_p = sub.add_parser("generate", help="Generate synthetic dataset")
    gen_p.add_argument("--output", default="data/dataset")
    gen_p.add_argument("--num", type=int, default=500)
    gen_p.add_argument("--imgsz", type=int, default=640)

    # Split
    split_p = sub.add_parser("split", help="Split dataset")
    split_p.add_argument("--data-dir", default="data/raw")
    split_p.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1])

    # Validate
    val_p = sub.add_parser("validate", help="Validate dataset")
    val_p.add_argument("--data-dir", default="data/dataset")

    args = parser.parse_args()

    if args.command == "generate":
        generate_synthetic_dataset(args.output, args.num, args.imgsz)
    elif args.command == "split":
        split_dataset(args.data_dir, tuple(args.ratio))
    elif args.command == "validate":
        stats = validate_dataset(args.data_dir)
        raise SystemExit(1 if stats.get("errors", 0) else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
