"""
Download and prepare real-world garbage detection datasets.

Supports multiple sources:
  1. trashnet    — Stanford TrashNet (2,527 images, 6 classes, classification → detection)
  2. trashbox    — TACO-style litter detection (COCO → YOLO conversion)
  3. custom      — user-provided folder with images

Class mapping (6 → 4 Chinese garbage categories):
  cardboard → recyclable (0)
  glass     → recyclable (0)
  metal     → recyclable (0)
  paper     → recyclable (0)
  plastic   → recyclable (0)
  trash     → other (3)

Usage:
    # Download TrashNet and convert to YOLO detection format
    python scripts/download_real_dataset.py --source trashnet --output data/real

    # Use auto-labeling with pretrained YOLOv8 for better bounding boxes
    python scripts/download_real_dataset.py --source trashnet --output data/real --auto-label

    # Use a custom image folder (each subfolder = one class)
    python scripts/download_real_dataset.py --source custom --input /path/to/images --output data/real
"""

import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_logger, log_kv, log_section, log_success, log_table

log = get_logger("download_dataset")

# ── Class mapping ─────────────────────────────────────────────────────────────
# Maps source dataset class names → yolocc 4-class IDs
# Must match configs/garbage.yaml and scripts/prepare_dataset.py

TRASHNET_CLASSES = {
    "cardboard": 0,  # recyclable
    "glass":     0,  # recyclable
    "metal":     0,  # recyclable
    "paper":     0,  # recyclable
    "plastic":   0,  # recyclable
    "trash":     3,  # other
}

YOLOCC_CLASSES = {
    0: "recyclable",
    1: "hazardous",
    2: "kitchen",
    3: "other",
}

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress logging."""
    import urllib.request

    log.info(f"Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(str(url), str(dest))
        log.info(f"Downloaded: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        log.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract zip and return the top-level directory."""
    log.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # Find the extracted top-level directory
    dirs = [d for d in dest.iterdir() if d.is_dir()]
    if len(dirs) == 1:
        return dirs[0]
    return dest


def find_images_in_folder(folder: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[Path]:
    """Recursively find all image files in a folder."""
    images = []
    for ext in exts:
        images.extend(folder.rglob(f"*{ext}"))
        images.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def classify_by_parent_folder(image_path: Path, base_dir: Path) -> Optional[str]:
    """Get class name from the parent folder name."""
    rel = image_path.relative_to(base_dir)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0].lower()
    return None


def generate_full_frame_label(img_w: int, img_h: int, margin: float = 0.02) -> Tuple[float, float, float, float]:
    """Generate a YOLO label for the full image frame with a small margin.

    For classification-origin images where the object fills most of the frame,
    we create a bounding box covering (nearly) the entire image.
    This is a reasonable approximation when no manual annotation exists.
    """
    cx = 0.5
    cy = 0.5
    w = 1.0 - 2 * margin
    h = 1.0 - 2 * margin
    return (cx, cy, w, h)


def auto_label_with_yolo(image_path: Path, model) -> List[Tuple[int, float, float, float, float]]:
    """Use a pretrained YOLOv8 model to generate bounding box labels.

    Returns list of (class_id, cx, cy, w, h) in YOLO normalized format.
    """
    results = model(str(image_path), verbose=False)
    labels = []
    for r in results:
        img_h, img_w = r.orig_shape
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # Convert to YOLO normalized format
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Map COCO class to our garbage classes (heuristic)
            # COCO has 80 classes, we map plausible ones
            coco_id = int(box.cls[0])
            garbage_id = _coco_to_garbage(coco_id)

            if garbage_id is not None and conf > 0.3:
                labels.append((garbage_id, cx, cy, w, h))

    return labels


def _coco_to_garbage(coco_id: int) -> Optional[int]:
    """Map COCO class ID to garbage class ID.

    This is a heuristic mapping for auto-labeling.
    Returns None if the class doesn't map to garbage.
    """
    coco_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        39: "bottle",     # → recyclable
        41: "cup",        # → recyclable
        43: "knife",      # → hazardous (sharp)
        44: "spoon",      # → recyclable (metal)
        45: "bowl",       # → recyclable
        46: "banana",     # → kitchen
        47: "apple",      # → kitchen
        48: "sandwich",   # → kitchen
        49: "orange",     # → kitchen
        50: "broccoli",   # → kitchen
        51: "carrot",     # → kitchen
        52: "hot dog",    # → kitchen
        53: "pizza",      # → kitchen
        54: "donut",      # → kitchen
        55: "cake",       # → kitchen
        56: "chair",      # → recyclable (furniture)
        57: "couch",      # → recyclable
        58: "potted plant",  # → kitchen (organic)
        59: "bed",        # → other
        61: "toilet",     # → other
        62: "tv",         # → recyclable (electronic)
        63: "laptop",     # → recyclable (electronic)
        64: "mouse",      # → recyclable (electronic)
        65: "remote",     # → recyclable (electronic)
        66: "keyboard",   # → recyclable (electronic)
        67: "cell phone", # → recyclable (electronic)
        72: "book",       # → recyclable (paper)
        73: "clock",      # → recyclable
        74: "scissors",   # → hazardous (sharp)
        75: "teddy bear", # → other
        76: "hair drier", # → recyclable (electronic)
        77: "toothbrush", # → other
    }

    recyclable_ids = {39, 41, 44, 45, 56, 57, 62, 63, 64, 65, 66, 67, 72, 73, 76}
    hazardous_ids = {43, 74}
    kitchen_ids = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58}
    other_ids = {59, 61, 75, 77}

    if coco_id in recyclable_ids:
        return 0
    elif coco_id in hazardous_ids:
        return 1
    elif coco_id in kitchen_ids:
        return 2
    elif coco_id in other_ids:
        return 3
    return None


def prepare_trashnet(source_dir: Path, output_dir: Path, class_map: Dict, auto_label: bool = False) -> Dict:
    """Convert TrashNet classification dataset to YOLO detection format.

    Args:
        source_dir: Path to extracted TrashNet dataset (with class subfolders).
        output_dir: Output path for YOLO format dataset.
        class_map: Mapping from source class name to yolocc class ID.
        auto_label: If True, use pretrained YOLOv8 to generate bounding boxes.
                    If False, use full-frame bounding boxes.

    Returns:
        Stats dict with counts per split and class.
    """
    log_section("Preparing TrashNet → YOLO Detection", log)

    # Find all images organized by class
    class_images: Dict[str, List[Path]] = {}
    for class_name in class_map:
        class_dir = source_dir / class_name
        if class_dir.exists():
            imgs = find_images_in_folder(class_dir)
            if imgs:
                class_images[class_name] = imgs
                log.info(f"  {class_name:15s}: {len(imgs)} images → class {class_map[class_name]} ({YOLOCC_CLASSES[class_map[class_name]]})")

    if not class_images:
        log.error(f"No class folders found in {source_dir}")
        return {}

    total_images = sum(len(v) for v in class_images.values())
    log_kv("Total images", total_images, log)

    # Auto-label with pretrained model
    model = None
    if auto_label:
        log.info("Loading pretrained YOLOv8n for auto-labeling...")
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        log.info("Auto-labeling enabled: generating bounding boxes from pretrained model")

    # Split 70/15/15
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    stats = {s: {"images": 0, "labels": 0, "boxes": 0, "classes": Counter()} for s in splits}

    for split in splits:
        for d in ["images", "labels"]:
            (output_dir / d / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    all_class_counters = Counter()

    for class_name, images in class_images.items():
        garbage_id = class_map[class_name]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])

        split_images = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, split_imgs in split_images.items():
            for idx, img_path in enumerate(split_imgs):
                img = cv2.imread(str(img_path))
                if img is None:
                    log.warning(f"  Cannot read: {img_path}")
                    continue

                h, w = img.shape[:2]

                # Generate YOLO labels
                if model is not None:
                    # Auto-label: use pretrained model to detect objects
                    detected_labels = auto_label_with_yolo(img_path, model)
                    if not detected_labels:
                        # Fallback: full frame with the known class
                        cx, cy, bw, bh = generate_full_frame_label(w, h)
                        detected_labels = [(garbage_id, cx, cy, bw, bh)]
                else:
                    # Full-frame label
                    cx, cy, bw, bh = generate_full_frame_label(w, h)
                    detected_labels = [(garbage_id, cx, cy, bw, bh)]

                # Save image
                out_name = f"{class_name}_{idx:04d}.jpg"
                out_img = output_dir / "images" / split / out_name
                cv2.imwrite(str(out_img), img)

                # Save label
                out_lbl = output_dir / "labels" / split / (Path(out_name).stem + ".txt")
                with open(out_lbl, "w") as f:
                    for lid, lx, ly, lw, lh in detected_labels:
                        f.write(f"{lid} {lx:.6f} {ly:.6f} {lw:.6f} {lh:.6f}\n")
                        stats[split]["boxes"] += 1
                        stats[split]["classes"][lid] += 1
                        all_class_counters[lid] += 1

                stats[split]["images"] += 1
                stats[split]["labels"] += 1

    # Report
    log_section("Dataset Summary", log)
    rows = []
    for split in splits:
        s = stats[split]
        rows.append([split, str(s["images"]), str(s["labels"]), str(s["boxes"])])
    log_table(["Split", "Images", "Labels", "Boxes"], rows, log)

    log.info("\n  Class distribution (all splits):")
    class_rows = []
    for cid in sorted(all_class_counters.keys()):
        name = YOLOCC_CLASSES.get(cid, f"class_{cid}")
        class_rows.append([name, str(all_class_counters[cid])])
    log_table(["Class", "Boxes"], class_rows, log)

    log_success(f"Dataset saved to: {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and prepare real garbage datasets")
    parser.add_argument("--source", choices=["trashnet", "custom"], default="trashnet",
                        help="Dataset source")
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory for custom source")
    parser.add_argument("--output", type=str, default="data/real",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--auto-label", action="store_true",
                        help="Use pretrained YOLOv8 to generate bounding boxes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)

    if args.source == "trashnet":
        # Download
        cache_dir = Path("data/.cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "trashnet.zip"
        extract_dir = cache_dir / "trashnet"

        if not extract_dir.exists():
            if not zip_path.exists():
                ok = download_file(TRASHNET_URL, zip_path, "TrashNet dataset")
                if not ok:
                    log.error("Download failed. Check network or download manually.")
                    log.info(f"Manual download: {TRASHNET_URL}")
                    log.info(f"Save to: {zip_path}")
                    sys.exit(1)

            source_dir = extract_zip(zip_path, extract_dir)
        else:
            source_dir = extract_dir

        # Find the actual dataset folder (may be nested)
        possible = list(extract_dir.rglob("dataset-resized"))
        if possible:
            source_dir = possible[0]
        else:
            source_dir = extract_dir

        log.info(f"Source directory: {source_dir}")

        stats = prepare_trashnet(
            source_dir=source_dir,
            output_dir=output_dir,
            class_map=TRASHNET_CLASSES,
            auto_label=args.auto_label,
        )

    elif args.source == "custom":
        if not args.input:
            log.error("--input required for custom source")
            sys.exit(1)
        input_dir = Path(args.input)
        if not input_dir.exists():
            log.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        # For custom, we use the same TrashNet class mapping
        # Users should organize images in class subfolders
        stats = prepare_trashnet(
            source_dir=input_dir,
            output_dir=output_dir,
            class_map=TRASHNET_CLASSES,
            auto_label=args.auto_label,
        )

    if stats:
        log_section("Next Steps", log)
        log.info("1. Validate the dataset:")
        log.info(f"   python scripts/prepare_dataset.py validate --data-dir {output_dir}")
        log.info("2. Copy or symlink to data/dataset for training:")
        log.info(f"   cp -r {output_dir}/* data/dataset/")
        log.info("3. Train:")
        log.info("   yolocc train --cfg configs/train_cfg.yaml --epochs 100 --device 0")


if __name__ == "__main__":
    main()
