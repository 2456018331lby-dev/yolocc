"""
Tests for download_real_dataset.py — class mapping, label generation, etc.
Run: pytest tests/test_download_dataset.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_real_dataset import (
    TRASHNET_CLASSES,
    YOLOCC_CLASSES,
    _coco_to_garbage,
    classify_by_parent_folder,
    generate_full_frame_label,
)


class TestClassMapping:
    """Test TrashNet 6→4 class mapping."""

    def test_recyclables_map_to_zero(self):
        for name in ("cardboard", "glass", "metal", "paper", "plastic"):
            assert TRASHNET_CLASSES[name] == 0, f"{name} should map to 0 (recyclable)"

    def test_trash_maps_to_three(self):
        assert TRASHNET_CLASSES["trash"] == 3

    def test_yolocc_classes_match_garbage_yaml(self):
        assert YOLOCC_CLASSES[0] == "recyclable"
        assert YOLOCC_CLASSES[1] == "hazardous"
        assert YOLOCC_CLASSES[2] == "kitchen"
        assert YOLOCC_CLASSES[3] == "other"

    def test_no_unknown_classes_in_map(self):
        for name, cid in TRASHNET_CLASSES.items():
            assert cid in YOLOCC_CLASSES, f"{name} maps to unknown class {cid}"


class TestFullFrameLabel:
    """Test full-frame YOLO label generation."""

    def test_label_centered(self):
        cx, cy, w, h = generate_full_frame_label(640, 480)
        assert cx == 0.5
        assert cy == 0.5

    def test_label_near_full_frame(self):
        cx, cy, w, h = generate_full_frame_label(640, 480)
        assert w > 0.9
        assert h > 0.9

    def test_label_within_bounds(self):
        cx, cy, w, h = generate_full_frame_label(640, 480)
        assert 0 <= cx - w / 2
        assert cx + w / 2 <= 1
        assert 0 <= cy - h / 2
        assert cy + h / 2 <= 1


class TestCocoToGarbageMapping:
    """Test COCO class → garbage class heuristic mapping."""

    def test_bottle_maps_recyclable(self):
        assert _coco_to_garbage(39) == 0  # bottle → recyclable

    def test_banana_maps_kitchen(self):
        assert _coco_to_garbage(46) == 2  # banana → kitchen

    def test_scissors_maps_hazardous(self):
        assert _coco_to_garbage(74) == 1  # scissors → hazardous

    def test_toothbrush_maps_other(self):
        assert _coco_to_garbage(77) == 3  # toothbrush → other

    def test_unmapped_class_returns_none(self):
        assert _coco_to_garbage(1) is None   # bicycle → None (not garbage)
        assert _coco_to_garbage(17) is None  # cat → None
        assert _coco_to_garbage(0) is None   # person → None


class TestClassifyByParentFolder:
    """Test extracting class name from folder structure."""

    def test_simple_structure(self, tmp_path):
        img = tmp_path / "cardboard" / "001.jpg"
        img.parent.mkdir()
        img.touch()
        result = classify_by_parent_folder(img, tmp_path)
        assert result == "cardboard"

    def test_nested_structure(self, tmp_path):
        img = tmp_path / "glass" / "train" / "001.jpg"
        img.parent.mkdir(parents=True)
        img.touch()
        result = classify_by_parent_folder(img, tmp_path)
        assert result == "glass"

    def test_no_class_folder(self, tmp_path):
        img = tmp_path / "001.jpg"
        img.touch()
        result = classify_by_parent_folder(img, tmp_path)
        assert result is None
