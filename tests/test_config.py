"""
Configuration validation tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    AugmentationConfig,
    DatasetConfig,
    ExportConfig,
    InferenceConfig,
    TrainingConfig,
    load_dataset_config,
    load_training_config,
)


class TestTrainingConfig:

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model == "yolov8n.pt"
        assert cfg.epochs == 100
        assert cfg.imgsz == 640
        assert cfg.batch == 16

    def test_custom_values(self):
        cfg = TrainingConfig(epochs=50, batch=8, imgsz=320)
        assert cfg.epochs == 50
        assert cfg.batch == 8
        assert cfg.imgsz == 320

    def test_validation_epochs_range(self):
        with pytest.raises(Exception):
            TrainingConfig(epochs=0)

    def test_validation_batch_range(self):
        with pytest.raises(Exception):
            TrainingConfig(batch=0)

    def test_augmentation_config(self):
        aug = AugmentationConfig(mosaic=0.5, fliplr=0.3)
        assert aug.mosaic == 0.5
        assert aug.fliplr == 0.3

    def test_augmentation_in_training(self):
        cfg = TrainingConfig(augmentation=AugmentationConfig(mosaic=0.0))
        assert cfg.augmentation.mosaic == 0.0


class TestDatasetConfig:

    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.nc is None
        assert 0 in cfg.names
        assert cfg.names[0] == "recyclable"
        assert len(cfg.names) == 4

    def test_custom_names(self):
        cfg = DatasetConfig(names={0: "cat", 1: "dog"})
        assert cfg.names[0] == "cat"


class TestExportConfig:

    def test_defaults(self):
        cfg = ExportConfig(weights="best.pt")
        assert cfg.imgsz == 640
        assert cfg.int8 is False
        assert cfg.simplify is True


class TestInferenceConfig:

    def test_defaults(self):
        cfg = InferenceConfig(weights="best.onnx")
        assert cfg.conf == 0.25
        assert cfg.iou == 0.45
        assert cfg.max_det == 300


class TestYAMLConfigs:

    def test_garbage_yaml_loads(self):
        cfg = load_dataset_config("configs/garbage.yaml")
        assert len(cfg.names) == 4
        assert cfg.names[0] == "recyclable"

    def test_template_object_yaml_loads(self):
        cfg = load_dataset_config("configs/template_object.yaml")
        assert len(cfg.names) == 6
        assert cfg.names[0] == "container"
        assert cfg.names[3] == "hazard"

    def test_household37_future_yaml_loads(self):
        cfg = load_dataset_config("configs/household37_future.yaml")
        assert len(cfg.names) == 37
        assert cfg.names[17] == "cup"
        assert cfg.names[35] == "computer"

    def test_train_cfg_loads(self):
        cfg = load_training_config("configs/train_cfg.yaml")
        assert cfg.epochs > 0
        assert cfg.imgsz > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
