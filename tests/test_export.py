"""
Export and config tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigs:

    def test_train_cfg_valid_yaml(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "train_cfg.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "model" in cfg
        assert "epochs" in cfg
        assert "imgsz" in cfg
        assert "batch" in cfg

    def test_garbage_yaml_valid(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "garbage.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "names" in cfg
        assert len(cfg["names"]) == 4
        assert cfg["names"][0] == "recyclable"
        assert cfg["names"][1] == "hazardous"
        assert cfg["names"][2] == "kitchen"
        assert cfg["names"][3] == "other"

    def test_garbage_yaml_has_colors(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "garbage.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "colors" in cfg
        assert len(cfg["colors"]) == 4

    def test_template_object_yaml_valid(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "template_object.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "names" in cfg
        assert len(cfg["names"]) == 6
        assert cfg["names"][0] == "container"
        assert cfg["names"][5] == "misc"

    def test_pyproject_valid(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("pyproject.toml not found")

        content = pyproject.read_text(encoding="utf-8")
        assert 'name = "yolocc"' in content
        assert "[project.scripts]" in content
        assert "yolocc" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
