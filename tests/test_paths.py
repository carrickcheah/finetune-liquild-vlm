"""Tests for path utilities."""

import pytest
from pathlib import Path


class TestPathUtilities:
    """Test path utility functions."""

    def test_get_path_model_checkpoints_import(self):
        """Test that path function can be imported."""
        try:
            from ft_vlm.settings.paths import get_path_model_checkpoints_in_modal_volume
            assert get_path_model_checkpoints_in_modal_volume is not None
        except ImportError as e:
            pytest.fail(f"Failed to import path function: {e}")

    def test_checkpoint_path_returns_path(self):
        """Test that checkpoint path returns a Path object."""
        from ft_vlm.settings.paths import get_path_model_checkpoints_in_modal_volume

        path = get_path_model_checkpoints_in_modal_volume("test-experiment")

        assert path is not None
        assert isinstance(path, (str, Path))

    def test_checkpoint_path_includes_experiment_name(self):
        """Test that checkpoint path includes experiment name."""
        from ft_vlm.settings.paths import get_path_model_checkpoints_in_modal_volume

        experiment_name = "my-test-experiment"
        path = get_path_model_checkpoints_in_modal_volume(experiment_name)

        assert experiment_name in str(path)

    def test_checkpoint_path_is_absolute(self):
        """Test that checkpoint path is absolute."""
        from ft_vlm.settings.paths import get_path_model_checkpoints_in_modal_volume

        path = get_path_model_checkpoints_in_modal_volume("test")
        path_obj = Path(path) if isinstance(path, str) else path

        assert path_obj.is_absolute(), "Checkpoint path should be absolute"

    def test_configs_directory_exists(self):
        """Test that configs directory exists."""
        configs_dir = Path("configs")
        assert configs_dir.exists(), "configs/ directory should exist"
        assert configs_dir.is_dir(), "configs should be a directory"

    def test_src_directory_structure(self):
        """Test that src directory has expected structure."""
        src_dir = Path("src/ft_vlm")

        expected_subdirs = ["cli", "data", "settings", "infrastructure"]
        for subdir in expected_subdirs:
            subdir_path = src_dir / subdir
            assert subdir_path.exists(), f"Expected {subdir} directory to exist"


class TestConfigPaths:
    """Test configuration file paths."""

    def test_yaml_config_files_exist(self):
        """Test that YAML config files exist."""
        configs_dir = Path("configs")
        yaml_files = list(configs_dir.glob("*.yaml"))

        assert len(yaml_files) > 0, "No YAML config files found in configs/"

    def test_finetune_config_exists(self):
        """Test that finetune config exists."""
        config_path = Path("configs/finetune_lfm_3B.yaml")
        assert config_path.exists(), "finetune_lfm_3B.yaml should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
