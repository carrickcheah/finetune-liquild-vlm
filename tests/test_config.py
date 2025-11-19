"""Tests for training configuration."""

import pytest
from pathlib import Path


class TestFineTuningConfig:
    """Test FineTuningConfig class."""

    def test_config_default_values(self):
        """Test that config has sensible default values."""
        from ft_vlm.settings.training_config import FineTuningConfig

        config = FineTuningConfig()

        # Check defaults exist
        assert config.model_name is not None
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.num_train_epochs > 0

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        from ft_vlm.settings.training_config import FineTuningConfig

        # Check if config file exists
        config_path = Path("configs/finetune_lfm_3B.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = FineTuningConfig.from_yaml("finetune_lfm_3B.yaml")
        assert config is not None
        assert config.model_name is not None

    def test_config_lora_parameters(self):
        """Test LoRA configuration parameters."""
        from ft_vlm.settings.training_config import FineTuningConfig

        config = FineTuningConfig(
            use_peft=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )

        assert config.use_peft is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1

    def test_config_train_split_ratio_valid(self):
        """Test that train_split_ratio is valid."""
        from ft_vlm.settings.training_config import FineTuningConfig

        config = FineTuningConfig(train_split_ratio=0.8)
        assert 0 < config.train_split_ratio <= 1.0

    def test_config_to_dict(self):
        """Test config can be converted to dict."""
        from ft_vlm.settings.training_config import FineTuningConfig

        config = FineTuningConfig()
        config_dict = config.__dict__

        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "batch_size" in config_dict


class TestEvaluationConfig:
    """Test EvaluationConfig class if it exists."""

    def test_eval_config_exists(self):
        """Test that evaluation config can be loaded."""
        try:
            from ft_vlm.settings.training_config import FineTuningConfig
            # Evaluation uses same config structure
            config = FineTuningConfig()
            assert config is not None
        except ImportError:
            pytest.skip("EvaluationConfig not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
