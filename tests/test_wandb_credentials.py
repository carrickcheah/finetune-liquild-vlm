"""Tests for Weights & Biases credentials and authentication."""

import os
import pytest


class TestWandbCredentials:
    """Test W&B credential validation and authentication."""

    def test_wandb_api_key_format(self):
        """Test that WANDB_API_KEY has valid format if set."""
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            # W&B API keys are typically 40 characters
            assert len(api_key) >= 32, "WANDB_API_KEY appears too short"
            assert api_key.isalnum(), "WANDB_API_KEY should be alphanumeric"

    def test_wandb_api_key_exists_in_env(self):
        """Test that WANDB_API_KEY environment variable exists."""
        api_key = os.environ.get("WANDB_API_KEY")
        # This test is informational - key may not be set locally
        if not api_key:
            pytest.skip("WANDB_API_KEY not set in environment (expected in Modal)")

    def test_wandb_import(self):
        """Test that wandb can be imported."""
        try:
            import wandb
            assert wandb is not None
        except ImportError:
            pytest.fail("wandb package is not installed")

    def test_wandb_api_connection(self):
        """Test W&B API connection (requires valid credentials)."""
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            pytest.skip("WANDB_API_KEY not set - skipping connection test")

        import wandb
        try:
            wandb.login(key=api_key)
            api = wandb.Api()
            # Simple API call to verify connection
            user = api.viewer
            assert user is not None, "Failed to get W&B user info"
        except Exception as e:
            pytest.fail(f"W&B API connection failed: {e}")

    def test_wandb_entity_project_config(self):
        """Test that W&B entity and project can be configured."""
        from ft_vlm.settings.training_config import FineTuningConfig

        config = FineTuningConfig(
            wandb_entity="test-entity",
            wandb_project_name="test-project",
            wandb_experiment_name="test-experiment"
        )

        assert config.wandb_entity == "test-entity"
        assert config.wandb_project_name == "test-project"
        assert config.wandb_experiment_name == "test-experiment"

    def test_wandb_disabled_mode(self):
        """Test that W&B can be disabled via config."""
        from ft_vlm.settings.training_config import FineTuningConfig

        config = FineTuningConfig(use_wandb=False)
        assert config.use_wandb is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
