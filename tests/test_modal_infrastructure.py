"""Tests for Modal infrastructure setup."""

import pytest


class TestModalSetup:
    """Test Modal infrastructure functions."""

    def test_get_modal_app_import(self):
        """Test that get_modal_app can be imported."""
        try:
            from ft_vlm.infrastructure.modal import get_modal_app
            assert get_modal_app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import get_modal_app: {e}")

    def test_get_docker_image_import(self):
        """Test that get_docker_image can be imported."""
        try:
            from ft_vlm.infrastructure.modal import get_docker_image
            assert get_docker_image is not None
        except ImportError as e:
            pytest.fail(f"Failed to import get_docker_image: {e}")

    def test_get_volume_import(self):
        """Test that get_volume can be imported."""
        try:
            from ft_vlm.infrastructure.modal import get_volume
            assert get_volume is not None
        except ImportError as e:
            pytest.fail(f"Failed to import get_volume: {e}")

    def test_get_secrets_import(self):
        """Test that get_secrets can be imported."""
        try:
            from ft_vlm.infrastructure.modal import get_secrets
            assert get_secrets is not None
        except ImportError as e:
            pytest.fail(f"Failed to import get_secrets: {e}")

    def test_get_retries_import(self):
        """Test that get_retries can be imported."""
        try:
            from ft_vlm.infrastructure.modal import get_retries
            assert get_retries is not None
        except ImportError as e:
            pytest.fail(f"Failed to import get_retries: {e}")

    def test_modal_app_creation(self):
        """Test that Modal app can be created."""
        from ft_vlm.infrastructure.modal import get_modal_app

        app = get_modal_app("test-app")
        assert app is not None

    def test_docker_image_creation(self):
        """Test that Docker image can be created."""
        from ft_vlm.infrastructure.modal import get_docker_image

        image = get_docker_image()
        assert image is not None

    def test_volume_creation(self):
        """Test that volume can be created."""
        from ft_vlm.infrastructure.modal import get_volume

        volume = get_volume("test-volume")
        assert volume is not None

    def test_secrets_list(self):
        """Test that secrets list can be retrieved."""
        from ft_vlm.infrastructure.modal import get_secrets

        secrets = get_secrets()
        assert secrets is not None
        assert isinstance(secrets, list)

    def test_retries_config(self):
        """Test that retries config can be created."""
        from ft_vlm.infrastructure.modal import get_retries

        retries = get_retries(max_retries=3)
        assert retries is not None


class TestModalIntegration:
    """Integration tests for Modal (requires Modal account)."""

    @pytest.mark.skip(reason="Requires Modal authentication")
    def test_modal_secret_exists(self):
        """Test that wandb-secret exists in Modal."""
        import modal

        # This would require Modal authentication
        secret = modal.Secret.from_name("wandb-secret")
        assert secret is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
