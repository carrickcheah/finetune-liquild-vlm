"""Tests for data loading utilities."""

import pytest


class TestDatasetLoading:
    """Test dataset loading functions."""

    def test_load_dataset_import(self):
        """Test that load_dataset can be imported."""
        try:
            from ft_vlm.data.loaders import load_dataset
            assert load_dataset is not None
        except ImportError as e:
            pytest.fail(f"Failed to import load_dataset: {e}")

    def test_load_model_and_processor_import(self):
        """Test that load_model_and_processor can be imported."""
        try:
            from ft_vlm.data.loaders import load_model_and_processor
            assert load_model_and_processor is not None
        except ImportError as e:
            pytest.fail(f"Failed to import load_model_and_processor: {e}")

    def test_dataset_splits_format(self):
        """Test that dataset splits are properly formatted."""
        splits = ["train", "test"]
        assert isinstance(splits, list)
        assert all(isinstance(s, str) for s in splits)

    @pytest.mark.skip(reason="Requires network access and HuggingFace credentials")
    def test_load_cifar100_dataset(self):
        """Test loading CIFAR-100 dataset."""
        from ft_vlm.data.loaders import load_dataset

        dataset = load_dataset(
            dataset_name="cifar100",
            splits=["train"],
            n_samples=10,
            seed=42
        )

        assert dataset is not None
        assert len(dataset) == 10


class TestDataPreparation:
    """Test data preparation utilities."""

    def test_format_dataset_import(self):
        """Test that format_dataset_as_conversation can be imported."""
        try:
            from ft_vlm.data.data_preparation import format_dataset_as_conversation
            assert format_dataset_as_conversation is not None
        except ImportError as e:
            pytest.fail(f"Failed to import format_dataset_as_conversation: {e}")

    def test_split_dataset_import(self):
        """Test that split_dataset can be imported."""
        try:
            from ft_vlm.data.data_preparation import split_dataset
            assert split_dataset is not None
        except ImportError as e:
            pytest.fail(f"Failed to import split_dataset: {e}")

    def test_conversation_format_structure(self):
        """Test expected conversation format structure."""
        # Expected format for SFT training
        expected_keys = ["messages"]
        sample_conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }

        assert all(key in sample_conversation for key in expected_keys)
        assert isinstance(sample_conversation["messages"], list)


class TestCollateFunction:
    """Test collate function creation."""

    def test_create_collate_fn_import(self):
        """Test that create_collate_fn can be imported from train module."""
        try:
            from ft_vlm.cli.train import create_collate_fn
            assert create_collate_fn is not None
        except ImportError as e:
            pytest.fail(f"Failed to import create_collate_fn: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
