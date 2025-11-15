"""Core business logic for training, evaluation, and inference."""

from ft_vlm.core.inference import (
    get_model_output,
    get_structured_model_output,
)

__all__ = [
    "get_model_output",
    "get_structured_model_output",
]
