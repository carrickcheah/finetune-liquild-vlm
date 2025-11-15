"""ft_vlm - Vision-Language Model Fine-tuning and Evaluation Framework."""

# Configuration
from ft_vlm.settings import (
    FineTuningConfig,
    EvaluationConfig,
    get_path_to_configs,
    get_path_to_evals,
)

# Core functionality
from ft_vlm.core import (
    get_model_output,
    get_structured_model_output,
)

# Data utilities
from ft_vlm.data import (
    load_dataset,
    load_model_and_processor,
    format_dataset_as_conversation,
    split_dataset,
)

# Evaluation
from ft_vlm.evaluation import EvalReport

# Model schemas
from ft_vlm.models import (
    ModelOutputType,
    get_model_output_schema,
)

__all__ = [
    # Configuration
    "FineTuningConfig",
    "EvaluationConfig",
    "get_path_to_configs",
    "get_path_to_evals",
    # Core
    "get_model_output",
    "get_structured_model_output",
    # Data
    "load_dataset",
    "load_model_and_processor",
    "format_dataset_as_conversation",
    "split_dataset",
    # Evaluation
    "EvalReport",
    # Models
    "ModelOutputType",
    "get_model_output_schema",
]
