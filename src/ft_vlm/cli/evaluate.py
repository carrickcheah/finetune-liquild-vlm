"""
Evaluates a VL model on a given dataset

Steps:
1. Download the dataset
2. Load the model (supports both HuggingFace Hub and local checkpoint paths)
3. Loop through the dataset and compute model outputs
4. Compute accuracy as a binary score: 1 if the model output matches the ground truth, 0 otherwise
"""

from tqdm import tqdm

from ft_vlm.settings.training_config import EvaluationConfig
from ft_vlm.core.inference import get_model_output, get_structured_model_output
from ft_vlm.data.loaders import load_dataset, load_model_and_processor
from ft_vlm.infrastructure.modal import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
    get_model_cache_volume,
)
from ft_vlm.evaluation.report import EvalReport  # , save_predictions_to_disk
from ft_vlm.models.output_types import ModelOutputType, get_model_output_schema

volume = get_volume("cifar100-identification")
app = get_modal_app("cifar100-identification")
image = get_docker_image()
model_cache_volume = get_model_cache_volume()


@app.function(
    image=image,
    gpu="L40S",
    # gpu="H100",
    volumes={
        # "/datasets": volume,
        "/model_checkpoints": volume,
        "/model_cache": model_cache_volume,
    },
    secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def evaluate(
    config: EvaluationConfig,
) -> EvalReport:
    """
    Runs a model evaluation on a given dataset using Modal serverless GPU

    Args:
        config: The configuration for the evaluation

    Returns:
        EvalReport: The evaluation report
    """
    print(f"Starting evaluation of {config.model} on {config.dataset}")

    # Debug: List available checkpoints if model path is local
    if config.model.startswith("/model_checkpoints"):
        import os
        print("\n📂 Available checkpoints in /model_checkpoints:")
        try:
            for item in os.listdir("/model_checkpoints"):
                print(f"  - {item}")
                # List contents of each checkpoint directory
                checkpoint_path = os.path.join("/model_checkpoints", item)
                if os.path.isdir(checkpoint_path):
                    try:
                        contents = sorted(os.listdir(checkpoint_path))
                        # Find all checkpoint directories
                        checkpoints = [c for c in contents if c.startswith("checkpoint-")]
                        if checkpoints:
                            print(f"    Checkpoints: {checkpoints}")
                        else:
                            print(f"    Contents: {contents[:10]}")
                    except Exception as e:
                        print(f"    Error listing contents: {e}")
        except Exception as e:
            print(f"  Error listing checkpoints: {e}")
        print()

    dataset = load_dataset(
        dataset_name=config.dataset,
        splits=[config.split],
        n_samples=config.n_samples,
        seed=config.seed,
    )

    model, processor = load_model_and_processor(
        model_id=config.model, base_model_id=config.base_model
    )

    # Prepare evaluation report
    eval_report = EvalReport()

    # Naive evaluation loop without batching
    accurate_predictions: int = 0
    for sample in tqdm(dataset):
        # Extracts sample image and normalized label
        image = sample[config.image_column]

        if config.label_mapping is not None:
            label = config.label_mapping[sample[config.label_column]]
        else:
            label = sample[config.label_column]

        # create the conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": config.user_prompt},
                ],
            },
        ]

        if config.structured_generation:
            # Using JSON structured output
            model_output: ModelOutputType | None = get_structured_model_output(
                model,
                processor,
                config.system_prompt,
                config.user_prompt,
                image,
                output_schema=get_model_output_schema(config.dataset),
            )

            if model_output is None:
                continue

            # Extract th predicted class from the structured output
            pred_class = model_output.pred_class

        else:
            # Using raw model output without structured generation
            pred_class: str = get_model_output(model, processor, conversation)


        # Compare predicton vs ground truth.
        accurate_predictions += 1 if pred_class == label else 0

        # Add record to evaluation report
        eval_report.add_record(image, label, pred_class)

    print(f"Accuracy: {eval_report.get_accuracy():.2f}")

    print("✅ Evaluation completed successfully")

    return eval_report


@app.local_entrypoint()
def main(
    config_file_name: str,
):
    """
    Evaluates a VL model on a given dataset using Modal serverless GPU
    acceleration and stores an evaluation report in the evals/ directory.

    Args:
        config_file_name: The name of the configuration file to use
    """
    config = EvaluationConfig.from_yaml(config_file_name)

    eval_report = evaluate.remote(config)

    output_path = eval_report.to_csv()
    print(f"Predictions saved to {output_path}")


