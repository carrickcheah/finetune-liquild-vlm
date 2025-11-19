import os
import random
import numpy as np
import datasets
from datasets import concatenate_datasets, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import login
import torch


def load_dataset(
    dataset_name: str,
    splits: list[str],
    n_samples: int | None = None,
    seed: int | None = 42,
) -> datasets.Dataset:
    """
    Loads a dataset from the Hugging Face dataset hub.
    """
    dataset_list: list[Dataset] = []
    for split in splits:
        print(f"📚 Loading dataset {dataset_name}, split={split}...")
        dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)
        dataset_list.append(dataset)

    # Concatenate datasets
    if len(dataset_list) >= 1:
        dataset = concatenate_datasets(dataset_list)
    else:
        raise Exception("No splits provided to load the dataset.")

    # print(f"📚 Loading dataset {dataset_name}, split={split}...")
    # dataset = datasets.load_dataset(dataset_name, split=split, num_proc=1)

    # Shuffle the dataset
    print(f"Shuffling dataset with seed {seed}...")
    dataset = dataset.shuffle(seed=seed)
    # Shuffle the dataset with generator for reproducibility
    # if seed is not None:
    #     generator = torch.Generator().manual_seed(seed)
    #     dataset = dataset.shuffle(seed=seed, generator=generator)
    # else:
    #     dataset = dataset.shuffle()

    # Select a subset of the dataset
    if n_samples is not None:
        n_samples = min(n_samples, dataset.num_rows)
        dataset = dataset.select(range(n_samples))

    print(f"Dataset {dataset_name} loaded successfully: {dataset.num_rows} rows")

    return dataset


def fix_model_type_in_config_json(model_id: str):
    """Fix config.json by replacing 'lfm2-vl' model_type with 'lfm2_vl'."""
    import json
    from pathlib import Path

    config_path = Path(model_id) / "config.json"

    # Check if model_id is a local path
    with open(config_path, "r") as f:
        config = json.load(f)

    # Fix the model_type if needed
    if config.get("model_type") == "lfm2-vl":
        print(f"Fixing config.json for model {model_id}...")
        config["model_type"] = "lfm2_vl"

        # Write back the fixed config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("config.json fixed successfully!")


def load_model_and_processor(
    model_id: str,
    base_model_id: str | None = None,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Loads a model and processor from the Hugging Face model hub or local path.

    Args:
        model_id: Model identifier (HF hub path or local path)
        base_model_id: If provided and model_id is a PEFT checkpoint, loads base model first then adapter
    """
    # Check if model_id is a local path
    is_local = model_id.startswith("/") or model_id.startswith("./")

    # Check if this is a PEFT checkpoint (local path without config.json)
    is_peft_checkpoint = False
    if is_local and base_model_id is not None:
        is_peft_checkpoint = True
        print(f"📦 Detected PEFT checkpoint: {model_id}")
        print(f"📦 Will load base model: {base_model_id}")

    # Login using HF_TOKEN from environment variables (only for HF hub models)
    if not is_local or is_peft_checkpoint:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("🔐 Logging in to Hugging Face Hub...")
            login(token=hf_token)
        else:
            print("⚠️ No HF_TOKEN found in environment variables")
    else:
        hf_token = None

    # Load processor from base model if PEFT, otherwise from model_id
    processor_source = base_model_id if is_peft_checkpoint else model_id

    # TODO: hack hack hack
    if not is_peft_checkpoint:
        try:
            fix_model_type_in_config_json(model_id)
        except Exception as e:
            print(f"Warning: could not fix config.json for model {model_id}: {e}")

    print(f"📚 Loading processor from {processor_source}...")
    processor = AutoProcessor.from_pretrained(
        processor_source,
        # trust_remote_code=True,
        max_image_tokens=256,
        token=hf_token,
        local_files_only=False if is_peft_checkpoint else is_local,
    )

    # Load model
    if is_peft_checkpoint:
        # Load base model first
        print(f"🧠 Loading base model from {base_model_id}...")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            dtype="bfloat16",
            device_map="auto",
            token=hf_token,
        )

        # Load PEFT adapter
        print(f"🔌 Loading PEFT adapter from {model_id}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_id)
        print("✅ PEFT adapter loaded successfully!")
    else:
        print(f"🧠 Loading model from {model_id}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="bfloat16",
            # trust_remote_code=True,
            device_map="auto",
            token=hf_token if not is_local else None,
            local_files_only=is_local,
        )

    print("\n✅ Model loaded successfully!")
    print(f"📖 Vocab size: {len(processor.tokenizer)}")
    # print(
    #     f"🖼️ Image processed in up to {processor.max_tiles} patches of size {processor.tile_size}"
    # )
    print(f"🔢 Parameters: {model.num_parameters():,}")
    print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

    return model, processor
