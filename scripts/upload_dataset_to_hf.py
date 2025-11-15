"""
Script to copy CIFAR-100 dataset to your HuggingFace account.

Usage:
    python scripts/upload_dataset_to_hf.py --repo-name <your-username>/cifar100

Requirements:
    pip install huggingface_hub datasets
"""

import argparse
from datasets import load_dataset
from huggingface_hub import HfApi


def upload_dataset_to_hf(source_dataset: str, target_repo: str):
    """
    Copy a dataset from HuggingFace to your own account.

    Args:
        source_dataset: Source dataset name (e.g., 'uoft-cs/cifar100')
        target_repo: Target repository name (e.g., 'your-username/cifar100')
    """
    print(f"Loading dataset from {source_dataset}...")
    dataset = load_dataset(source_dataset)

    print(f"\nDataset loaded successfully!")
    print(f"Splits: {list(dataset.keys())}")
    print(f"Train samples: {len(dataset['train'])}")
    if 'test' in dataset:
        print(f"Test samples: {len(dataset['test'])}")

    print(f"\nUploading dataset to {target_repo}...")
    dataset.push_to_hub(target_repo, private=False)

    print(f"\n✅ Dataset successfully uploaded to: https://huggingface.co/datasets/{target_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy CIFAR-100 dataset to your HuggingFace account"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="uoft-cs/cifar100",
        help="Source dataset name (default: uoft-cs/cifar100)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Target repository name (e.g., your-username/cifar100)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )

    args = parser.parse_args()

    # Load the dataset
    print(f"Loading dataset from {args.source}...")
    dataset = load_dataset(args.source)

    print(f"\nDataset loaded successfully!")
    print(f"Splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"{split} samples: {len(dataset[split])}")

    # Upload to HuggingFace
    print(f"\nUploading dataset to {args.repo_name}...")
    print("Note: You need to be logged in to HuggingFace (run: huggingface-cli login)")

    dataset.push_to_hub(args.repo_name, private=args.private)

    print(f"\n✅ Dataset successfully uploaded!")
    print(f"View at: https://huggingface.co/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()
