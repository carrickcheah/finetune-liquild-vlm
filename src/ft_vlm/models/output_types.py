from typing import Literal, TypeVar

from pydantic import BaseModel

ModelOutputType = TypeVar("ModelOutputType", bound=BaseModel)


class CIFAR100OutputType(BaseModel):
    pred_class: Literal[
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
        "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
        "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
        "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
        "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    ]

    @classmethod
    def from_pred_class(cls, pred_class: str) -> str:
        """Create instance from pred_class and return as JSON string."""
        instance = cls(pred_class=pred_class)
        return instance.model_dump_json()


def get_model_output_schema(dataset_name: str) -> BaseModel:
    if "cifar100" in dataset_name:
        return CIFAR100OutputType
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
