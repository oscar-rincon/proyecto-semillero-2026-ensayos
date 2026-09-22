"""Utilities for generating, validating, saving, and displaying binary images."""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ImageSize = Tuple[int, int]


def generate_random_images(
    count: int,
    size: ImageSize = (64, 64),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a batch of binary images with shape ``(count, height, width)``."""
    if count < 1:
        raise ValueError("count must be greater than zero")
    if len(size) != 2 or any(dimension < 1 for dimension in size):
        raise ValueError("size must contain two positive dimensions")

    generator = rng if rng is not None else np.random.default_rng()
    return generator.integers(
        low=0,
        high=2,
        size=(count, *size),
        dtype=np.uint8,
    )


def count_unique_images(images: np.ndarray) -> int:
    """Return the number of distinct images in a batch."""
    if images.ndim != 3:
        raise ValueError("images must have shape (count, height, width)")

    flattened_images = images.reshape(images.shape[0], -1)
    return np.unique(flattened_images, axis=0).shape[0]


def save_image_dataset(
    images: np.ndarray,
    output_path: str | Path,
    seed: int,
) -> Path:
    """Save images and their generation metadata to a compressed NumPy archive."""
    if images.ndim != 3:
        raise ValueError("images must have shape (count, height, width)")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=images,
        seed=np.array(seed),
        image_size=np.array(images.shape[1:], dtype=np.int64),
    )
    return path


def load_image_dataset(input_path: str | Path) -> tuple[np.ndarray, int, ImageSize]:
    """Load images and metadata from a compressed NumPy archive."""
    with np.load(input_path) as data:
        images = data["images"]
        seed = int(data["seed"])
        image_size = tuple(int(value) for value in data["image_size"])

    return images, seed, image_size


def plot_image(image: np.ndarray, title: str) -> None:
    """Display one binary image using the project plotting conventions."""
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.show()
