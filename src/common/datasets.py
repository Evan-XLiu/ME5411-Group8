"""Dataset discovery and deterministic train/test splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .image_io import load_grayscale, require_numpy

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_classification_dataset(root: Union[str, Path]):
    root = Path(root)
    samples, labels, class_names = [], [], []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            continue
        class_index = len(class_names)
        class_names.append(class_dir.name)
        for path in files:
            samples.append(load_grayscale(path))
            labels.append(class_index)
    if not samples:
        raise FileNotFoundError(f"No class directories with images found under {root}")
    return samples, require_numpy().asarray(labels, dtype="int64"), class_names


def stratified_split(labels, test_fraction: float = 0.25, seed: int = 42):
    np = require_numpy()
    labels = np.asarray(labels)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * test_fraction))) if len(indices) > 1 else 0
        test.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    if not train or not test:
        raise ValueError("Dataset is too small for the requested split")
    return np.asarray(sorted(train)), np.asarray(sorted(test))
