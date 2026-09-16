"""Small, dependency-light image I/O helpers.

Images are represented as NumPy arrays with grayscale values in ``uint8``
range. Pillow is intentionally imported lazily so the processing modules can
still be imported in documentation or test environments without Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("This task requires numpy. Install requirements.txt first.") from exc
    return np


def load_grayscale(path: Union[str, Path]):
    """Load *path* as a two-dimensional uint8 grayscale array."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Image loading requires Pillow. Install requirements.txt first.") from exc
    np = require_numpy()
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def save_grayscale(image, path: Union[str, Path]) -> None:
    """Save a grayscale array, creating its parent directory if necessary."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Image saving requires Pillow. Install requirements.txt first.") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = require_numpy().clip(image, 0, 255).astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def normalize_uint8(image):
    """Convert any numeric image to a finite uint8 image without changing shape."""
    np = require_numpy()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint((array - lo) * 255.0 / (hi - lo)).astype(np.uint8)
