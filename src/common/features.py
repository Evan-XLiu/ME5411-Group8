"""Classical image features for the non-CNN baseline."""

from __future__ import annotations

from .image_io import require_numpy


def resize_nearest(image, shape: tuple[int, int] = (32, 32)):
    np = require_numpy()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("resize_nearest expects a grayscale image")
    out_h, out_w = map(int, shape)
    ys = np.minimum((np.arange(out_h) * array.shape[0] / out_h).astype(int), array.shape[0] - 1)
    xs = np.minimum((np.arange(out_w) * array.shape[1] / out_w).astype(int), array.shape[1] - 1)
    return array[ys[:, None], xs[None, :]]


def hog_descriptor(image, cell_size: int = 4, bins: int = 9):
    """Compute a compact unsigned-gradient HOG descriptor."""
    np = require_numpy()
    array = resize_nearest(image).astype(np.float32) / 255.0
    gy, gx = np.gradient(array)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0) / 180.0 * bins
    features = []
    for y in range(0, array.shape[0] - cell_size + 1, cell_size):
        for x in range(0, array.shape[1] - cell_size + 1, cell_size):
            hist = np.zeros(bins, dtype=np.float32)
            cell_mag = magnitude[y : y + cell_size, x : x + cell_size].ravel()
            cell_angle = angle[y : y + cell_size, x : x + cell_size].ravel()
            lower = np.floor(cell_angle).astype(int) % bins
            upper = (lower + 1) % bins
            fraction = cell_angle - np.floor(cell_angle)
            for index in range(cell_mag.size):
                hist[lower[index]] += cell_mag[index] * (1.0 - fraction[index])
                hist[upper[index]] += cell_mag[index] * fraction[index]
            features.extend(hist)
    vector = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-8)


def extract_features(images):
    np = require_numpy()
    return np.stack([hog_descriptor(image) for image in images]).astype(np.float32)
