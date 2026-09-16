"""NumPy implementations of the image-processing operations used in Tasks 1-4.

The operations follow the lecture definitions: point-wise contrast transforms,
normalized averaging masks, thresholding, Sobel gradient magnitude, and
connected-component labelling with selectable 4/8 connectivity.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from .image_io import require_numpy


def _as_float(image):
    np = require_numpy()
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def _odd_size(size: int) -> int:
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError("filter/window size must be a positive odd integer")
    return size


def contrast_stretch(image, low: Optional[float] = None, high: Optional[float] = None):
    """Apply the piecewise-linear contrast stretch from the lectures."""
    np = require_numpy()
    array = _as_float(image)
    low = float(array.min()) if low is None else float(low)
    high = float(array.max()) if high is None else float(high)
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip(np.rint((array - low) * 255.0 / (high - low)), 0, 255).astype(np.uint8)


def histogram_equalization(image):
    """Global histogram equalization using a 256-bin cumulative histogram."""
    np = require_numpy()
    array = np.clip(_as_float(image), 0, 255).astype(np.uint8)
    hist = np.bincount(array.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = np.flatnonzero(hist)
    if nonzero.size < 2:
        return array.copy()
    cdf_min = cdf[nonzero[0]]
    scale = 255.0 / max(1, int(array.size - cdf_min))
    lut = np.clip(np.rint((cdf - cdf_min) * scale), 0, 255).astype(np.uint8)
    return lut[array]


def _pad(array, radius: int):
    np = require_numpy()
    return np.pad(array, radius, mode="reflect")


def mean_filter(image, size: int = 3):
    """Normalized square averaging mask (Eq. 3.2.3 generalised)."""
    np = require_numpy()
    size = _odd_size(size)
    radius = size // 2
    padded = _pad(_as_float(image), radius)
    out = np.zeros_like(_as_float(image), dtype=np.float32)
    for row in range(size):
        for col in range(size):
            out += padded[row : row + out.shape[0], col : col + out.shape[1]]
    return np.rint(out / float(size * size)).clip(0, 255).astype(np.uint8)


def gaussian_filter(image, size: int = 3, sigma: Optional[float] = None):
    """Apply a normalized discrete Gaussian mask."""
    np = require_numpy()
    size = _odd_size(size)
    sigma = float(sigma) if sigma is not None else max(0.5, size / 3.0)
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(axis**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    # Separable convolution avoids constructing a large 2-D mask.
    array = _as_float(image)
    padded = _pad(array, radius)
    horizontal = np.zeros_like(array)
    for col, weight in enumerate(kernel):
        horizontal += weight * padded[:, col : col + array.shape[1]]
    padded = _pad(horizontal, radius)
    out = np.zeros_like(array)
    for row, weight in enumerate(kernel):
        out += weight * padded[row : row + array.shape[0], :]
    return np.rint(out).clip(0, 255).astype(np.uint8)


def median_filter(image, size: int = 3):
    """Median filter, useful for impulse (salt-and-pepper) noise."""
    np = require_numpy()
    size = _odd_size(size)
    radius = size // 2
    array = _as_float(image)
    padded = _pad(array, radius)
    windows = []
    for row in range(size):
        for col in range(size):
            windows.append(padded[row : row + array.shape[0], col : col + array.shape[1]])
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.uint8)


def rotating_mean_filter(image, size: int = 3, footprint: int = 5):
    """Edge-preserving rotating-mask average described in the lectures.

    Among all ``size``-square windows inside a ``footprint``-square region,
    average the window with the smallest local variance.
    """
    np = require_numpy()
    size, footprint = _odd_size(size), _odd_size(footprint)
    if footprint < size:
        raise ValueError("footprint must be at least as large as size")
    array = _as_float(image)
    outer = footprint // 2
    padded = _pad(array, outer)
    out = np.empty_like(array)
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            patch = padded[r : r + footprint, c : c + footprint]
            windows = np.stack([patch[row : row + size, col : col + size] for row in range(footprint - size + 1) for col in range(footprint - size + 1)])
            out[r, c] = windows[np.argmin(windows.var(axis=(1, 2)))].mean()
    return np.rint(out).clip(0, 255).astype(np.uint8)


def otsu_threshold(image) -> tuple[int, object]:
    """Return Otsu's global threshold and a foreground mask (0/1)."""
    np = require_numpy()
    array = np.clip(_as_float(image), 0, 255).astype(np.uint8)
    hist = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / max(1, array.size)
    levels = np.arange(256, dtype=np.float64)
    cumulative = probabilities.cumsum()
    means = (probabilities * levels).cumsum()
    total_mean = means[-1]
    denominator = cumulative * (1.0 - cumulative)
    between = np.divide((total_mean * cumulative - means) ** 2, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    threshold = int(np.argmax(between))
    return threshold, (array > threshold).astype(np.uint8)


def adaptive_threshold(image, window: int = 15, offset: float = 5.0):
    """Local mean threshold: foreground is brighter than neighbourhood mean-offset."""
    np = require_numpy()
    local_mean = mean_filter(image, window).astype(np.float32)
    return (_as_float(image) > local_mean - float(offset)).astype(np.uint8)


def sobel_edges(image, threshold: Optional[float] = None):
    """Compute Sobel ``Gx``, ``Gy``, magnitude and an optional binary edge map."""
    np = require_numpy()
    array = _as_float(image)
    padded = _pad(array, 1)
    gx = (-padded[:-2, :-2] + padded[:-2, 2:] - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] - padded[2:, :-2] + padded[2:, 2:])
    gy = (-padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:] + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    magnitude = np.hypot(gx, gy)
    if threshold is None:
        return gx, gy, magnitude
    return gx, gy, magnitude, (magnitude >= float(threshold)).astype(np.uint8)


def connected_components(binary, connectivity: int = 8, min_area: int = 1):
    """Label foreground pixels and return ``(labels, components)``.

    Components are dictionaries containing label, area, bounding box, centroid
    and perimeter estimate. Bounding boxes use ``(x, y, width, height)``.
    """
    np = require_numpy()
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    mask = np.asarray(binary).astype(bool)
    if mask.ndim != 2:
        raise ValueError("binary image must be two-dimensional")
    labels = np.zeros(mask.shape, dtype=np.int32)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    components = []
    next_label = 1
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            queue = deque([(y, x)])
            labels[y, x] = next_label
            pixels = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = next_label
                        queue.append((ny, nx))
            if len(pixels) < int(min_area):
                labels[labels == next_label] = 0
                continue
            coords = np.asarray(pixels)
            ys, xs = coords[:, 0], coords[:, 1]
            component_mask = labels == next_label
            neighbors = np.zeros(component_mask.shape, dtype=bool)
            for dy, dx in offsets[:4]:
                shifted = np.zeros_like(component_mask)
                y0, y1 = max(0, dy), min(height, height + dy)
                x0, x1 = max(0, dx), min(width, width + dx)
                shifted[y0:y1, x0:x1] = component_mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
                neighbors |= shifted
            perimeter = int(np.logical_and(component_mask, ~neighbors).sum())
            components.append({"label": next_label, "area": int(len(pixels)), "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)], "centroid": [float(xs.mean()), float(ys.mean())], "perimeter": perimeter})
            next_label += 1
    return labels, components


def crop_components(binary, components: Iterable[dict], padding: int = 2):
    """Return padded binary crops sorted left-to-right for character recognition."""
    np = require_numpy()
    mask = (np.asarray(binary) > 0).astype(np.uint8)
    height, width = mask.shape
    crops = []
    for component in sorted(components, key=lambda item: item["bbox"][0]):
        x, y, w, h = component["bbox"]
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
        crops.append(mask[y0:y1, x0:x1] * 255)
    return crops
