"""Task 1: compare three contrast-enhancement methods."""

from pathlib import Path

import cv2
import numpy as np

from common.pipeline_io import image_statistics, labeled_panel, save_image, write_json


def contrast_stretch(image, low_percentile=1.0, high_percentile=99.0):
    """Stretch the central intensity range to 0-255 to suppress outliers."""
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high <= low:
        return image.copy()
    stretched = (image.astype(np.float32) - low) * 255.0 / (high - low)
    return np.clip(np.rint(stretched), 0, 255).astype(np.uint8)


def histogram_equalization(image):
    """Apply global histogram equalization using an explicit 256-bin CDF."""
    histogram = np.bincount(image.ravel(), minlength=256)
    cdf = histogram.cumsum()
    nonzero = np.flatnonzero(histogram)
    if nonzero.size < 2:
        return image.copy()
    cdf_min = cdf[nonzero[0]]
    denominator = image.size - cdf_min
    if denominator <= 0:
        return image.copy()
    lookup = np.clip(
        np.rint((cdf - cdf_min) * 255.0 / denominator), 0, 255
    ).astype(np.uint8)
    return lookup[image]


def centered_contrast(image, alpha=1.5, beta=0):
    """Scale intensities around the image mean."""
    mean_intensity = np.mean(image)
    enhanced = alpha * (image - mean_intensity) + mean_intensity + beta
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def run_task1(image, output_dir):
    output_dir = Path(output_dir)
    stretched = contrast_stretch(image)
    equalized = histogram_equalization(image)
    centered = centered_contrast(image, alpha=1.5)

    save_image(output_dir / "02_contrast_stretch.png", stretched)
    save_image(output_dir / "03_hist_equalized.png", equalized)
    save_image(output_dir / "04_centered_contrast.png", centered)

    top = np.hstack(
        [labeled_panel(image, "Original"), labeled_panel(stretched, "1%-99% contrast stretch")]
    )
    bottom = np.hstack(
        [
            labeled_panel(equalized, "Histogram equalization"),
            labeled_panel(centered, "Mean-centered alpha=1.5"),
        ]
    )
    comparison = cv2.resize(
        np.vstack([top, bottom]), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )
    comparison_path = output_dir / "task1_contrast_comparison.png"
    metrics_path = output_dir / "task1_contrast_metrics.json"
    save_image(comparison_path, comparison)
    write_json(
        metrics_path,
        {
            "original": image_statistics(image),
            "contrast_stretch_1_99_percentile": image_statistics(stretched),
            "histogram_equalization": image_statistics(equalized),
            "mean_centered_alpha_1_5": image_statistics(centered),
        },
    )
    return {
        "stretched": stretched,
        "equalized": equalized,
        "centered": centered,
        "comparison": comparison,
        "comparison_path": comparison_path,
        "metrics_path": metrics_path,
    }
