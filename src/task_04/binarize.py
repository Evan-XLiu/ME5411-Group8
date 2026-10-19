"""Task 4: compare global Otsu and adaptive Gaussian binarization."""

from pathlib import Path

import cv2
import numpy as np

from common.pipeline_io import labeled_panel, save_image, write_json


def otsu_binarization(image):
    """Return Otsu's threshold and a bright-foreground binary image."""
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / image.size
    levels = np.arange(256, dtype=np.float64)
    cumulative_probability = probabilities.cumsum()
    cumulative_mean = (probabilities * levels).cumsum()
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between_class_variance = np.divide(
        (total_mean * cumulative_probability - cumulative_mean) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    threshold = int(np.argmax(between_class_variance))
    binary = np.where(image > threshold, 255, 0).astype(np.uint8)
    return threshold, binary


def adaptive_binarization(image, block_size=31, c=-5.0):
    """Apply a local Gaussian threshold for uneven illumination."""
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("Adaptive-threshold block size must be odd and at least 3")
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def run_task4(roi_smoothed, output_dir, adaptive_block_size=31, adaptive_c=-5.0):
    output_dir = Path(output_dir)
    otsu_threshold, binary_otsu = otsu_binarization(roi_smoothed)
    binary_adaptive = adaptive_binarization(
        roi_smoothed, adaptive_block_size, adaptive_c
    )
    save_image(output_dir / "11_binary_otsu.png", binary_otsu)
    save_image(output_dir / "12_binary_adaptive.png", binary_adaptive)

    comparison = np.vstack(
        [
            labeled_panel(roi_smoothed, "5x5 mean-filtered ROI"),
            labeled_panel(binary_otsu, f"Otsu threshold = {otsu_threshold}"),
            labeled_panel(
                binary_adaptive,
                f"Adaptive Gaussian: block={adaptive_block_size}, C={adaptive_c:g}",
            ),
        ]
    )
    comparison_path = output_dir / "task4_binarization_comparison.png"
    metrics_path = output_dir / "task4_binarization_metrics.json"
    save_image(comparison_path, comparison)
    write_json(
        metrics_path,
        {
            "otsu": {
                "threshold": otsu_threshold,
                "foreground_fraction": round(float((binary_otsu > 0).mean()), 6),
            },
            "adaptive_gaussian": {
                "block_size": adaptive_block_size,
                "c": adaptive_c,
                "foreground_fraction": round(float((binary_adaptive > 0).mean()), 6),
            },
        },
    )
    return {
        "otsu_threshold": otsu_threshold,
        "binary_otsu": binary_otsu,
        "binary_adaptive": binary_adaptive,
        "comparison": comparison,
        "comparison_path": comparison_path,
        "metrics_path": metrics_path,
    }
