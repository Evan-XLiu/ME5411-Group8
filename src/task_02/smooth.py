"""Task 2: compare averaging filters with different kernel sizes."""

from pathlib import Path

import cv2
import numpy as np

from common.pipeline_io import image_statistics, labeled_panel, save_image, write_json


class AveragingFilter:
    def __init__(self, size):
        if size < 1 or size % 2 == 0:
            raise ValueError("Averaging-filter size must be a positive odd integer")
        self.size = size
        self.kernel = np.ones((size, size), dtype=np.float32) / (size * size)

    def __call__(self, image):
        return cv2.filter2D(
            src=image,
            ddepth=-1,
            kernel=self.kernel,
            borderType=cv2.BORDER_REFLECT,
        )


def run_task2(stretched, output_dir):
    output_dir = Path(output_dir)
    mean_filtered = {size: AveragingFilter(size)(stretched) for size in (3, 5, 7)}
    output_numbers = {3: 5, 5: 6, 7: 7}
    for size, filtered_image in mean_filtered.items():
        save_image(
            output_dir / f"{output_numbers[size]:02d}_mean_{size}x{size}.png",
            filtered_image,
        )

    comparison = np.vstack(
        [
            np.hstack(
                [
                    labeled_panel(stretched, "Contrast-stretched input"),
                    labeled_panel(mean_filtered[3], "3x3 averaging filter"),
                ]
            ),
            np.hstack(
                [
                    labeled_panel(mean_filtered[5], "5x5 averaging filter"),
                    labeled_panel(mean_filtered[7], "7x7 averaging filter"),
                ]
            ),
        ]
    )
    comparison = cv2.resize(
        comparison, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )
    comparison_path = output_dir / "task2_mean_filter_comparison.png"
    metrics_path = output_dir / "task2_mean_filter_metrics.json"
    save_image(comparison_path, comparison)

    metrics = {}
    source_float = stretched.astype(np.float32)
    for name, filtered_image in {
        "unfiltered_contrast_stretch": stretched,
        **{f"mean_{size}x{size}": image for size, image in mean_filtered.items()},
    }.items():
        residual = source_float - filtered_image.astype(np.float32)
        metrics[name] = {
            **image_statistics(filtered_image),
            "laplacian_variance": round(
                float(cv2.Laplacian(filtered_image, cv2.CV_64F).var()), 3
            ),
            "residual_standard_deviation": round(float(residual.std()), 3),
        }
    write_json(metrics_path, metrics)
    return {
        "mean_filtered": mean_filtered,
        "comparison": comparison,
        "comparison_path": comparison_path,
        "metrics_path": metrics_path,
    }
