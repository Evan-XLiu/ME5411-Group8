"""Task 3: crop the HD44780A00 text-line region of interest."""

from pathlib import Path

import numpy as np

from common.pipeline_io import labeled_panel, save_image, write_json


def crop_roi(image, roi):
    """Crop an ``(x, y, width, height)`` region after validating its bounds."""
    x, y, width, height = map(int, roi)
    image_height, image_width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {(x, y, width, height)} exceeds image size "
            f"{(image_width, image_height)}"
        )
    return image[y : y + height, x : x + width]


def run_task3(image, stretched, mean_5x5, roi, output_dir, target_text="HD44780A00"):
    output_dir = Path(output_dir)
    image_height, image_width = image.shape
    roi_original = crop_roi(image, roi)
    roi_stretched = crop_roi(stretched, roi)
    roi_smoothed = crop_roi(mean_5x5, roi)

    save_image(output_dir / "08_roi_original.png", roi_original)
    save_image(output_dir / "09_roi_contrast_stretch.png", roi_stretched)
    save_image(output_dir / "10_roi_mean_5x5.png", roi_smoothed)

    comparison = np.vstack(
        [
            labeled_panel(roi_original, f"Original ROI: {target_text}"),
            labeled_panel(roi_stretched, "Contrast-stretched ROI"),
            labeled_panel(roi_smoothed, "5x5 mean-filtered ROI"),
        ]
    )
    comparison_path = output_dir / "task3_roi_comparison.png"
    metadata_path = output_dir / "task3_roi_metadata.json"
    save_image(comparison_path, comparison)
    write_json(
        metadata_path,
        {
            "roi_xywh": list(roi),
            "source_image_size_wh": [image_width, image_height],
            "cropped_image_size_wh": [roi_original.shape[1], roi_original.shape[0]],
            "target_text": target_text,
        },
    )
    return {
        "original": roi_original,
        "stretched": roi_stretched,
        "smoothed": roi_smoothed,
        "comparison": comparison,
        "comparison_path": comparison_path,
        "metadata_path": metadata_path,
    }
