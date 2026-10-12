"""Task 5: clean the binary mask and extract character outlines."""

from pathlib import Path

import cv2
import numpy as np

from common.pipeline_io import labeled_panel, save_image, write_json
from task_04.binarize import otsu_binarization


def remove_small_components(binary, min_area=200):
    """Remove connected foreground regions smaller than ``min_area`` pixels."""
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(binary)
    retained = []
    for label in range(1, component_count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
            retained.append(area)
    return cleaned, component_count - 1, retained


def character_outlines(grayscale, cleaned_binary):
    """Generate Sobel and binary-boundary outlines around character regions."""
    gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    maximum = float(magnitude.max())
    normalized = (
        np.rint(magnitude * 255.0 / maximum).astype(np.uint8)
        if maximum > 0
        else np.zeros_like(grayscale)
    )
    sobel_threshold, sobel = otsu_binarization(normalized)
    character_neighbourhood = cv2.dilate(
        cleaned_binary, np.ones((5, 5), dtype=np.uint8), iterations=1
    )
    sobel = cv2.bitwise_and(sobel, character_neighbourhood)

    eroded = cv2.erode(
        cleaned_binary, np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    morphological = cv2.subtract(cleaned_binary, eroded)
    return sobel_threshold, sobel, morphological


def run_task5(roi_smoothed, roi_original, binary_otsu, output_dir, min_noise_area=200):
    output_dir = Path(output_dir)
    cleaned_binary, component_count_before, retained_areas = remove_small_components(
        binary_otsu, min_noise_area
    )
    sobel_threshold, sobel_outline, morphological_outline = character_outlines(
        roi_smoothed, cleaned_binary
    )
    contours, _ = cv2.findContours(
        cleaned_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_overlay = cv2.cvtColor(roi_original, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), thickness=2)

    save_image(output_dir / "13_binary_cleaned.png", cleaned_binary)
    save_image(output_dir / "14_outline_sobel.png", sobel_outline)
    save_image(output_dir / "15_outline_morphological.png", morphological_outline)
    save_image(output_dir / "16_contour_overlay.png", contour_overlay)

    comparison = np.vstack(
        [
            np.hstack(
                [
                    labeled_panel(cleaned_binary, "Cleaned Otsu binary"),
                    labeled_panel(sobel_outline, "Masked Sobel outline"),
                ]
            ),
            np.hstack(
                [
                    labeled_panel(morphological_outline, "Morphological outline"),
                    labeled_panel(contour_overlay, "Contours on original ROI"),
                ]
            ),
        ]
    )
    comparison = cv2.resize(
        comparison, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )
    comparison_path = output_dir / "task5_outline_comparison.png"
    metrics_path = output_dir / "task5_outline_metrics.json"
    save_image(comparison_path, comparison)
    write_json(
        metrics_path,
        {
            "minimum_component_area": min_noise_area,
            "components_before_cleanup": component_count_before,
            "components_retained": len(retained_areas),
            "retained_component_areas": retained_areas,
            "sobel_threshold": sobel_threshold,
            "contours_drawn_including_inner_boundaries": len(contours),
        },
    )
    return {
        "cleaned_binary": cleaned_binary,
        "retained_areas": retained_areas,
        "comparison": comparison,
        "comparison_path": comparison_path,
        "metrics_path": metrics_path,
    }
