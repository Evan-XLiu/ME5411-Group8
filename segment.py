import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "Test_Image_2026.bmp"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "figures" / "01_original.png"


def load_grayscale(path):
    """Read a grayscale image, including paths containing non-ASCII text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"无法读取图片：{path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return image


def save_image(path, image):
    """Save an image to a path that may contain non-ASCII text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"无法编码图片：{path}")
    encoded.tofile(path)


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


def image_statistics(image):
    return {
        "minimum": int(image.min()),
        "maximum": int(image.max()),
        "mean": round(float(image.mean()), 3),
        "standard_deviation": round(float(image.std()), 3),
    }


def labeled_panel(image, label):
    """Create a report-ready panel with a readable method label."""
    panel = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image.copy()
    )
    panel = cv2.copyMakeBorder(
        panel, 38, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cv2.putText(
        panel,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


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

# 2. 定义一个均值滤波器类
class AveragingFilter:
    def __init__(self, size):
        if size < 1 or size % 2 == 0:
            raise ValueError("Averaging-filter size must be a positive odd integer")
        self.size = size
        self.kernel = np.ones((size, size), dtype=np.float32) / (size * size)

    def __call__(self, image):
        return cv2.filter2D(src=image, ddepth=-1, kernel=self.kernel, borderType=cv2.BORDER_REFLECT)
# 3. 对比度增强类
class ContrastEnhancement:
    def __init__(self, alpha=1.5, beta=0):
        self.alpha = alpha  # 对比度控制
        self.beta = beta    # 亮度控制

    def __call__(self, image):
        return cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
# 4. 平均灰度对比度增强类
class CenterContrastEnhancement:
    def __init__(self, alpha=1.5, beta=0):
        self.alpha = alpha  # 对比度控制
        self.beta = beta    # 亮度控制

    def __call__(self, image):
        mean_intensity = np.mean(image)
        enhanced_image = self.alpha*(image - mean_intensity) + mean_intensity + self.beta
        return np.clip(enhanced_image, 0, 255).astype(np.uint8)
# 5. 二值化处理类
class Binarization:
    def __init__(self, threshold=127):
        self.threshold = threshold

    def __call__(self, image):
        _, binary_image = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
        return binary_image
# 6.图形分割


def main():
    parser = argparse.ArgumentParser(description="ME5411 character-processing pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="input chip image")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="saved grayscale image")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        default=(45, 180, 930, 180),
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="HD44780A00 text-line region",
    )
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=31,
        help="odd neighbourhood size for adaptive thresholding",
    )
    parser.add_argument(
        "--adaptive-c",
        type=float,
        default=-5.0,
        help="constant subtracted from the adaptive local threshold",
    )
    parser.add_argument(
        "--min-noise-area",
        type=int,
        default=200,
        help="minimum connected-component area retained before outline extraction",
    )
    parser.add_argument("--show", action="store_true", help="display the original image in a window")
    args = parser.parse_args()

    image = load_grayscale(args.input)
    height, width = image.shape
    save_image(args.output, image)

    output_dir = args.output.parent
    stretched = contrast_stretch(image)
    equalized = histogram_equalization(image)
    centered = CenterContrastEnhancement(alpha=1.5)(image)

    stretch_path = output_dir / "02_contrast_stretch.png"
    equalized_path = output_dir / "03_hist_equalized.png"
    centered_path = output_dir / "04_centered_contrast.png"
    comparison_path = output_dir / "task1_contrast_comparison.png"
    metrics_path = output_dir / "task1_contrast_metrics.json"

    save_image(stretch_path, stretched)
    save_image(equalized_path, equalized)
    save_image(centered_path, centered)

    top = np.hstack(
        [labeled_panel(image, "Original"), labeled_panel(stretched, "1%-99% contrast stretch")]
    )
    bottom = np.hstack(
        [labeled_panel(equalized, "Histogram equalization"), labeled_panel(centered, "Mean-centered alpha=1.5")]
    )
    comparison = np.vstack([top, bottom])
    comparison = cv2.resize(comparison, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    save_image(comparison_path, comparison)

    metrics = {
        "original": image_statistics(image),
        "contrast_stretch_1_99_percentile": image_statistics(stretched),
        "histogram_equalization": image_statistics(equalized),
        "mean_centered_alpha_1_5": image_statistics(centered),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    mean_filtered = {
        size: AveragingFilter(size)(stretched)
        for size in (3, 5, 7)
    }
    output_numbers = {3: 5, 5: 6, 7: 7}
    for size, filtered_image in mean_filtered.items():
        save_image(
            output_dir / f"{output_numbers[size]:02d}_mean_{size}x{size}.png",
            filtered_image,
        )

    filter_comparison = np.vstack(
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
    filter_comparison = cv2.resize(
        filter_comparison, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )
    filter_comparison_path = output_dir / "task2_mean_filter_comparison.png"
    save_image(filter_comparison_path, filter_comparison)

    filter_metrics = {}
    source_float = stretched.astype(np.float32)
    for name, filtered_image in {
        "unfiltered_contrast_stretch": stretched,
        **{f"mean_{size}x{size}": image for size, image in mean_filtered.items()},
    }.items():
        residual = source_float - filtered_image.astype(np.float32)
        filter_metrics[name] = {
            **image_statistics(filtered_image),
            "laplacian_variance": round(
                float(cv2.Laplacian(filtered_image, cv2.CV_64F).var()), 3
            ),
            "residual_standard_deviation": round(float(residual.std()), 3),
        }
    filter_metrics_path = output_dir / "task2_mean_filter_metrics.json"
    filter_metrics_path.write_text(
        json.dumps(filter_metrics, indent=2) + "\n", encoding="utf-8"
    )

    roi_original = crop_roi(image, args.roi)
    roi_stretched = crop_roi(stretched, args.roi)
    roi_smoothed = crop_roi(mean_filtered[5], args.roi)
    roi_original_path = output_dir / "08_roi_original.png"
    roi_stretched_path = output_dir / "09_roi_contrast_stretch.png"
    roi_smoothed_path = output_dir / "10_roi_mean_5x5.png"
    save_image(roi_original_path, roi_original)
    save_image(roi_stretched_path, roi_stretched)
    save_image(roi_smoothed_path, roi_smoothed)

    roi_comparison = np.vstack(
        [
            labeled_panel(roi_original, "Original ROI: HD44780A00"),
            labeled_panel(roi_stretched, "Contrast-stretched ROI"),
            labeled_panel(roi_smoothed, "5x5 mean-filtered ROI"),
        ]
    )
    roi_comparison_path = output_dir / "task3_roi_comparison.png"
    save_image(roi_comparison_path, roi_comparison)
    roi_metadata_path = output_dir / "task3_roi_metadata.json"
    roi_metadata_path.write_text(
        json.dumps(
            {
                "roi_xywh": list(args.roi),
                "source_image_size_wh": [width, height],
                "cropped_image_size_wh": [roi_original.shape[1], roi_original.shape[0]],
                "target_text": "HD44780A00",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    otsu_threshold, binary_otsu = otsu_binarization(roi_smoothed)
    binary_adaptive = adaptive_binarization(
        roi_smoothed, args.adaptive_block_size, args.adaptive_c
    )
    binary_otsu_path = output_dir / "11_binary_otsu.png"
    binary_adaptive_path = output_dir / "12_binary_adaptive.png"
    save_image(binary_otsu_path, binary_otsu)
    save_image(binary_adaptive_path, binary_adaptive)

    binary_comparison = np.vstack(
        [
            labeled_panel(roi_smoothed, "5x5 mean-filtered ROI"),
            labeled_panel(binary_otsu, f"Otsu threshold = {otsu_threshold}"),
            labeled_panel(
                binary_adaptive,
                f"Adaptive Gaussian: block={args.adaptive_block_size}, C={args.adaptive_c:g}",
            ),
        ]
    )
    binary_comparison_path = output_dir / "task4_binarization_comparison.png"
    save_image(binary_comparison_path, binary_comparison)
    binary_metrics_path = output_dir / "task4_binarization_metrics.json"
    binary_metrics_path.write_text(
        json.dumps(
            {
                "otsu": {
                    "threshold": otsu_threshold,
                    "foreground_fraction": round(float((binary_otsu > 0).mean()), 6),
                },
                "adaptive_gaussian": {
                    "block_size": args.adaptive_block_size,
                    "c": args.adaptive_c,
                    "foreground_fraction": round(
                        float((binary_adaptive > 0).mean()), 6
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cleaned_binary, component_count_before, retained_areas = remove_small_components(
        binary_otsu, args.min_noise_area
    )
    sobel_threshold, sobel_outline, morphological_outline = character_outlines(
        roi_smoothed, cleaned_binary
    )
    contours, _ = cv2.findContours(
        cleaned_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_overlay = cv2.cvtColor(roi_original, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), thickness=2)

    cleaned_binary_path = output_dir / "13_binary_cleaned.png"
    sobel_outline_path = output_dir / "14_outline_sobel.png"
    morphological_outline_path = output_dir / "15_outline_morphological.png"
    contour_overlay_path = output_dir / "16_contour_overlay.png"
    save_image(cleaned_binary_path, cleaned_binary)
    save_image(sobel_outline_path, sobel_outline)
    save_image(morphological_outline_path, morphological_outline)
    save_image(contour_overlay_path, contour_overlay)

    outline_comparison = np.vstack(
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
    outline_comparison = cv2.resize(
        outline_comparison, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )
    outline_comparison_path = output_dir / "task5_outline_comparison.png"
    save_image(outline_comparison_path, outline_comparison)
    outline_metrics_path = output_dir / "task5_outline_metrics.json"
    outline_metrics_path.write_text(
        json.dumps(
            {
                "minimum_component_area": args.min_noise_area,
                "components_before_cleanup": component_count_before,
                "components_retained": len(retained_areas),
                "retained_component_areas": retained_areas,
                "sobel_threshold": sobel_threshold,
                "contours_drawn_including_inner_boundaries": len(contours),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Loaded image: {args.input}")
    print(f"Image size: {width} x {height}")
    print(f"Saved grayscale original: {args.output}")
    print(f"Saved Task 1 comparison: {comparison_path}")
    print(f"Saved Task 1 metrics: {metrics_path}")
    print(f"Saved Task 2 comparison: {filter_comparison_path}")
    print(f"Saved Task 2 metrics: {filter_metrics_path}")
    print(f"Saved Task 3 ROI comparison: {roi_comparison_path}")
    print(f"Saved Task 3 ROI metadata: {roi_metadata_path}")
    print(f"Saved Task 4 binarization comparison: {binary_comparison_path}")
    print(f"Saved Task 4 binarization metrics: {binary_metrics_path}")
    print(f"Saved Task 5 outline comparison: {outline_comparison_path}")
    print(f"Saved Task 5 outline metrics: {outline_metrics_path}")

    if args.show:
        cv2.imshow("ME5411 - Task 1 contrast comparison", comparison)
        cv2.imshow("ME5411 - Task 2 averaging-filter comparison", filter_comparison)
        cv2.imshow("ME5411 - Task 3 HD44780A00 ROI", roi_comparison)
        cv2.imshow("ME5411 - Task 4 binarization", binary_comparison)
        cv2.imshow("ME5411 - Task 5 character outlines", outline_comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
