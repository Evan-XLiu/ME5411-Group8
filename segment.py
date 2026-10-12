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
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (0, 0), (360, 38), (0, 0, 0), thickness=-1)
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
    parser.add_argument("--show", action="store_true", help="display the original image in a window")
    args = parser.parse_args()

    image = load_grayscale(args.input)
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

    height, width = image.shape
    print(f"Loaded image: {args.input}")
    print(f"Image size: {width} x {height}")
    print(f"Saved grayscale original: {args.output}")
    print(f"Saved Task 1 comparison: {comparison_path}")
    print(f"Saved Task 1 metrics: {metrics_path}")
    print(f"Saved Task 2 comparison: {filter_comparison_path}")
    print(f"Saved Task 2 metrics: {filter_metrics_path}")

    if args.show:
        cv2.imshow("ME5411 - Task 1 contrast comparison", comparison)
        cv2.imshow("ME5411 - Task 2 averaging-filter comparison", filter_comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
