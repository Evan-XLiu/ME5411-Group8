import argparse
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

# 2. 定义一个均值滤波器类
class AveragingFilter:
    def __init__(self, size):
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

    height, width = image.shape
    print(f"Loaded image: {args.input}")
    print(f"Image size: {width} x {height}")
    print(f"Saved grayscale original: {args.output}")

    if args.show:
        cv2.imshow("ME5411 - Original grayscale image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
