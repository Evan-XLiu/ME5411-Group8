import cv2
import numpy as np

# 1. 以灰度形式读取图片
image = cv2.imread("Test_Image_2026.bmp", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("无法读取 Test_Image_2026.bmp，请检查文件路径")

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
