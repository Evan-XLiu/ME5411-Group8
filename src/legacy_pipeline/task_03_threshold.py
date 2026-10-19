"""Task 3 - crop the text ROI and create a binary foreground mask.

An explicit ``x,y,width,height`` ROI is preferred for a fixed camera setup.
Otsu's threshold is the reproducible global fallback; local mean thresholding
is useful when illumination varies across the chip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

from common.image_io import load_grayscale, save_grayscale
from common.image_ops import adaptive_threshold, contrast_stretch, otsu_threshold


def crop_roi(image, roi: Optional[Tuple[int, int, int, int]] = None):
    if roi is None:
        return image
    x, y, width, height = roi
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    image_height, image_width = image.shape
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"ROI {(x, y, width, height)} is outside image {(image_width, image_height)}")
    return image[y : y + height, x : x + width]


def binarize(image, method: str = "otsu", window: int = 15, offset: float = 5.0, foreground: str = "bright"):
    if method == "otsu":
        threshold, mask = otsu_threshold(image)
        if foreground == "dark":
            mask = 1 - mask
        return threshold, mask
    if method == "adaptive":
        mask = adaptive_threshold(image, window, offset)
        if foreground == "dark":
            mask = 1 - mask
        return None, mask
    raise ValueError("method must be 'otsu' or 'adaptive'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="binary PNG (foreground=255)")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--method", choices=("otsu", "adaptive"), default="otsu")
    parser.add_argument("--foreground", choices=("bright", "dark"), default="bright", help="whether printed characters are brighter or darker than the ROI")
    parser.add_argument("--window", type=int, default=15, help="odd local window for adaptive threshold")
    parser.add_argument("--offset", type=float, default=5.0)
    parser.add_argument("--metadata", help="optional JSON metadata output")
    args = parser.parse_args()
    image = crop_roi(load_grayscale(args.input), tuple(args.roi) if args.roi else None)
    image = contrast_stretch(image)
    threshold, mask = binarize(image, args.method, args.window, args.offset, args.foreground)
    save_grayscale(mask * 255, args.output)
    if args.metadata:
        Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata).write_text(json.dumps({"method": args.method, "threshold": threshold, "foreground": args.foreground, "roi": args.roi, "foreground_fraction": float(mask.mean())}, indent=2) + "\n")


if __name__ == "__main__":
    main()
