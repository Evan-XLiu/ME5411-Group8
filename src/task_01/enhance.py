"""Task 1 - point-wise enhancement for low-contrast chip text.

Examples
--------
``python -m task_01.enhance --input data/raw/chip.png --output results/figures/enhanced.png``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from common.image_io import load_grayscale, save_grayscale
from common.image_ops import contrast_stretch, histogram_equalization


def enhance(image, method: str = "stretch", low: Optional[float] = None, high: Optional[float] = None):
    if method == "stretch":
        return contrast_stretch(image, low, high)
    if method == "equalize":
        return histogram_equalization(image)
    raise ValueError("method must be 'stretch' or 'equalize'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input image")
    parser.add_argument("--output", required=True, help="enhanced grayscale image")
    parser.add_argument("--method", choices=("stretch", "equalize"), default="stretch")
    parser.add_argument("--low", type=float, help="lower input grey level; default is image minimum")
    parser.add_argument("--high", type=float, help="upper input grey level; default is image maximum")
    parser.add_argument("--metrics", help="optional JSON output for reproducibility")
    args = parser.parse_args()
    image = load_grayscale(args.input)
    output = enhance(image, args.method, args.low, args.high)
    save_grayscale(output, args.output)
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps({"method": args.method, "input_min": int(image.min()), "input_max": int(image.max()), "output_min": int(output.min()), "output_max": int(output.max())}, indent=2) + "\n")


if __name__ == "__main__":
    main()
