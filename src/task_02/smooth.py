"""Task 2 - smoothing with averaging, Gaussian, median, or rotating masks."""

from __future__ import annotations

import argparse
from typing import Optional

from common.image_io import load_grayscale, save_grayscale
from common.image_ops import gaussian_filter, mean_filter, median_filter, rotating_mean_filter


def smooth(image, method: str = "mean", size: int = 3, sigma: Optional[float] = None):
    methods = {"mean": mean_filter, "gaussian": gaussian_filter, "median": median_filter, "rotating": rotating_mean_filter}
    if method not in methods:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(methods)}")
    if method == "gaussian":
        return methods[method](image, size, sigma)
    return methods[method](image, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("mean", "gaussian", "median", "rotating"), default="mean")
    parser.add_argument("--size", type=int, default=3, help="odd mask size")
    parser.add_argument("--sigma", type=float, help="Gaussian sigma; defaults to size/3")
    args = parser.parse_args()
    save_grayscale(smooth(load_grayscale(args.input), args.method, args.size, args.sigma), args.output)


if __name__ == "__main__":
    main()
