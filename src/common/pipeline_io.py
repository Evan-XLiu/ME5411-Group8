"""Shared image and report helpers for the Task 1-6 pipeline."""

import json
from pathlib import Path

import cv2
import numpy as np


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


def write_json(path, payload):
    """Write consistently formatted UTF-8 JSON output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def image_statistics(image):
    return {
        "minimum": int(image.min()),
        "maximum": int(image.max()),
        "mean": round(float(image.mean()), 3),
        "standard_deviation": round(float(image.std()), 3),
    }


def labeled_panel(image, label):
    """Create a report-ready panel with a readable method label."""
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
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
