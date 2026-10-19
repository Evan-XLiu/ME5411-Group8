"""Task 4 - Sobel outline extraction and connected-component characters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.image_io import load_grayscale, save_grayscale
from common.image_ops import connected_components, crop_components, sobel_edges


def segment_characters(binary, connectivity: int = 8, min_area: int = 10, padding: int = 2):
    labels, components = connected_components(binary, connectivity, min_area)
    crops = crop_components(binary, components, padding)
    return labels, components, crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="binary or grayscale ROI")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge-threshold", type=float, default=80.0)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--padding", type=int, default=2)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_grayscale(args.input)
    _, _, _, edge_mask = sobel_edges(image, args.edge_threshold)
    save_grayscale(edge_mask * 255, output_dir / "outline.png")
    foreground = (image > 0).astype("uint8")
    labels, components, crops = segment_characters(foreground, args.connectivity, args.min_area, args.padding)
    # Label image is useful for diagnosing merged/split character errors.
    save_grayscale((labels > 0).astype("uint8") * 255, output_dir / "foreground.png")
    for index, crop in enumerate(crops, start=1):
        save_grayscale(crop, output_dir / f"char_{index:03d}.png")
    metadata = {"connectivity": args.connectivity, "min_area": args.min_area, "padding": args.padding, "edge_threshold": args.edge_threshold, "count": len(components), "components": components}
    (output_dir / "components.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
