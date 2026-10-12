"""Task 6: split the cleaned text line into individual characters."""

from pathlib import Path

import cv2
import numpy as np

from common.pipeline_io import labeled_panel, save_image, write_json


def _foreground_bbox(binary, x_start, x_end):
    """Return a tight foreground box inside the half-open x interval."""
    region = binary[:, x_start:x_end] > 0
    rows, columns = np.nonzero(region)
    if rows.size == 0:
        return None
    x0 = x_start + int(columns.min())
    x1 = x_start + int(columns.max()) + 1
    y0 = int(rows.min())
    y1 = int(rows.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def segment_character_boxes(cleaned_binary, expected_count=10):
    """Split wide connected regions at central vertical-projection valleys."""
    count, _, statistics, _ = cv2.connectedComponentsWithStats(
        (cleaned_binary > 0).astype(np.uint8), connectivity=8
    )
    components = []
    for label in range(1, count):
        x, y, width, height, area = map(int, statistics[label])
        components.append(
            {"x": x, "y": y, "width": width, "height": height, "area": area}
        )
    components.sort(key=lambda item: item["x"])
    if not components:
        raise ValueError("No foreground components are available for character segmentation")

    component_widths = np.asarray(
        [item["width"] for item in components], dtype=np.float32
    )
    typical_width = float(np.median(component_widths))
    projection = (cleaned_binary > 0).sum(axis=0)
    boxes = []
    split_columns = []

    for component in components:
        x = component["x"]
        width = component["width"]
        estimated_characters = max(1, int(round(width / typical_width)))
        if estimated_characters == 1:
            box = _foreground_bbox(cleaned_binary, x, x + width)
            if box is not None:
                boxes.append(box)
            continue

        boundaries = [x]
        for part in range(1, estimated_characters):
            expected = x + width * part / estimated_characters
            half_window = max(2, int(round(width * 0.10)))
            search_start = max(x + 1, int(round(expected)) - half_window)
            search_end = min(x + width - 1, int(round(expected)) + half_window + 1)
            candidates = np.arange(search_start, search_end)
            cut = int(candidates[np.argmin(projection[candidates])])
            boundaries.append(cut)
            split_columns.append(cut)
        boundaries.append(x + width)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            box = _foreground_bbox(cleaned_binary, start, end)
            if box is not None:
                boxes.append(box)

    boxes.sort(key=lambda item: item[0])
    if len(boxes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} character boxes, obtained {len(boxes)}; "
            "adjust preprocessing or the ROI"
        )
    return boxes, typical_width, split_columns


def crop_with_padding(image, box, padding=6):
    x, y, width, height = box
    image_height, image_width = image.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image_width, x + width + padding)
    y1 = min(image_height, y + height + padding)
    return image[y0:y1, x0:x1]


def normalize_character_crop(crop, canvas_size=96, margin=8):
    """Fit a binary crop onto a square canvas without changing its aspect ratio."""
    height, width = crop.shape
    available = canvas_size - 2 * margin
    scale = min(available / max(1, width), available / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x0 = (canvas_size - resized_width) // 2
    y0 = (canvas_size - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def run_task6(
    roi_original,
    cleaned_binary,
    retained_areas,
    output_dir,
    project_root,
    target_text="HD44780A00",
):
    output_dir = Path(output_dir)
    project_root = Path(project_root)
    character_boxes, typical_character_width, split_columns = segment_character_boxes(
        cleaned_binary, expected_count=len(target_text)
    )
    character_dir = output_dir / "characters"
    character_dir.mkdir(parents=True, exist_ok=True)
    labelled_overlay = cv2.cvtColor(roi_original, cv2.COLOR_GRAY2BGR)
    strip_panels = []
    character_metadata = []
    colours = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for index, (character, box) in enumerate(
        zip(target_text, character_boxes), start=1
    ):
        x, y, box_width, box_height = box
        crop = crop_with_padding(cleaned_binary, box)
        crop_path = character_dir / f"{index:02d}_{character}.png"
        save_image(crop_path, crop)
        strip_panels.append(
            labeled_panel(normalize_character_crop(crop), f"{index}: {character}")
        )
        colour = colours[(index - 1) % len(colours)]
        cv2.rectangle(
            labelled_overlay,
            (x, y),
            (x + box_width - 1, y + box_height - 1),
            colour,
            thickness=2,
        )
        cv2.putText(
            labelled_overlay,
            f"{index}:{character}",
            (x, max(16, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
            cv2.LINE_AA,
        )
        character_metadata.append(
            {
                "index": index,
                "label": character,
                "bbox_xywh": list(box),
                "crop_file": str(crop_path.relative_to(project_root)),
            }
        )

    labelled_overlay_path = output_dir / "17_labeled_characters.png"
    character_strip_path = output_dir / "task6_character_strip.png"
    metadata_path = output_dir / "task6_segmentation_metadata.json"
    save_image(labelled_overlay_path, labelled_overlay)
    character_strip = np.hstack(strip_panels)
    save_image(character_strip_path, character_strip)
    write_json(
        metadata_path,
        {
            "target_text": target_text,
            "detected_character_count": len(character_boxes),
            "connected_components_before_splitting": len(retained_areas),
            "typical_component_width": round(typical_character_width, 3),
            "projection_split_columns": split_columns,
            "characters": character_metadata,
        },
    )
    return {
        "labelled_overlay": labelled_overlay,
        "character_strip": character_strip,
        "labelled_overlay_path": labelled_overlay_path,
        "character_strip_path": character_strip_path,
        "metadata_path": metadata_path,
    }
