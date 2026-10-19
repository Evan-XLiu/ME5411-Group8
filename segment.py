"""Run the complete ME5411 Task 1-6 character-processing pipeline."""

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.pipeline_io import load_grayscale, save_image
from task_01.enhance import run_task1
from task_02.smooth import run_task2
from task_03.crop import run_task3
from task_04.binarize import run_task4
from task_05.outline import run_task5
from task_06.segment import run_task6


DEFAULT_INPUT = PROJECT_ROOT / "dataset_2026" / "Test_Image_2026.bmp"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "figures" / "01_original.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--show", action="store_true", help="display all task comparisons")
    args = parser.parse_args()

    image = load_grayscale(args.input)
    height, width = image.shape
    save_image(args.output, image)
    output_dir = args.output.parent

    task1 = run_task1(image, output_dir)
    task2 = run_task2(task1["stretched"], output_dir)
    task3 = run_task3(
        image,
        task1["stretched"],
        task2["mean_filtered"][5],
        args.roi,
        output_dir,
    )
    task4 = run_task4(
        task3["smoothed"],
        output_dir,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
    )
    task5 = run_task5(
        task3["smoothed"],
        task3["original"],
        task4["binary_otsu"],
        output_dir,
        min_noise_area=args.min_noise_area,
    )
    task6 = run_task6(
        task3["original"],
        task5["cleaned_binary"],
        task5["retained_areas"],
        output_dir,
        PROJECT_ROOT,
    )

    print(f"Loaded image: {args.input}")
    print(f"Image size: {width} x {height}")
    print(f"Saved grayscale original: {args.output}")
    print(f"Saved Task 1 comparison: {task1['comparison_path']}")
    print(f"Saved Task 1 metrics: {task1['metrics_path']}")
    print(f"Saved Task 2 comparison: {task2['comparison_path']}")
    print(f"Saved Task 2 metrics: {task2['metrics_path']}")
    print(f"Saved Task 3 ROI comparison: {task3['comparison_path']}")
    print(f"Saved Task 3 ROI metadata: {task3['metadata_path']}")
    print(f"Saved Task 4 binarization comparison: {task4['comparison_path']}")
    print(f"Saved Task 4 binarization metrics: {task4['metrics_path']}")
    print(f"Saved Task 5 outline comparison: {task5['comparison_path']}")
    print(f"Saved Task 5 outline metrics: {task5['metrics_path']}")
    print(f"Saved Task 6 labelled characters: {task6['labelled_overlay_path']}")
    print(f"Saved Task 6 character strip: {task6['character_strip_path']}")
    print(f"Saved Task 6 segmentation metadata: {task6['metadata_path']}")

    if args.show:
        cv2.imshow("ME5411 - Original image", image)
        cv2.imshow("ME5411 - Task 1 contrast comparison", task1["comparison"])
        cv2.imshow("ME5411 - Task 2 averaging-filter comparison", task2["comparison"])
        cv2.imshow("ME5411 - Task 3 HD44780A00 ROI", task3["comparison"])
        cv2.imshow("ME5411 - Task 4 binarization", task4["comparison"])
        cv2.imshow("ME5411 - Task 5 character outlines", task5["comparison"])
        cv2.imshow("ME5411 - Task 6 labelled characters", task6["labelled_overlay"])
        cv2.imshow("ME5411 - Task 6 character crops", task6["character_strip"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
