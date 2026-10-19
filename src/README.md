# Source code

The source tree follows the project brief directly:

| Task | Module | Responsibility |
| --- | --- | --- |
| 1 | `task_01/enhance.py` | Contrast stretching, histogram equalisation, and mean-centred contrast |
| 2 | `task_02/smooth.py` | 3x3, 5x5, and 7x7 averaging filters |
| 3 | `task_03/crop.py` | Crop the HD44780A00 region of interest |
| 4 | `task_04/binarize.py` | Otsu and adaptive Gaussian binarisation |
| 5 | `task_05/outline.py` | Noise cleanup, Sobel/morphological outlines, and contours |
| 6 | `task_06/segment.py` | Connected-component and projection-based character segmentation |
| 7-1 | `task_07/cnn.py` | CNN character classifier |
| 7-2 | `task_07/classical.py` | HOG with an RBF-SVM or nearest-centroid fallback |

`common/pipeline_io.py` contains the shared Unicode-safe image I/O, JSON, and
comparison-panel helpers. The top-level `segment.py` is the single entry point
for Tasks 1-6 and calls these modules in order:

```powershell
python segment.py
python segment.py --show
```

Use `python segment.py --help` to change the input image, output location, ROI,
adaptive-threshold parameters, or minimum component area. The generated Task
1-6 figures and metadata are written under `results/figures/`.

Task 7 needs a larger labelled dataset arranged as one directory per character,
such as `data/processed/characters/0/*.png` and
`data/processed/characters/A/*.png`. The ten crops from the supplied test image
are useful for checking segmentation, but they are not enough to train or
evaluate a classifier reliably.
