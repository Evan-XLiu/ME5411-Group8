# Source code

The first six tasks form the complete character-recognition pipeline described
by the ME5411 lectures and the report template:

| Task | Implementation | Main lecture concept |
| --- | --- | --- |
| 1 | `task_01/enhance.py` | Piecewise-linear contrast stretch and histogram equalisation |
| 2 | `task_02/smooth.py` | Mean/Gaussian/median and rotating-mask smoothing |
| 3 | `task_03/threshold.py` | ROI cropping, Otsu global threshold, local threshold |
| 4 | `task_04/segment.py` | Sobel outline, connected components, character crops |
| 5 | `task_05/cnn.py` | Supervised CNN classifier with a deterministic split |
| 6 | `task_06/classical.py` | HOG + RBF-SVM, with nearest-centroid fallback |

Reusable implementations are in `common/`. Commands are run from the project
root with `PYTHONPATH=src`, for example:

```bash
python -m task_01.enhance --input data/raw/chip.png \
  --output results/figures/enhanced.png --metrics results/figures/enhanced.json
python -m task_02.smooth --input results/figures/enhanced.png \
  --output results/figures/smoothed.png --method gaussian --size 3
python -m task_03.threshold --input results/figures/smoothed.png \
  --output results/figures/binary.png --method otsu \
  --metadata results/figures/threshold.json
python -m task_04.segment --input results/figures/binary.png \
  --output-dir data/processed/characters
python -m task_05.cnn --data data/processed/characters \
  --model results/models/character_cnn.pt --metrics results/models/cnn.json
python -m task_06.classical train --data data/processed/characters \
  --model results/models/hog_svm.pkl --metrics results/models/classical.json
```

The classifier dataset uses one directory per character label, such as
`characters/0/*.png` and `characters/A/*.png`. Task 4 emits unlabeled crops;
they must be assigned to class directories using the project ground truth
before training. All random splits use seed 42 by default.
