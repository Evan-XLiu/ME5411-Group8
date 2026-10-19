# YJN integration

Imported from `origin/YJN_CODE` commit `ddfd865` without replacing the main
report, dataset or existing pipeline.

- `CNN_net.py`: YJN's three-block CNN with global average pooling.
- `CNN_train.py`: working entry point backed by the shared training pipeline.
- `CNN_train_original.py`: original contributor script, preserved for reference.
  It loads data from a different directory and does not call training/evaluation.
- Root `segment.py`: original unfinished OpenCV prototype, preserved unchanged.
  Use `src/task_01` through `src/task_04` for the working preprocessing pipeline.

Run from the repository root after installing `requirements.txt`:

```bash
PYTHONPATH=src python -m task_05.cnn --data dataset_2026 \
  --architecture baseline --model results/models/baseline.pt \
  --metrics results/models/baseline.json
PYTHONPATH=src python -m task_05.cnn --data dataset_2026 \
  --architecture yjn --model results/models/yjn.pt \
  --metrics results/models/yjn.json
PYTHONPATH=src python -m task_06.classical train --data dataset_2026 \
  --model results/models/hog_svm.pkl --metrics results/models/hog_svm.json
```

Alternatively run `python CNN/CNN_train.py --data dataset_2026 --model
results/models/yjn.pt` for YJN. Both CNNs resize to 32x32 and use the same
stratified 75/25 split, seed 42, batch size 32 and Adam learning rate 0.001.
CUDA is selected when available; use `--device cpu` to force CPU execution.
The test split is evaluated each epoch for diagnostics; do not select
hyperparameters on this test set. Use a training-only validation split for tuning.

The baseline retains 4x4 spatial features and a 64-unit hidden classifier;
YJN pools to 1x1 and uses a smaller classification head. Baseline has the more
complete original training interface; YJN has fewer parameters. Neither is
established as more accurate without controlled experiments. Task 7-3 still
requires measured accuracy/runtime comparisons, preprocessing and hyperparameter
experiments, and interpretation by the group.

## Initial real-data check (2026-10-19)

Three CPU epochs, one thread, seed 42, the same 5,334 training and 1,778 test
images, no augmentation, 32x32 nearest-neighbour resizing:

| Model | Parameters | Final test accuracy | Training and evaluation time |
| --- | ---: | ---: | ---: |
| baseline | 89,447 | 97.81% | 4.96 s |
| YJN | 23,975 | 93.53% | 5.09 s |

Recorded metrics: `comparison_smoke.json`. These are integration checks, not
a tuned or repeated benchmark. Baseline converged faster in this short run;
YJN uses about 73% fewer parameters. Keep baseline as the default and retain
YJN for controlled longer experiments. Dataset loading is excluded from timing.
