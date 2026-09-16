# ME5411 Group 8

ME5411 Robot Vision and AI 小组项目仓库。

## 文件结构

```text
data/       数据集
src/        Task 1--7 的代码
results/    实验结果与模型
report/     IEEE LaTeX 报告
tests/      测试代码
```

## LaTeX 报告

- 主文件：`report/main.tex`
- 参考文献：`report/references.bib`
- 图片目录：`report/figures/`

先安装 MacTeX、TeX Live 或 MiKTeX，然后运行：

```bash
make pdf
```

生成的报告位于 `report/main.pdf`。清理编译文件：

```bash
make clean
```

使用 Overleaf 时，上传 `report/` 文件夹，选择 `main.tex` 作为主文件并点击 **Recompile**。

请勿提交原始数据集、模型权重或包含个人信息的文件。

## Task 1--6 pipeline

The implemented pipeline is in \`src/\`:

1. contrast stretching or histogram equalisation;
2. mean/Gaussian/median/rotating-mask smoothing;
3. ROI cropping and Otsu or local thresholding;
4. Sobel outlines and connected-component character crops;
5. a small PyTorch CNN;
6. HOG with an RBF-SVM (nearest-centroid fallback).

Install the dependencies in \`requirements.txt\`, then run the commands in
[\`src/README.md\`](src/README.md) with \`PYTHONPATH=src\`. The classifier data
must be arranged as one directory per character label, for example
\`data/processed/characters/A/*.png\`; raw course data and generated model files
remain ignored by Git.
