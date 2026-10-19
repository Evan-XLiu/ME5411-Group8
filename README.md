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

`report/` 是唯一维护的报告主稿，已合入 Wanghongchao 的方法与实验配置扩写。
原 `report_from_Wanghongchao/` 已合并移除，历史版本可在 Git 记录中查看。
报告采用课程 Task 1--7 编号，Python 模块仍沿用 `task_01`--`task_06`；映射见报告 Introduction。
结果、讨论和结论仍需用实际实验补齐。

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

## YJN_CODE 集成

已保留 YJN 的 CNN 和原始分割原型，运行方式与比较见 [CNN/README.md](CNN/README.md)。
统一训练入口支持 `--architecture baseline` 和 `--architecture yjn`，
CNN 和 HOG/SVM 默认均采用课程要求的 75/25 分层划分。
Task 7-1 和 7-2 的训练代码已具备；Task 7-3 的实验比较与报告仍待完成。
