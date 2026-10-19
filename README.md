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

## Task 1--6 image-processing pipeline

The top-level `segment.py` runs the implemented stages in order:

1. contrast enhancement;
2. averaging-filter comparison;
3. HD44780A00 ROI cropping;
4. global and adaptive binarisation;
5. outline extraction and noise cleanup;
6. individual-character segmentation.

Install the dependencies in `requirements.txt`, then run:

```powershell
python segment.py
```

Add `--show` to display each comparison window. Figures, metrics, and the ten
character crops are saved under `results/figures/`. Each stage is implemented
in the matching `src/task_01` through `src/task_06` directory; see
[`src/README.md`](src/README.md) for the detailed mapping.

Task 7 contains the CNN and non-CNN classifiers. It requires a larger labelled
character dataset before meaningful training and evaluation can begin.
