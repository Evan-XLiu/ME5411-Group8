# ME5411 Group 8

ME5411 Robot Vision and AI 小组项目仓库。

## 文件结构

```text
data/       数据集（不会上传到 GitHub）
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
