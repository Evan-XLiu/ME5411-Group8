# ME5411 Group 8 Project

这是 ME5411 Robot Vision and AI 小组项目的代码与报告仓库。仓库目前只提供项目骨架、Task 1--7 的工作目录和 IEEE 风格 LaTeX 报告模板；具体算法与实验内容将在后续逐步补充。

## 仓库结构

```text
ME5411-Group8/
├── data/
│   ├── raw/                 # 原始数据，只保存在本地
│   └── processed/           # 预处理后的数据，只保存在本地
├── report/
│   ├── figures/             # 报告使用的图片
│   ├── main.tex             # IEEEtran 报告主文件
│   └── references.bib       # BibTeX 参考文献库
├── results/
│   ├── figures/             # 实验生成的图表
│   └── models/              # 模型权重与检查点
├── src/
│   ├── common/              # 多个任务共用的代码
│   ├── task_01/
│   ├── task_02/
│   ├── task_03/
│   ├── task_04/
│   ├── task_05/
│   ├── task_06/
│   └── task_07/
├── tests/                   # 自动化测试
├── .gitignore
├── Makefile
└── README.md
```

`data/` 和 `results/` 下的大文件默认不会提交到 Git。请不要把课程数据集、训练好的模型或包含个人信息的文件直接提交到远程仓库。

## LaTeX 快速开始

报告入口是 `report/main.tex`，文档类使用 `IEEEtran` 的 conference 模式。最简单的编译方式是在仓库根目录执行：

```bash
make pdf
```

成功后会生成：

```text
report/main.pdf
```

如果希望保存后自动重新编译：

```bash
make watch
```

停止自动编译可按 `Ctrl+C`。清理 LaTeX 中间文件：

```bash
make clean
```

连同生成的 PDF 一起清理：

```bash
make distclean
```

## 1. 安装 LaTeX 环境

### macOS

推荐安装 MacTeX。完整版本体积较大，但包含 `IEEEtran`、BibTeX 和 `latexmk` 等常用工具。安装完成后重新打开终端，并检查：

```bash
pdflatex --version
latexmk --version
kpsewhich IEEEtran.cls
```

最后一条命令应输出 `IEEEtran.cls` 的路径。如果没有输出，说明 IEEEtran 尚未安装。

### Windows

可安装 MiKTeX 或 TeX Live。使用 MiKTeX 时，建议在设置中允许自动安装缺少的宏包。安装后在 PowerShell 中检查：

```powershell
pdflatex --version
latexmk --version
kpsewhich IEEEtran.cls
```

Windows 如果没有 `make`，可以直接进入 `report` 目录运行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Ubuntu / Debian

可以安装 TeX Live、IEEEtran 和 latexmk：

```bash
sudo apt update
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-publishers latexmk
```

然后运行：

```bash
kpsewhich IEEEtran.cls
```

确认 IEEEtran 已经可用。

## 2. 使用 Overleaf

不想配置本地环境时，可以使用 Overleaf：

1. 将整个仓库压缩成 ZIP，或只压缩 `report/` 目录。
2. 在 Overleaf 新建项目并选择 **Upload Project**。
3. 上传 ZIP。
4. 在项目设置中将主文档设为 `report/main.tex`；如果只上传了 `report/`，则设为 `main.tex`。
5. 编译器选择 pdfLaTeX。
6. 点击 **Recompile**。

从 Overleaf 下载内容覆盖本地文件前，请先提交或备份本地修改，避免丢失组员的工作。

## 3. 编辑标题与作者

打开 `report/main.tex`，修改以下区域：

```latex
\title{Microchip Character Recognition Using Image Processing and Machine Learning}

\author{
  \IEEEauthorblockN{Group 8}
  \IEEEauthorblockA{
    ME5411 Robot Vision and AI\\
    National University of Singapore\\
    Singapore
  }
}
```

需要列出组员姓名和学号时，可以把作者部分改成：

```latex
\author{
  \IEEEauthorblockN{Student A (A0123456X), Student B (A0234567Y)}
  \IEEEauthorblockA{Group 8, ME5411 Robot Vision and AI\\
  National University of Singapore}
}
```

不要在 GitHub 等公开仓库中暴露真实学号；最终提交版本再补齐个人信息更安全。

## 4. 编写正文

一级、二级和三级标题分别使用：

```latex
\section{Methodology}
\subsection{Image Pre-processing}
\subsubsection{Contrast Enhancement}
```

普通文字直接写在标题下。新段落之间留一个空行，不需要手动添加缩进或换行命令。

LaTeX 会自动管理编号。使用标签和引用时，建议统一添加类型前缀：

```latex
\section{Experiments}\label{sec:experiments}

The evaluation protocol is described in Section~\ref{sec:experiments}.
```

常用前缀包括 `sec:`、`fig:`、`tab:` 和 `eq:`。

## 5. 插入图片

先把图片放到 `report/figures/`，优先使用 PDF、PNG 或 JPG。流程图和曲线图优先导出为矢量 PDF，截图或照片使用 PNG/JPG。

单栏图片示例：

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/pipeline.pdf}
  \caption{Overview of the proposed character-recognition pipeline.}
  \label{fig:pipeline}
\end{figure}
```

在正文中引用：

```latex
As shown in Fig.~\ref{fig:pipeline}, the input image is processed in stages.
```

跨越 IEEE 双栏的宽图使用：

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/all_results.pdf}
  \caption{Intermediate results of the complete processing pipeline.}
  \label{fig:all-results}
\end{figure*}
```

图片文件名建议只使用小写字母、数字、连字符或下划线，不要使用空格和中文字符。

## 6. 插入表格

```latex
\begin{table}[t]
  \caption{Classification Performance}
  \label{tab:performance}
  \centering
  \begin{tabular}{lcc}
    \toprule
    Method & Accuracy (\%) & Time (ms) \\
    \midrule
    CNN & 00.0 & 00.0 \\
    Non-CNN & 00.0 & 00.0 \\
    \bottomrule
  \end{tabular}
\end{table}
```

IEEE 表格的标题通常位于表格上方，图片标题位于图片下方。尽量不要使用竖线；`booktabs` 提供的 `\toprule`、`\midrule` 和 `\bottomrule` 通常更清晰。

## 7. 编写公式

行内公式使用 `$...$`：

```latex
The binary threshold is denoted by $T$.
```

带编号的独立公式使用：

```latex
\begin{equation}
  g(x,y) =
  \begin{cases}
    1, & f(x,y) \geq T,\\
    0, & f(x,y) < T.
  \end{cases}
  \label{eq:threshold}
\end{equation}
```

引用公式：

```latex
The binary image is calculated using \eqref{eq:threshold}.
```

## 8. 管理参考文献

所有参考文献条目保存在 `report/references.bib`。例如：

```bibtex
@book{gonzalez2018digital,
  author    = {Rafael C. Gonzalez and Richard E. Woods},
  title     = {Digital Image Processing},
  edition   = {4},
  publisher = {Pearson},
  year      = {2018}
}
```

正文中使用：

```latex
Classical image-processing methods are described in \cite{gonzalez2018digital}.
```

`main.tex` 末尾已经包含：

```latex
\bibliographystyle{IEEEtran}
\bibliography{references}
```

运行 `make pdf` 时，`latexmk` 会自动判断是否需要执行 BibTeX。如果手动编译，完整顺序是：

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

需要多次运行 pdfLaTeX，是因为交叉引用、编号和参考文献需要额外轮次才能稳定。

## 9. 使用 VS Code 编辑

可以安装 VS Code 的 **LaTeX Workshop** 扩展，然后打开整个仓库文件夹。常用操作包括：

- 从命令面板运行 `LaTeX Workshop: Build LaTeX project`。
- 使用 `LaTeX Workshop: View LaTeX PDF` 预览报告。
- 从 PDF 反向跳转到源码，或从源码定位到 PDF。

如果扩展找不到主文件，请直接打开 `report/main.tex` 后再次编译，或在工作区设置中把它指定为 root file。

## 10. 常见错误与排查

### `IEEEtran.cls not found`

IEEEtran 没有安装。先运行：

```bash
kpsewhich IEEEtran.cls
```

若无输出，请安装 MacTeX/TeX Live 的 publishers collection，或让 MiKTeX 自动安装缺少的宏包。

### `File 'figures/xxx' not found`

检查图片是否放在 `report/figures/`、文件名大小写是否一致，以及编译入口是否为 `report/main.tex`。

### 引用显示为 `??`

重新运行 `make pdf`。手动编译时必须执行完整的 pdfLaTeX、BibTeX、pdfLaTeX、pdfLaTeX 流程。

### 参考文献显示为 `[?]`

检查 `\cite{...}` 中的键是否与 `references.bib` 完全一致；然后查看 `report/main.blg` 中的 BibTeX 错误。

### `Unicode character ... not set up for use with LaTeX`

当前模板使用 pdfLaTeX，报告正文建议使用英文。若必须输入特殊 Unicode 字符，请使用对应的 LaTeX 写法，或在团队统一确认后切换到 XeLaTeX；不要让不同组员分别使用不同编译器。

### 图片或表格越过栏宽

单栏内容宽度不要超过 `\columnwidth`。图片可以使用：

```latex
\includegraphics[width=\columnwidth]{...}
```

表格过宽时，优先缩短列名或拆分表格，不建议把字体缩得难以阅读。

### 编译缓存导致结果异常

执行：

```bash
make clean
make pdf
```

仍然失败时，阅读终端输出中出现的第一条以 `!` 开头的错误；后面的报错常常只是连锁反应。

## 11. 推荐协作流程

开始工作前同步远程更新：

```bash
git pull --rebase
```

每个人尽量在独立分支工作：

```bash
git switch -c task-01-contrast-enhancement
```

提交前检查：

```bash
git status
make pdf
```

然后提交：

```bash
git add src report README.md
git commit -m "Add contrast enhancement experiment"
```

不要提交原始数据、模型权重、LaTeX 中间文件或包含真实学号的公开草稿。合并前至少确认报告可以从干净环境成功编译，并检查 PDF 中是否存在溢出、模糊图片、错误引用和未替换的占位文字。

## 12. 报告提交前检查

- 补齐 Group 8 所有成员姓名和学号。
- 确认标题、摘要和关键词准确。
- 所有图、表、公式均有编号并在正文中引用。
- 每个处理阶段都有清晰截图和解释。
- 实验设置、数据划分、指标和随机种子写清楚。
- 外部库、代码、图片和资料均正确致谢或引用。
- 删除所有 `TODO` 和占位数值。
- 在另一台电脑或 Overleaf 上完成一次干净编译。
- 检查最终页数、字体、行距及课程要求。
- 按课程要求将报告和代码放入最终提交 ZIP。

