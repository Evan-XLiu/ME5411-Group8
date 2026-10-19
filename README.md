# ME5411 Group 8

ME5411 Robot Vision and AI 小组项目仓库。

## 文件结构

```text
data/       数据集
src/        Task 1--7 的代码
results/    实验结果与模型
report/     LaTeX 报告主稿（格式待按作业要求调整）
tests/      测试代码
```

## LaTeX 报告

- 主文件：`report/main.tex`
- 参考文献：`report/references.bib`
- 图片目录：`report/figures/`

`report/` 是唯一维护的报告主稿，已合入 Wanghongchao 的方法与实验配置扩写。
原 `report_from_Wanghongchao/` 已合并移除，历史版本可在 Git 记录中查看。
代码按课程 Task 1–7 编号：Task 1–6 为图像处理，Task 7 为分类。
使用 XeLaTeX 编译 Arial 11pt 单倍行距主稿：`make pdf`。
Overleaf 需选择 XeLaTeX 并配置 Arial 字体；本机需要安装 Arial。

## 安装与复现

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python segment.py                 # 加 --show 显示原图和各阶段
python scripts/run_experiments.py --epochs 20
python scripts/build_report_results.py
python scripts/predict_characters.py --model results/models/baseline_final.pt
python scripts/predict_characters.py --model results/models/yjn_final.pt
python scripts/predict_characters.py --model results/models/hog_svm_final.pkl
python -m pytest -q
make pdf
python scripts/package_submission.py
```

先将课程官方数据放到 `dataset_2026/`：七个类别文件夹与
`Test_Image_2026.bmp`；此图只做 Task 1–6 和推理，不加入训练集。
生成的 `submission/Group 08.zip` 不包含官方数据；复现时须按此布局提供。
Windows 激活虚拟环境使用 `.venv\Scripts\activate`。

- `segment.py`：PR #1 的完整处理入口，默认 ROI `(45,180,930,180)`。
- `src/task_07/cnn.py`、`classical.py`：保留当前双 CNN 和传统分类器。
- `src/task_05/cnn.py`、`src/task_06/classical.py`：旧路径兼容入口。
- `src/legacy_pipeline/`：集成前代码参考备份，不作为运行入口。
- `CNN/CNN_train_original.py`：YJN 原始参考脚本；`CNN_train.py` 为可用入口。
- 主要结果：`results/experiments/task7_results.json`；
  样本路径、SHA-256 和划分索引：`results/experiments/split_manifest.json`。
- 每类先固定 75/25 训练/测试；仅在训练部分内部划出 20% 验证集选择
  resize/padding 和学习率/SVM C，再在全部 75% 训练集重训、最后评估测试集。
- 历史 smoke 实验已查看过同一测试集，因此本次结果不能称作从未接触的盲测；
  样本级随机划分也不能保证字体/来源层面独立，需组员讨论适用边界。
- 模型由命令本地生成，不上传权重；只加载自己生成或可信来源的 checkpoint/pickle。

## 完成状态与剩余 TODO（2026-10-19）

依据 `ME5411_Project_AY2627.pdf`。截止为 **2026-11-20 23:59（新加坡时间）**。

### 已完成的工程与实验

- [x] 选择性整合 PR #1 完整 Task 1–6、对比图、指标和十个字符裁剪；
  保留当前双 CNN、新训练接口和旧模块兼容入口。
- [x] 加入原图显示、3/5/7 均值比较、ROI、两种二值化、轮廓及投影分割。
- [x] 统一 Task 7 分类器；保存样本哈希和固定 75/25 划分索引；
  只用训练集内部验证集选择预处理和超参数，不用芯片图训练。
- [x] 提供独立的模型加载和字符预测命令、实验重现脚本与打包脚本。
- [x] 增加课程流水线测试，两种 CNN 的训练/保存/重载测试保持通过。
- [x] 报告改为 XeLaTeX、Arial 11pt、单倍行距；加入流程图和各阶段截图。
- [x] 标题页加入 Xuanwei Liu、Jianan Yan、Jiayi Guo、Hongchao Wang，
  学号按用户要求暂填 XXX。

### 必须由组员完成 / 确认

- [ ] 用实际学号替换 `report/members.tex` 中的 XXX。
- [ ] 核对 CANVAS：本地 `Test_Image_2026.bmp` 是否为指定输入，
  作业 PDF 的图注使用 `charact2.bmp`；目前未获得确认。
- [ ] 核对十个分割区域的字符质量；文件名和图上的 H/D/4 等是人工参考标签，
  不是模型预测。分类器预测见实验 JSON，失败字符必须如实讨论。
- [ ] 根据真实结果撰写 Task 1–2 对比解释、Task 7-3 效果/效率解释、
  Discussion 和 Conclusion。作业明确要求想法、分析与推理源于组员，
  LLM 只可用于报告语言编辑；当前报告保留明确的组员分析待填位置。
- [ ] 核对 HOG/SVM 符合课程 Part 2 范围，审阅预处理/调参敏感性结果；
  当前单 seed、小网格实验不足以证明跨来源鲁棒性。
- [ ] 实际收集完整的作者贡献、外部代码/资料引用和 AI 辅助披露；
  清除最终稿中的工作稿提示，核实所有实验记录。
- [ ] 最终逐页复核版式、引用和内容（上限 50 页，典型 30–40 页，不凑页）；
  更新包后在干净环境重新复现。
- [ ] 全组确认后上传 `submission/Group 08.zip` 至 CANVAS，并核对上传版本。

### 审核说明

- 实验表由 `python scripts/build_report_results.py` 从 JSON 生成；
  JSON 是数值证据，不替代组员自己的论证。
- CNN 与 SVM 训练时间均不含输入预处理（SVM 不含 HOG 提取）；
  比较总效率时须额外测量完整端到端成本。
- 已有 smoke 实验看过相同测试分区，不声称最终结果是全新盲测。
- `submission/` 和本地模型被忽略；包包含 PDF、源代码、结果和复现说明，
  数据需按课程授权渠道取得。

### 实测状态

- [x] 完成 12 个验证配置和 3 个最终模型，CNN 每次 20 轮；保存逐类指标及混淆矩阵。
- [x] 完成 CNN/SVM 对芯片字符的加载与推理，二值裁剪反色以匹配训练集。
- [ ] **端到端识别仍未全部正确**：baseline 输出 `HD4A780A00`（9/10），
  YJN 输出 `8848888A88`，SVM 输出 `AAAAAAAAAA`。不能将数据集高准确率
  或人工参考标签当作整行识别成功；需组员研究训练/测试图像的差异与失败原因。
