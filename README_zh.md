# ProSIU 复现说明

本仓库提供已经确认结果的最终版 ProSIU，包含三个模块、三个 GFP 数据集的固定配置以及完整五个种子的参考结果。

**Built with Qwen。** 固定数值程序库来自 Qwen2.5-3B-Instruct。原始提示词与生成结果随代码提供，正式复现直接使用这份程序库。

## 环境

参考环境为 Python 3.10.20、PyTorch 2.6.0、CUDA 12.4、RTX 3090 24 GB。单次运行使用一张 GPU。文件读写和文本解析在 CPU 上完成，特征构建、训练、矩阵运算和批量评测使用 GPU。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
export CUDA_VISIBLE_DEVICES=0
```

## 完整流程

在仓库根目录运行：

```bash
prosiu download
prosiu prepare
prosiu features
prosiu run
prosiu evaluate
prosiu summarize
prosiu verify
```

各步依次完成：

1. 下载固定的 ProteinGym v1.3 数据和原始 GFP 数据表。
2. 构造 WT、单点、双点组成的测定四元组，生成固定划分和撤销请求。
3. 使用冻结 ESM-2 编码器构造上下文与序列特征。
4. 对三个蛋白分别运行种子 0、1、2、3、4，完成原模型训练、选择性遗忘和正式评测。
5. 从保存的预测重新计算指标。
6. 生成均值、样本标准差以及主表。
7. 与 `reference/results.json` 中的 15 次参考运行逐项比较。

环境安装后，也可以执行 `bash scripts/reproduce.sh`。已完成的准备阶段可以单独复用，后续直接从 `prosiu run` 开始。

首次特征提取需要下载公开的 ESM-2 35M 和 650M 权重。650M 仅用于 amacGFP 的最终排序头。已有模型目录时可指定：

```bash
prosiu features --encoder /path/to/esm2-35m --language-encoder /path/to/esm2-650m
```

## 参考结果

所有指标以百分数报告，结果为五个种子的均值±样本标准差。

| 数据集 | CA ↑ | RA ↑ | TA ↑ | PA ↑ |
|---|---:|---:|---:|---:|
| amacGFP | 58.95±0.68 | 73.08±0.50 | 59.82±1.06 | 57.21±1.70 |
| cgreGFP | 70.03±1.76 | 80.41±0.56 | 69.62±0.89 | 62.42±1.52 |
| avGFP | 62.38±1.10 | 74.34±0.45 | 61.48±0.24 | 61.72±0.89 |

CA、RA、TA 分别是撤销组合、保留组合和测试组合上的交互符号平衡准确率。PA 是固定程序候选对的交互偏好准确率。

`prosiu verify` 比较全部 60 个逐种子指标，默认容差为 0.1 个百分点，并将每项差异写入 `results/reproduction.json`。浮点计算和 GPU 库版本可能影响接近决策边界的样本。该命令保持实测数据原值。

2026 年 10 月 18 日，本发布版在[记录的环境](reference/environment.json)中重新下载原始数据、重建全部特征并重跑 15 次实验，60 个逐种子指标与参考值完全一致。核对记录见[复现报告](reference/reproduction_report.json)、[预处理对照](reference/preprocessing_check.json)和[预测重算报告](reference/evaluation.json)。

## 单独运行与双卡并行

```bash
prosiu run --assays amacGFP --seeds 0
```

两张卡可在两个终端分别执行：

```bash
CUDA_VISIBLE_DEVICES=0 prosiu run --seeds 0 1 2
CUDA_VISIBLE_DEVICES=1 prosiu run --seeds 3 4
```

不同种子的输出目录独立，全部完成后运行 `prosiu summarize`。使用 `--data-dir` 和 `--output-dir` 可以指定数据及结果目录，后续命令使用相同参数。

## 代码与结果

三个方法模块位于 `src/prosiu/`：

- `epistatic_program_selection.py`：Epistatic Program Selection。
- `mutation_preference_optimization.py`：Mutation Preference Optimization。
- `selective_epistasis_forgetting.py`：Selective Epistasis Forgetting。

最终超参数位于 `src/prosiu/assets/configs/`。每次运行保存原模型、遗忘模型及保留集重训参考参数。正式指标使用遗忘模型的预测，重训参考用于检查优化误差。

输出包括：

- `results/<assay>/seed_<seed>/result.json`：逐种子指标、配置和求解诊断。
- `checkpoint.pt`、`predictions.pt`：模型参数、预测和程序候选对。
- `results/summary.json`、`summary.csv`、`table.md`：完整三组数据的汇总。

本版本使用已固定的配置，直接复现实验。具体数据访问规则、损失函数和撤销机制见 [docs/protocol.md](docs/protocol.md)。

源代码包包含最终方法实现、固定程序库、参考指标及文档。数据、模型权重、本地环境和运行缓存通过 `.gitignore` 排除，使用公开下载流程获取。第三方资源来源和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
