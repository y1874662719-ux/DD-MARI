# DD-MARI: Defect-Driven Multi-Agent Rule Induction
# DD-MARI：缺陷驱动的多智能体规则归纳框架

[![Paper](https://img.shields.io/badge/Paper-IEEE_TII-blue)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of the paper:

> **Beyond Black-Box Scoring: An Interpretable and Defect-Driven Few-Shot Rule Induction Framework for Document Quality Assessment**

本文的代码实现：

> **超越黑箱评分：一种可解释的缺陷驱动少样本规则归纳框架用于文档质量评估**

---

![Algorithm Overview](algorithm_overview.png)

---

## Overview / 概述

DD-MARI addresses three black-box problems in automated Document Quality Assessment (DQA):

DD-MARI 解决了自动化文档质量评估（DQA）中的三个黑箱问题：

| Problem / 问题 | Source / 来源 | DD-MARI Solution / DD-MARI 解决方案 |
|---|---|---|
| **Parameter Black-Box** / **参数黑箱** | End-to-end neural models / 端到端神经模型 | Explicit natural-language scoring rules / 显式自然语言评分规则 |
| **Logical Black-Box** / **逻辑黑箱** | Naive LLM self-refinement / 朴素 LLM 自我优化 | Defect-driven multi-agent optimization / 缺陷驱动多智能体优化 |
| **Cognitive Black-Box** / **认知黑箱** | Human–AI psychometric misalignment / 人机心理测量失配 | Double-blind arbitration + nonlinear calibration / 双盲仲裁 + 非线性校准 |


---

## Framework Architecture / 框架架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    DD-MARI Pipeline                             │
│                                                                 │
│  Phase 2: Cold Start          Phase 3: Iterative Optimization   │
│  阶段2：冷启动                  阶段3：迭代优化                   │
│  ┌──────────────────┐         ┌──────────────────────────────┐  │
│  │ Standard         │         │ Defect      Feature          │  │
│  │ Generation  ───► │ R₀      │ Diagnosis ──Library (RAG)    │  │
│  │ Agent (SGA) │    │ ──────► │ Agent       ▼                │  │
│  │ 标准生成智能体│    │         │ (DDA)   Rule Optimization   │  │
│  └─────────────┘    │         │ 缺陷诊断    Agent (ROA)       │  │
│  Spearman-guided    │         │ 智能体      规则优化智能体     │  │
│  selection          │         │                 ▼            │  │
│  Spearman引导选择    │         │ Scoring Agent (SA) + Rollback│  │
│                     │         │ 评分智能体 + 回滚机制          │  │
│                     │         └──────────────────────────────┘  │
│                                                                 │
│  Phase 4: Cognitive-Aligned Execution (subjective tasks)        │
│  阶段4：认知对齐执行（主观任务）                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Agent A (length-sensitive) ──┐                         │     │
│  │ 智能体A（长度敏感）             ├─► N = |Sₐ−S_b|         │     │
│  │ Agent B (desensitized)   ────┘   长度依赖指数 ↓         │      │
│  │ 智能体B（脱敏）                   Verbosity Filter       │     │
│  │                                  冗余过滤               │     │
│  │                                        ↓               │     │
│  │                              Nonlinear Calibration     │     │
│  │                              非线性校准                 │     │
│  │                              f(s) = s + κΣ(·)^αᵢ       │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start / 快速开始

### 1. Clone and Install / 克隆与安装

```bash
git clone https://github.com/your-username/DD-MARI.git
cd DD-MARI
pip install -r requirements.txt
```

### 2. Configure API Key / 配置 API 密钥

```bash
cp .env.example .env
# Edit .env and set your LLM API key
# 编辑 .env 文件，填入你的 LLM API 密钥
```

The framework uses the [DeepSeek API](https://api.deepseek.com) by default (compatible with the OpenAI SDK).
Any OpenAI-compatible endpoint can be used by changing `LLM_BASE_URL` in `.env`.

框架默认使用 [DeepSeek API](https://api.deepseek.com)（兼容 OpenAI SDK）。
可通过修改 `.env` 中的 `LLM_BASE_URL` 切换至任意 OpenAI 兼容接口。

### 3. Prepare Data / 准备数据

Download the ASAP dataset from [Kaggle](https://www.kaggle.com/c/asap-aes/data) and place the files in `data/`:

从 [Kaggle](https://www.kaggle.com/c/asap-aes/data) 下载 ASAP 数据集，将文件放入 `data/` 目录：

```
data/
├── asap_set1_gradient.xlsx    # 10–20 quality-stratified essays for cold-start calibration
│                              # 10–20 篇质量分层样本，用于冷启动校准
├── asap_set1_training.xlsx    # 10 gold-standard + 30 hard-negative essays (40 total)
│                              # 10 篇金标准 + 30 篇困难负样本（共 40 篇）
├── asap_set1_test.xlsx        # Hold-out test set (~1,643 essays from ASAP Set 1)
│                              # 保留测试集（约 1,643 篇）
└── Set1_standard.docx         # Official ASAP Set 1 scoring guidelines document
                               # ASAP Set 1 官方评分标准文档
```

#### Data Format / 数据格式（Excel 列名）

| Column / 列名 | Description / 描述 |
|---|---|
| `essay_id` | Unique essay identifier / 唯一样本标识符 |
| `essay` | Essay text (may contain `@CAPS1`, `@NUM1` etc. — privacy markers) / 作文文本（可能含有隐私标记符） |
| `domain1_score` | ASAP combined score, **2–12 scale** (sum of two raters × 1–6 each) / ASAP 综合分，**2–12 分制**（两位评分者各 1–6 分之和） |

#### Score Mapping / 分数约定

DD-MARI uses the **ASAP combined score (2–12)** directly — the sum of two raters each scoring 1–6.
Store the combined score as-is in the `domain1_score` column. No conversion is needed.

DD-MARI 直接使用 **ASAP 综合分（2–12）** —— 两位评分者各 1–6 分之和。
`domain1_score` 列直接存储综合分，无需任何转换。

#### Training Pool Construction / 训练集构建

The `asap_set1_training.xlsx` file should have an additional column `sample_type`:

`asap_set1_training.xlsx` 需要额外添加 `sample_type` 列：

| `sample_type` value / 取值 | Meaning / 含义 |
|---|---|
| `gold` / `original` | High-quality anchor essay / 高质量锚样本 |
| `aug` / `augmented` / `negative` | Hard-negative variant (proximal score, ∆ ≤ δ) / 困难负样本（相邻分数，∆ ≤ δ） |

If `sample_type` is absent, the framework infers it from the `essay_id` string
(looks for keywords: `aug`, `enhanced`, `negative`, `neg`).

若缺少 `sample_type` 列，框架将从 `essay_id` 字符串自动推断
（识别关键词：`aug`、`enhanced`、`negative`、`neg`）。

---

### 4. Run Training / 运行训练

```bash
# Full pipeline (cold start → iterative optimization)
# 完整流程（冷启动 → 迭代优化）
python main.py

# Resume from checkpoint (useful after interruption)
# 从检查点恢复（中断后续跑）
python main.py --resume
```

Training output is saved to `output/`:

训练输出保存至 `output/`：
- `rule_history.json`  — all rule versions with their QWK scores / 全部规则版本及对应 QWK 分数
- `final_rule.json`    — the best induced rule / 最优归纳规则
- `feature_library.json` — accumulated defect/quality features / 累积的缺陷/质量特征库

### 5. Evaluate / 评估

```bash
# Evaluate the final rule on a test set
# 在测试集上评估最终规则
python main.py --eval --test_file data/asap_set1_test.xlsx --output output/result.xlsx
```


## Project Structure / 项目结构

```
DD-MARI/
├── main.py                  # CLI entry point / 命令行入口
├── workflow.py              # Full DD-MARI pipeline (Algorithm 1) / 完整流程（算法1）
├── agents.py                # Four LLM agents: SGA, SA, DDA, ROA / 四个 LLM 智能体
├── cognitive_alignment.py   # Phase 4: verbosity decoupling + calibration / 阶段4：冗余解耦 + 校准
├── data_manager.py          # Data loading, anchor extraction, feature library / 数据加载、锚点提取、特征库
├── evaluator.py             # QWK and auxiliary metrics / QWK 及辅助指标
├── data_models.py           # Data classes: Essay, ScoringRule, EvaluationRecord / 数据类定义
├── config.py                # All configuration and hyperparameters / 所有配置与超参数
├── data/                    # Dataset directory (not tracked by git) / 数据目录（不纳入 git）
│   ├── asap_set1_gradient.xlsx
│   ├── asap_set1_training.xlsx
│   ├── asap_set1_test.xlsx
│   └── Set1_standard.docx
├── output/                  # Generated output (not tracked by git) / 生成输出（不纳入 git）
│   ├── rule_history.json
│   ├── final_rule.json
│   └── feature_library.json
├── requirements.txt
├── .env.example
└── README.md
```


## License / 许可证

MIT License. See [LICENSE](LICENSE) for details.

MIT 许可证，详见 [LICENSE](LICENSE)。
