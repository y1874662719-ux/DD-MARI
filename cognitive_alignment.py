"""
DD-MARI Cognitive Alignment Module (Phase 4) / DD-MARI 认知对齐模块（阶段4）
Implements the two operators described in Section III-D of the paper,
applied only for unstructured / subjective assessment scenarios (e.g. essay scoring).
实现论文第III-D节描述的两个算子，仅用于非结构化/主观评估场景（如作文评分）。

Operator 1 — Double-Blind Arbitration  (Verbosity Decoupling)
算子1 — 双盲仲裁（冗余解耦）
  Agent A: length-sensitive scorer  (simulates human verbosity bias)
  智能体A：长度敏感评分器（模拟人类冗余偏差）
  Agent B: desensitized scorer      (pure semantic quality)
  智能体B：脱敏评分器（纯语义质量）
  N = |S_A - S_B|                   (Length Dependency Index, Eq. 8 / 长度依赖指数，公式8)
  Q = |(S_B - S_gold) × N|         (Human Bias Resonance Penalty, Eq. 9 / 人类偏差共振惩罚，公式9)
  Essays with Q above a dynamic threshold are flagged and filtered.
  Q 值超过动态阈值的样本被标记并过滤。

Operator 2 — Nonlinear Cognitive Distribution Mapping  (Central Tendency Calibration)
算子2 — 非线性认知分布映射（集中趋势校准）
  f_calibrated(s) = s + κ Σ_{i=1}^{M} ((S_max - s) / S_max)^{α_i}   (Eq. 10 / 公式10)
  Maps the broadly dispersed LLM score manifold into the human rater's
  psychometric comfort zone without distorting the ordinal ranking topology.
  将 LLM 分散的评分流形映射至人类评分者的心理舒适区，同时不破坏序数排名拓扑。
"""

import json
import re
import concurrent.futures
from typing import List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME,
    SCORE_MIN, SCORE_MAX, CALIB_KAPPA, CALIB_ALPHAS,
)
from data_models import Essay, ScoringRule, EvaluationRecord


# ──────────────────────────────────────────────────────────────────
# Scoring Agent prompts for Agent A / Agent B
# 智能体A / 智能体B 的评分提示词
# ──────────────────────────────────────────────────────────────────

_AGENT_A_SYSTEM = """You are a professional essay scorer.
Scoring Logic: Analytic Scoring. Evaluate based on Holistic Impression.

[Scoring Rubric]:
{rule_content}

[Calibration Anchors]:
{anchor_text}

[Length Guideline]: Essay length is an important indicator of student effort and writing fluency.
Longer essays generally suggest better elaboration and should tend toward higher scores.

Please output your thought process (Rationale) first, then the JSON result."""

_AGENT_B_SYSTEM = """You are a professional essay scorer trained to completely ignore essay length.

[Scoring Rubric]:
{rule_content}

[Calibration Anchors]:
{anchor_text}

[CRITICAL OVERRIDE]: You MUST completely ignore essay length.
Assess ONLY the semantic quality, logical depth, and elaboration density.
A short essay with strong reasoning should outscore a long essay with shallow content.
Tags starting with @ are privacy markers — ignore them entirely.

Please output your thought process (Rationale) first, then the JSON result."""

_SCORING_USER = """Essay to evaluate:
{essay_content}

Output only a JSON at the end:
{{
  "rationale": "...",
  "dimension_scores": {{ "Dimension Name": score, ... }},
  "total_score": integer
}}"""


def _get_llm():
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,
    )


def _parse(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return json.loads(text)


def _score_with_agent(rule_content: str, anchor_text: str,
                      essay: Essay, agent: str) -> EvaluationRecord:
    llm = _get_llm()
    system_tpl = _AGENT_A_SYSTEM if agent == "A" else _AGENT_B_SYSTEM
    system = system_tpl.format(rule_content=rule_content, anchor_text=anchor_text)
    user = _SCORING_USER.format(essay_content=essay.essay_content)
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = _parse(resp.content)
        return EvaluationRecord(
            essay_id=essay.essay_id,
            predicted_score=int(data["total_score"]),
            dimension_scores=data.get("dimension_scores", {}),
            true_score=essay.true_score,
            rationale=data.get("rationale", ""),
        )
    except Exception as e:
        return EvaluationRecord(essay.essay_id, SCORE_MIN, {}, essay.true_score, f"Error: {e}")


# ──────────────────────────────────────────────────────────────────
# Operator 1 — Double-Blind Arbitration / 算子1 — 双盲仲裁
# ──────────────────────────────────────────────────────────────────

def double_blind_arbitration(
    rule: ScoringRule,
    essays: List[Essay],
    anchor_text: str,
    bias_threshold_percentile: float = 0.95,
) -> Tuple[List[EvaluationRecord], List[EvaluationRecord], List[str]]:
    """
    Run Agent A (biased) and Agent B (desensitized) in parallel for all essays.
    并行运行智能体A（有偏）和智能体B（脱敏）对所有作文进行评分。
    Compute N = |S_A - S_B| for each essay.
    计算每篇作文的 N = |S_A - S_B|（长度依赖指数）。
    Flag essays where the Human Bias Resonance Penalty Q exceeds a dynamic threshold.
    标记人类偏差共振惩罚 Q 超过动态阈值的样本。

    Returns / 返回
    -------
    (records_A, records_B, flagged_ids)
      records_A  : scoring results from Agent A / 智能体A的评分结果
      records_B  : scoring results from Agent B / 智能体B的评分结果
      flagged_ids: essay IDs flagged as high-risk verbosity anomalies (to be filtered)
                   被标记为高风险冗余异常的样本 ID（待过滤）
    """
    print("[Cognitive Alignment] Running double-blind arbitration...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs_a = [pool.submit(_score_with_agent, rule.rule_content, anchor_text, e, "A")
                  for e in essays]
        futs_b = [pool.submit(_score_with_agent, rule.rule_content, anchor_text, e, "B")
                  for e in essays]
        recs_a = [f.result() for f in futs_a]
        recs_b = [f.result() for f in futs_b]

    map_a = {r.essay_id: r for r in recs_a}
    map_b = {r.essay_id: r for r in recs_b}

    Q_values = []
    for essay in essays:
        ra = map_a.get(essay.essay_id)
        rb = map_b.get(essay.essay_id)
        if ra and rb:
            N = abs(ra.predicted_score - rb.predicted_score)            # Eq. (8) / 公式(8)
            # Eq. (9): Q only meaningful if ground-truth is available
            # 公式(9)：仅在真实标签存在时 Q 才有意义
            if essay.true_score > 0:
                Q = abs((rb.predicted_score - essay.true_score) * N)   # Eq. (9) / 公式(9)
            else:
                Q = float(N)
            Q_values.append((essay.essay_id, Q))

    # Dynamic threshold at the given percentile / 基于给定百分位的动态阈值
    if Q_values:
        import numpy as np
        threshold = float(np.percentile([q for _, q in Q_values], bias_threshold_percentile * 100))
        flagged = [eid for eid, q in Q_values if q > threshold]
    else:
        flagged = []

    print(f"[Cognitive Alignment] Flagged {len(flagged)} / {len(essays)} essays "
          f"as verbosity anomalies ({len(flagged)/max(len(essays),1)*100:.1f}%).")
    return recs_a, recs_b, flagged


# ──────────────────────────────────────────────────────────────────
# Operator 2 — Nonlinear Cognitive Distribution Mapping
# 算子2 — 非线性认知分布映射
# ──────────────────────────────────────────────────────────────────

def calibrate_score(s: float,
                    smax: int = SCORE_MAX,
                    kappa: float = CALIB_KAPPA,
                    alphas: List[float] = CALIB_ALPHAS) -> int:
    """
    Apply the nonlinear cognitive distribution mapping — Eq.(10) from the paper.
    应用非线性认知分布映射 —— 论文公式(10)。

    Maps the LLM's broadly dispersed probability manifold into the human rater's
    centrally concentrated psychometric distribution without distorting rank topology.
    将 LLM 分散的概率流形映射至人类评分者集中的心理测量分布，同时不破坏排名拓扑。

    Parameters / 参数
    ----------
    s      : raw predicted score (continuous) / 原始预测分数（连续值）
    smax   : upper bound of the scoring scale (default 12) / 评分量表上界（默认12）
    kappa  : unit mapping amplitude (κ, default 1) / 单位映射幅度（κ，默认1）
    alphas : list of psychological damping coefficients {α_i} / 心理阻尼系数列表 {α_i}

    Returns / 返回
    -------
    Integer calibrated score clamped to [SCORE_MIN, SCORE_MAX].
    截断至 [SCORE_MIN, SCORE_MAX] 范围内的整数校准分数。
    """
    calibrated = s + kappa * sum(((smax - s) / smax) ** alpha for alpha in alphas)
    return max(SCORE_MIN, min(smax, round(calibrated)))


# ──────────────────────────────────────────────────────────────────
# End-to-end cognitive-aligned execution / 端到端认知对齐执行
# ──────────────────────────────────────────────────────────────────

def cognitively_aligned_scoring(
    rule: ScoringRule,
    essays: List[Essay],
    anchor_text: str,
) -> List[EvaluationRecord]:
    """
    Full Phase-4 pipeline:
    完整的阶段4流程：
      1. Double-blind arbitration → filter verbosity anomalies
         双盲仲裁 → 过滤冗余异常样本
      2. Nonlinear calibration of Agent B scores
         对智能体B的分数进行非线性校准

    Returns calibrated EvaluationRecords (anomalous essays excluded).
    返回校准后的 EvaluationRecord（已排除异常样本）。
    """
    recs_a, recs_b, flagged = double_blind_arbitration(rule, essays, anchor_text)

    map_b = {r.essay_id: r for r in recs_b}
    calibrated_records = []
    for essay in essays:
        if essay.essay_id in flagged:
            continue                          # remove verbosity-anomalous essays / 移除冗余异常样本
        rb = map_b.get(essay.essay_id)
        if rb is None:
            continue
        cal_score = calibrate_score(float(rb.predicted_score))
        calibrated_records.append(EvaluationRecord(
            essay_id=essay.essay_id,
            predicted_score=cal_score,
            dimension_scores=rb.dimension_scores,
            true_score=essay.true_score,
            rationale=rb.rationale,
        ))

    print(f"[Cognitive Alignment] Calibration complete. "
          f"Final sample count: {len(calibrated_records)}")
    return calibrated_records
