"""
DD-MARI Cognitive Alignment Module / DD-MARI 认知对齐模块

Implements two operators described in the paper:
实现论文描述的两个算子：

Operator 1 — CDAP (Contrastive Divergence Arbitration Pre-Screener)
算子1 — CDAP（对比散度仲裁预筛选器）
  Used in two places / 在两处使用:
  Phase 1: Filter the training pool (no rule, simplified holistic prompts) → Dclean
  阶段1：过滤训练池（无规则，简化整体提示）→ Dclean
  Final Inference: Cognitive Purification Filter on holdout (uses final rule)
  最终推理：保留集上的认知净化过滤器（使用最终规则）
  N = |SA - SB|; essays with N >= τN are removed.
  N = |SA - SB|；N >= τN 的样本被移除。

Operator 2 — Nonlinear Cognitive Distribution Mapping  (Eq. 10)
算子2 — 非线性认知分布映射（公式10）
  f(s) = s + κ Σ((S_max - s) / S_max)^{α_i}
  Calibration parameters {α_i} are selected inside Phase 3 loop via grid search
  minimizing the Wasserstein distance between calibrated and true score distributions.
  校准参数 {α_i} 通过网格搜索在阶段3循环内选择，最小化校准分数与真实分数分布之间的 Wasserstein 距离。
"""

import json
import re
import concurrent.futures
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME,
    SCORE_MIN, SCORE_MAX, CALIB_KAPPA, CALIB_ALPHAS,
    CDAP_TAU_N, CALIB_ALPHA_GRID,
)
from data_models import Essay, ScoringRule, EvaluationRecord


# ──────────────────────────────────────────────────────────────────
# Phase 1 CDAP prompts — no rule, simplified holistic assessment
# 阶段1 CDAP 提示词 —— 无规则，简化整体评估
# ──────────────────────────────────────────────────────────────────

_CDAP_A_SYSTEM = """You are a professional essay scorer.
Score the essay holistically on a {score_min}–{score_max} scale.

[Calibration Anchors]:
{anchor_text}

[Length Guideline]: Essay length is an important quality signal.
Longer essays generally demonstrate better elaboration and should tend toward higher scores.

Output only JSON: {{"total_score": integer}}"""

_CDAP_B_SYSTEM = """You are a professional essay scorer.
Score the essay holistically on a {score_min}–{score_max} scale.

[Calibration Anchors]:
{anchor_text}

[CRITICAL OVERRIDE]: Completely ignore essay length.
Assess ONLY semantic quality, logical depth, and elaboration density.
A short essay with strong reasoning should outscore a long essay with shallow content.
Tags starting with @ are privacy markers — ignore them entirely.

Output only JSON: {{"total_score": integer}}"""

_CDAP_USER = """Essay:
{essay_content}

Output only JSON: {{"total_score": integer}}"""


# ──────────────────────────────────────────────────────────────────
# Final Inference CDAP prompts — with rule
# 最终推理 CDAP 提示词 —— 含规则
# ──────────────────────────────────────────────────────────────────

_INFER_A_SYSTEM = """You are a professional essay scorer.
Scoring Logic: Analytic Scoring. Evaluate based on Holistic Impression.

[Scoring Rubric]:
{rule_content}

[Calibration Anchors]:
{anchor_text}

[Length Guideline]: Essay length is an important indicator of student effort and writing fluency.
Longer essays generally suggest better elaboration and should tend toward higher scores.

Output only JSON: {{"total_score": integer}}"""

_INFER_B_SYSTEM = """You are a professional essay scorer trained to completely ignore essay length.

[Scoring Rubric]:
{rule_content}

[Calibration Anchors]:
{anchor_text}

[CRITICAL OVERRIDE]: You MUST completely ignore essay length.
Assess ONLY the semantic quality, logical depth, and elaboration density.
Tags starting with @ are privacy markers — ignore them entirely.

Output only JSON: {{"total_score": integer}}"""

_INFER_USER = """Essay:
{essay_content}

Output only JSON: {{"total_score": integer}}"""


# ──────────────────────────────────────────────────────────────────
# Shared helpers / 共享辅助函数
# ──────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,
    )


def _parse_score(text: str) -> int:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(text[start:end])
        return int(data.get("total_score", SCORE_MIN))
    raise ValueError(f"Cannot parse score from: {text[:200]}")


# ──────────────────────────────────────────────────────────────────
# Operator 1a — Phase 1: CDAP Training Pool Filter
# 算子1a — 阶段1：CDAP 训练池过滤
# ──────────────────────────────────────────────────────────────────

def _cdap_score_essay(essay: Essay, anchor_text: str, agent: str) -> int:
    """Score a single essay with the simplified Phase 1 CDAP agent (no rule)."""
    llm = _get_llm()
    system_tpl = _CDAP_A_SYSTEM if agent == "A" else _CDAP_B_SYSTEM
    system = system_tpl.format(
        score_min=SCORE_MIN, score_max=SCORE_MAX, anchor_text=anchor_text
    )
    user = _CDAP_USER.format(essay_content=essay.essay_content)
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return max(SCORE_MIN, min(SCORE_MAX, _parse_score(resp.content)))
    except Exception as e:
        print(f"[CDAP] Scoring error for {essay.essay_id}: {e}")
        return SCORE_MIN


def cdap_filter(
    essays: List[Essay],
    anchor_text: str,
    threshold: float = CDAP_TAU_N,
) -> List[Essay]:
    """
    Phase 1: CDAP filter on training pool (Dpool → Dclean).
    阶段1：对训练池执行 CDAP 过滤（Dpool → Dclean）。

    Runs Agent A (length-sensitive) and Agent B (desensitized) in parallel.
    并行运行智能体A（长度敏感）和智能体B（脱敏）。
    Essays with N = |SA - SB| >= threshold are removed.
    N = |SA - SB| >= 阈值的样本被移除。

    Returns / 返回: Dclean (filtered essay list)
    """
    print(f"[CDAP Phase 1] Filtering {len(essays)} training essays (τN = {threshold})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs_a = [pool.submit(_cdap_score_essay, e, anchor_text, "A") for e in essays]
        futs_b = [pool.submit(_cdap_score_essay, e, anchor_text, "B") for e in essays]
        scores_a = [f.result() for f in futs_a]
        scores_b = [f.result() for f in futs_b]

    clean = []
    filtered_count = 0
    for essay, sa, sb in zip(essays, scores_a, scores_b):
        n = abs(sa - sb)
        if n < threshold:
            clean.append(essay)
        else:
            filtered_count += 1
            print(f"  Filtered: {essay.essay_id}  N={n}  (SA={sa}, SB={sb})")

    print(f"[CDAP Phase 1] {filtered_count} essays removed. Dclean: {len(clean)} essays.")
    return clean


# ──────────────────────────────────────────────────────────────────
# Operator 1b — Final Inference: Cognitive Purification Filter
# 算子1b — 最终推理：认知净化过滤器
# ──────────────────────────────────────────────────────────────────

def _infer_score_essay(rule: ScoringRule, essay: Essay, anchor_text: str, agent: str) -> int:
    """Score a single essay using the inference CDAP agent (with rule)."""
    llm = _get_llm()
    system_tpl = _INFER_A_SYSTEM if agent == "A" else _INFER_B_SYSTEM
    system = system_tpl.format(rule_content=rule.rule_content, anchor_text=anchor_text)
    user = _INFER_USER.format(essay_content=essay.essay_content)
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return max(SCORE_MIN, min(SCORE_MAX, _parse_score(resp.content)))
    except Exception as e:
        print(f"[CDAP Inference] Scoring error for {essay.essay_id}: {e}")
        return SCORE_MIN


def cdap_inference_filter(
    rule: ScoringRule,
    essays: List[Essay],
    anchor_text: str,
    threshold: float = CDAP_TAU_N,
) -> List[Essay]:
    """
    Cognitive Purification Filter applied during final inference.
    最终推理时应用的认知净化过滤器。

    Uses the final scoring rule; removes essays with N = |SA - SB| >= threshold.
    使用最终评分规则；移除 N = |SA - SB| >= 阈值的样本。

    Returns / 返回: filtered essay list (anomalous essays excluded)
    """
    print(f"[CDAP Inference] Filtering {len(essays)} holdout essays (τN = {threshold})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs_a = [pool.submit(_infer_score_essay, rule, e, anchor_text, "A") for e in essays]
        futs_b = [pool.submit(_infer_score_essay, rule, e, anchor_text, "B") for e in essays]
        scores_a = [f.result() for f in futs_a]
        scores_b = [f.result() for f in futs_b]

    clean = []
    filtered_count = 0
    for essay, sa, sb in zip(essays, scores_a, scores_b):
        n = abs(sa - sb)
        if n < threshold:
            clean.append(essay)
        else:
            filtered_count += 1

    rate = filtered_count / max(len(essays), 1) * 100
    print(f"[CDAP Inference] {filtered_count} essays removed ({rate:.1f}%). "
          f"Retained: {len(clean)} essays.")
    return clean


# ──────────────────────────────────────────────────────────────────
# Operator 2 — Nonlinear Cognitive Distribution Mapping
# 算子2 — 非线性认知分布映射
# ──────────────────────────────────────────────────────────────────

def calibrate_score(
    s: float,
    smax: int = SCORE_MAX,
    kappa: float = CALIB_KAPPA,
    alphas: List[float] = CALIB_ALPHAS,
) -> int:
    """
    Apply nonlinear cognitive distribution mapping — Eq.(10) from the paper.
    应用非线性认知分布映射 —— 论文公式(10)。

    f(s) = s + κ Σ_{i=1}^{M} ((S_max - s) / S_max)^{α_i}
    Maps the LLM's dispersed score distribution into the human rater's centrally
    concentrated psychometric distribution without distorting ordinal ranking.
    将 LLM 分散的评分分布映射至人类评分者集中的心理测量分布，同时不破坏序数排名。
    """
    calibrated = s + kappa * sum(((smax - s) / smax) ** alpha for alpha in alphas)
    return max(SCORE_MIN, min(smax, round(calibrated)))


def apply_calibration_to_records(
    records: List[EvaluationRecord],
    alphas: List[float],
    kappa: float = CALIB_KAPPA,
) -> List[EvaluationRecord]:
    """
    Apply nonlinear calibration to a list of EvaluationRecords.
    对 EvaluationRecord 列表应用非线性校准。
    Returns new records with calibrated predicted_score values.
    返回校准后的 predicted_score 值的新记录列表。
    """
    calibrated = []
    for r in records:
        cal_score = calibrate_score(float(r.predicted_score), SCORE_MAX, kappa, alphas)
        calibrated.append(EvaluationRecord(
            essay_id=r.essay_id,
            predicted_score=cal_score,
            dimension_scores=r.dimension_scores,
            true_score=r.true_score,
            rationale=r.rationale,
        ))
    return calibrated


def grid_search_alphas(
    raw_scores: List[float],
    true_scores: List[float],
    smax: int = SCORE_MAX,
    kappa: float = CALIB_KAPPA,
    alpha_grid: List[float] = CALIB_ALPHA_GRID,
) -> List[float]:
    """
    Find optimal {α1, α2} by minimizing Wasserstein distance between the
    calibrated score distribution and the true score distribution.
    通过最小化校准分数分布与真实分数分布之间的 Wasserstein 距离，找到最优 {α1, α2}。

    Parameters / 参数
    ----------
    raw_scores  : predicted scores from the Scoring Agent / 评分智能体预测的原始分数
    true_scores : ground-truth scores for the calibration set / 校准集的真实标签分数
    alpha_grid  : candidate α values for grid search / 网格搜索的候选 α 值列表

    Returns / 返回
    -------
    List[float]: best [α1, α2] pair / 最优 [α1, α2] 组合
    """
    from scipy.stats import wasserstein_distance

    best_alphas = list(CALIB_ALPHAS)
    best_dist = float("inf")

    for a1 in alpha_grid:
        for a2 in alpha_grid:
            alphas = [a1, a2]
            calibrated = [calibrate_score(s, smax, kappa, alphas) for s in raw_scores]
            dist = wasserstein_distance(calibrated, true_scores)
            if dist < best_dist:
                best_dist = dist
                best_alphas = alphas

    return best_alphas
