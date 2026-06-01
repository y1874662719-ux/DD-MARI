"""
DD-MARI Main Workflow / DD-MARI 主工作流
Implements Algorithm 1 from the paper:
实现论文中的算法1：

  Phase 1 — CDAP Training Pool Filtering (Dpool → Dclean)
  阶段1 — CDAP 训练池过滤（Dpool → Dclean）

  Phase 2 — Spearman-Guided Cold-Start Initialization (on Dcal)
  阶段2 — Spearman 引导的冷启动初始化（基于 Dcal）

  Phase 3 — RAG-Enhanced Iterative Optimization with Intra-Loop Calibration
  阶段3 — 带循环内校准的 RAG 增强迭代优化

  Final  — Cognitive Purification Filter (CDAP inference) + Calibrated QWK on holdout
  最终   — 认知净化过滤器（CDAP 推理）+ 保留集上的校准 QWK
"""

import json
from typing import Optional, List, Tuple
from scipy.stats import spearmanr

from config import (
    MAX_ITERATIONS, BATCH_SIZE, ROLLBACK_THRESHOLD,
    RAG_TOP_K, HUMAN_IN_THE_LOOP, TASK_MODE,
    RULE_HISTORY_FILE, FINAL_RULE_FILE, FINAL_ALPHAS_FILE,
    ITER_EVAL_SIZE, ITER_EVAL_SEED,
    CALIB_ALPHAS, CALIB_KAPPA, SCORE_MAX, CALIB_ALPHA_GRID,
)
from data_models import Essay, ScoringRule, FeatureEntry
from agents import run_sga, run_sa, run_dda, run_roa
from evaluator import calculate_qwk, calculate_qwk_detailed
from cognitive_alignment import (
    cdap_filter, cdap_inference_filter,
    grid_search_alphas, apply_calibration_to_records,
)
import data_manager as dm


# ──────────────────────────────────────────────────────────────────
# Persistence helpers / 持久化辅助函数
# ──────────────────────────────────────────────────────────────────

def _save_rule(rule: ScoringRule, is_final: bool = False) -> None:
    history = []
    if RULE_HISTORY_FILE.exists():
        with open(RULE_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append(rule.to_dict())
    with open(RULE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if is_final:
        with open(FINAL_RULE_FILE, "w", encoding="utf-8") as f:
            json.dump(rule.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Workflow] Final rule saved → {FINAL_RULE_FILE.name}")


def _save_alphas(alphas: List[float]) -> None:
    with open(FINAL_ALPHAS_FILE, "w", encoding="utf-8") as f:
        json.dump(alphas, f)


def _load_latest_rule() -> Optional[ScoringRule]:
    if not RULE_HISTORY_FILE.exists():
        return None
    with open(RULE_HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    return ScoringRule.from_dict(history[-1]) if history else None


def _load_saved_alphas() -> List[float]:
    if FINAL_ALPHAS_FILE.exists():
        with open(FINAL_ALPHAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return list(CALIB_ALPHAS)


# ──────────────────────────────────────────────────────────────────
# Phase 2 — Spearman-Guided Cold-Start Initialization (on Dcal)
# 阶段2 — Spearman 引导的冷启动初始化（基于 Dcal）
# ──────────────────────────────────────────────────────────────────

def cold_start(
    doc_content: str,
    d_cal: List[Essay],
    anchor_text: str,
    num_candidates: int = 3,
) -> Tuple[ScoringRule, List[float]]:
    """
    Generate N candidate rules via SGA, evaluate each on Dcal using Spearman rank
    correlation (Eq. 5), select the best candidate, then run intra-loop grid search
    over {αi} to find initial calibration parameters minimizing Wasserstein distance.
    通过 SGA 生成 N 个候选规则，使用 Spearman 秩相关（公式5）在 Dcal 上评估每个候选，
    选择最优候选，然后通过网格搜索找到最小化 Wasserstein 距离的初始校准参数 {αi}。

    Returns / 返回: (R0, initial_alphas)
    """
    print("\n[Phase 2] Cold-Start Initialization")
    candidates = run_sga(doc_content, num_variants=num_candidates)

    best_rule: Optional[ScoringRule] = None
    best_rho = -1.0
    best_records = []

    gold_ranks = [e.true_score for e in d_cal]

    for idx, rule in enumerate(candidates):
        records = run_sa(rule, d_cal, anchor_text)
        pred_map = {r.essay_id: r.predicted_score for r in records}
        pred_ranks = [pred_map.get(e.essay_id, 0) for e in d_cal]

        rho, _ = spearmanr(gold_ranks, pred_ranks)        # Eq. (5) / 公式(5)
        print(f"  Candidate {idx + 1}: Spearman rho = {rho:.4f}")

        if rho > best_rho:
            best_rho = rho
            best_rule = rule
            best_records = records

    print(f"[Phase 2] Best candidate (rho = {best_rho:.4f}) selected as R0.")

    # Grid search for initial calibration alphas on Dcal
    # 在 Dcal 上网格搜索初始校准参数
    raw_scores = [r.predicted_score for r in best_records]
    true_scores = [e.true_score for e in d_cal]
    best_alphas = grid_search_alphas(raw_scores, true_scores, SCORE_MAX, CALIB_KAPPA, CALIB_ALPHA_GRID)
    print(f"[Phase 2] Initial calibration alphas: {best_alphas}")

    # Compute initial Dcal calibrated QWK as Phase 3 rollback baseline
    # 计算初始 Dcal 校准 QWK 作为阶段3回滚基线
    cal_records = apply_calibration_to_records(best_records, best_alphas)
    qwk = calculate_qwk(d_cal, cal_records)
    best_rule.qwk_score = qwk
    best_rule.rule_id = "v0"
    print(f"[Phase 2] Initial Dcal calibrated QWK = {qwk:.4f}")

    _save_rule(best_rule)
    return best_rule, best_alphas


# ──────────────────────────────────────────────────────────────────
# Phase 3 — RAG-Enhanced Iterative Optimization
# 阶段3 — RAG 增强迭代优化
# ──────────────────────────────────────────────────────────────────

def _human_review(iteration: int, analysis: dict) -> str:
    """Optional interactive expert validation of DDA-extracted features.
    可选的领域专家对 DDA 提取特征的交互式验证。
    """
    print(f"\n{'!' * 60}")
    print(f"[HITL] Iteration {iteration} — Expert Review")
    print(f"  Analysis: {analysis.get('analysis', '')}")
    print(f"  Features:")
    for feat in analysis.get("extracted_features", []):
        print(f"    [{feat.get('type')}] {feat.get('description')}")
    print(f"  Suggestion: {analysis.get('optimization_suggestion', '')}")
    print("  Options: [y] Approve  [e] Edit  [s] Skip")
    choice = input("  Your choice (y/e/s): ").lower().strip()
    if choice == "s":
        return "SKIP"
    if choice == "e":
        feedback = input("  Enter your expert instruction: ").strip()
        return feedback or "Approve and execute."
    return "Approve and execute."


def iterative_optimization(
    initial_rule: ScoringRule,
    initial_alphas: List[float],
    training_pairs: List[Tuple[Essay, List[Essay]]],
    d_cal: List[Essay],
    anchor_text: str,
    start_iteration: int = 1,
) -> Tuple[ScoringRule, List[float]]:
    """
    Defect-driven RAG-enhanced optimization loop with intra-loop calibration (Algorithm 1, Phase 3).
    带循环内校准的缺陷驱动 RAG 增强优化循环（算法1，阶段3）。

    For each iteration / 每轮迭代:
      1. Sample a mini-batch of (original, augmented) pairs
      2. Score with current rule → detect scoring inversions (Eq. 4)
      3. DDA diagnoses root-cause features
      4. [Optional] Human validates features
      5. Update feature library; RAG retrieves top-k features
      6. ROA proposes an updated rule Rnew
      7. Score Dcal with Rnew → grid search {αi} → calibrated QWK
      8. Rollback if calibrated QWK does not improve by τ (Algorithm 1 line 30)

    Returns / 返回: (final_rule, final_alphas)
    """
    print("\n[Phase 3] Iterative Optimization Started")
    print(f"  Rollback evaluated on Dcal ({len(d_cal)} essays) with intra-loop calibration.")
    best_rule = initial_rule
    best_alphas = list(initial_alphas)

    for i in range(start_iteration, MAX_ITERATIONS + 1):
        print(f"\n  --- Iteration {i}/{MAX_ITERATIONS}  "
              f"(Best Dcal calibrated QWK: {best_rule.qwk_score:.4f}) ---")

        # 1. Sample mini-batch / 采样小批量
        batch = dm.sample_batch(training_pairs, BATCH_SIZE)
        if not batch:
            print("  Training pool exhausted — stopping.")
            break
        orig, aug_list = batch[0]

        # 2. Score all samples in the batch / 对批次内所有样本评分
        all_samples = [orig] + aug_list
        records = run_sa(best_rule, all_samples, anchor_text)
        rec_map = {r.essay_id: r for r in records}

        orig_rec = rec_map.get(orig.essay_id)
        if orig_rec is None:
            continue

        # 3. Detect scoring inversions (Eq. 4) / 检测评分倒置（公式4）
        anomalies = [
            aug for aug in aug_list
            if (rec := rec_map.get(aug.essay_id)) and rec.predicted_score >= orig_rec.predicted_score
        ]

        if not anomalies:
            print("  No scoring inversions detected — rule is stable on this batch.")
            continue

        aug = anomalies[0]
        print(f"  Inversion detected: orig score={orig_rec.predicted_score} "
              f"(true={orig.true_score})  aug score={rec_map[aug.essay_id].predicted_score} "
              f"(true={aug.true_score})")

        # 4. DDA diagnosis / DDA 诊断
        analysis = run_dda(best_rule, orig, aug)

        # 5. Human-in-the-Loop (optional) / 人机协作（可选）
        expert_feedback = "Approve and execute the optimization."
        if HUMAN_IN_THE_LOOP:
            expert_feedback = _human_review(i, analysis)
            if expert_feedback == "SKIP":
                print("  Expert skipped this iteration.")
                continue

        # 6. Update feature library / 更新特征库
        extracted = analysis.get("extracted_features", [])
        if extracted:
            new_feats = [
                FeatureEntry(
                    feature_id=f"feat_iter{i}_{k}",
                    description=item["description"],
                    feature_type=item.get("type", "defect"),
                    source_id=orig.essay_id,
                )
                for k, item in enumerate(extracted)
            ]
            dm.save_features(new_feats)
            print(f"  Feature library updated (+{len(new_feats)} entries).")

        # 7. RAG retrieval + rule optimization / RAG 检索 + 规则优化
        defect_text = analysis.get("analysis", "")
        retrieved_features = dm.retrieve_features(defect_text, top_k=RAG_TOP_K)
        new_rule = run_roa(best_rule, analysis, retrieved_features, expert_feedback)
        new_rule.rule_id = f"v{i}"

        # 8. Score Dcal → grid search alphas → calibrated QWK → rollback decision
        # 在 Dcal 上评分 → 网格搜索 alphas → 校准 QWK → 回滚决策
        print("  Validating updated rule on Dcal (intra-loop calibration)...")
        d_cal_records = run_sa(new_rule, d_cal, anchor_text)
        raw_scores = [r.predicted_score for r in d_cal_records]
        true_scores_dcal = [e.true_score for e in d_cal]
        new_alphas = grid_search_alphas(raw_scores, true_scores_dcal, SCORE_MAX, CALIB_KAPPA, CALIB_ALPHA_GRID)
        cal_records = apply_calibration_to_records(d_cal_records, new_alphas)
        new_qwk = calculate_qwk(d_cal, cal_records)
        print(f"  Dcal calibrated QWK: {new_qwk:.4f}  "
              f"(delta = {new_qwk - best_rule.qwk_score:+.4f})  alphas = {new_alphas}")

        if new_qwk > best_rule.qwk_score + ROLLBACK_THRESHOLD:
            best_rule = new_rule
            best_rule.qwk_score = new_qwk
            best_alphas = new_alphas
            _save_rule(best_rule, is_final=False)
            print("  Rule accepted.")
        else:
            print("  No improvement — rule rolled back.")

    print(f"\n[Phase 3] Optimization complete.  "
          f"Final Dcal calibrated QWK = {best_rule.qwk_score:.4f}  "
          f"alphas = {best_alphas}")
    return best_rule, best_alphas


# ──────────────────────────────────────────────────────────────────
# Full DD-MARI Pipeline / 完整 DD-MARI 流程
# ──────────────────────────────────────────────────────────────────

def run_dd_mari(resume: bool = False) -> ScoringRule:
    """
    Entry point for the complete DD-MARI pipeline (Algorithm 1).
    完整 DD-MARI 流程的入口点（算法1）。

    Phase 1 : CDAP filter training pool (Dpool → Dclean) → build contrastive pairs
    阶段1   : CDAP 过滤训练池（Dpool → Dclean）→ 构建对比样本对
    Phase 2 : SGA cold start with Spearman on Dcal + initial Wasserstein calibration
    阶段2   : SGA 冷启动（Dcal 上的 Spearman 选优）+ 初始 Wasserstein 校准
    Phase 3 : Defect-driven iteration with intra-loop calibration on Dcal
    阶段3   : 基于 Dcal 循环内校准的缺陷驱动迭代优化
    Final   : CDAP inference filter + calibrated QWK on holdout
    最终    : CDAP 推理过滤 + 保留集上的校准 QWK
    """
    print("=" * 70)
    print("  DD-MARI: Defect-Driven Multi-Agent Rule Induction")
    print("=" * 70)

    # Load data / 加载数据
    print("\n[Data] Loading datasets...")
    gradient_essays     = dm.load_gradient_set()
    all_training_essays = dm.load_all_training_essays()
    test_essays         = dm.load_test_set()
    doc_content         = dm.load_standard_document()
    anchor_text         = dm.build_anchor_text(gradient_essays)

    # Split test set: Dcal (Phase 3 rollback) + holdout (final eval only)
    # 划分测试集：Dcal（阶段3回滚）+ 保留集（仅最终评估）
    d_cal, holdout_essays = dm.split_test_set(
        test_essays, iter_eval_size=ITER_EVAL_SIZE, seed=ITER_EVAL_SEED
    )

    print(f"  Gradient set    : {len(gradient_essays)} essays  (anchor building)")
    print(f"  Training pool   : {len(all_training_essays)} essays  (Dpool, before CDAP)")
    print(f"  Dcal (rollback) : {len(d_cal)} essays")
    print(f"  Hold-out (final): {len(holdout_essays)} essays")
    dm.load_feature_library()

    # ── Phase 1: CDAP filter training pool → Dclean → contrastive pairs
    # 阶段1：CDAP 过滤训练池 → Dclean → 对比样本对
    print("\n[Phase 1] CDAP Training Pool Filtering...")
    clean_essays   = cdap_filter(all_training_essays, anchor_text)
    training_pairs = dm.build_training_pairs(clean_essays)
    print(f"[Phase 1] Dclean: {len(clean_essays)} essays → {len(training_pairs)} contrastive pairs")

    # ── Phase 2: Cold Start (Spearman on Dcal, Wasserstein calibration)
    # 阶段2：冷启动（Dcal 上 Spearman 选优，Wasserstein 校准）
    start_iter  = 1
    best_alphas = list(CALIB_ALPHAS)

    if resume:
        rule = _load_latest_rule()
        if rule:
            print(f"\n[Resume] Loaded checkpoint: {rule.rule_id}  QWK={rule.qwk_score:.4f}")
            best_alphas = _load_saved_alphas()
            try:
                start_iter = int(rule.rule_id.lstrip("v")) + 1
            except ValueError:
                start_iter = 1
        else:
            print("[Resume] No checkpoint found — starting fresh cold start.")
            rule, best_alphas = cold_start(doc_content, d_cal, anchor_text, num_candidates=3)
    else:
        rule, best_alphas = cold_start(doc_content, d_cal, anchor_text, num_candidates=3)

    # ── Phase 3: Iterative Optimization with intra-loop calibration on Dcal
    # 阶段3：带循环内校准的迭代优化（仅基于 Dcal）
    final_rule, final_alphas = iterative_optimization(
        rule, best_alphas, training_pairs, d_cal, anchor_text,
        start_iteration=start_iter,
    )
    _save_rule(final_rule, is_final=True)
    _save_alphas(final_alphas)

    # ── Final Evaluation: CDAP inference filter + calibration on holdout
    # 最终评估：CDAP 推理过滤 + 保留集上的校准评估
    print("\n[Final Evaluation] Running Cognitive Purification Filter on hold-out set...")
    filtered_holdout = cdap_inference_filter(final_rule, holdout_essays, anchor_text)
    print(f"  Holdout: {len(holdout_essays)} → {len(filtered_holdout)} essays after CDAP filter")

    print("[Final Evaluation] Scoring filtered hold-out set...")
    holdout_records = run_sa(final_rule, filtered_holdout, anchor_text)
    cal_holdout_records = apply_calibration_to_records(holdout_records, final_alphas)
    holdout_qwk, details = calculate_qwk_detailed(filtered_holdout, cal_holdout_records)
    print(f"[Final Evaluation] Hold-out calibrated QWK = {holdout_qwk:.4f}")
    print(f"  Adjacent Accuracy = {details.get('adjacent_accuracy', 0):.4f}")
    print(f"  MAE               = {details.get('mae', 0):.4f}")
    print(f"  Sample Count      = {details.get('sample_count', 0)}")

    return final_rule
