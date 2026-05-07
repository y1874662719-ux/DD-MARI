"""
DD-MARI Main Workflow / DD-MARI 主工作流
Implements Algorithm 1 from the paper:
实现论文中的算法1：

  Phase 1 — Contrastive Sample Construction / 阶段1 — 对比样本构建
  Phase 2 — Spearman-Guided Cold-Start Initialization / 阶段2 — Spearman 引导的冷启动初始化
  Phase 3 — RAG-Enhanced Iterative Optimization / 阶段3 — RAG 增强的迭代优化
  Phase 4 — Cognitive-Aligned Execution  (subjective tasks only)
             认知对齐执行（仅用于主观任务）
"""

import json
from typing import Optional, List, Tuple
from scipy.stats import spearmanr

from config import (
    MAX_ITERATIONS, BATCH_SIZE, ROLLBACK_THRESHOLD,
    RAG_TOP_K, HUMAN_IN_THE_LOOP, TASK_MODE,
    RULE_HISTORY_FILE, FINAL_RULE_FILE,
    ITER_EVAL_SIZE, ITER_EVAL_SEED,
)
from data_models import Essay, ScoringRule, FeatureEntry
from agents import run_sga, run_sa, run_dda, run_roa
from evaluator import calculate_qwk, calculate_qwk_detailed
import data_manager as dm


# ──────────────────────────────────────────────────────────────────
# Persistence helpers / 持久化辅助函数
# ──────────────────────────────────────────────────────────────────

def _save_rule(rule: ScoringRule, is_final: bool = False) -> None:
    # Append to history / 追加到历史记录
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


def _load_latest_rule() -> Optional[ScoringRule]:
    if not RULE_HISTORY_FILE.exists():
        return None
    with open(RULE_HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    return ScoringRule.from_dict(history[-1]) if history else None


# ──────────────────────────────────────────────────────────────────
# Phase 2 — Spearman-Guided Cold-Start Initialization
# 阶段2 — Spearman 引导的冷启动初始化
# ──────────────────────────────────────────────────────────────────

def cold_start(doc_content: str,
               gradient_essays: List[Essay],
               d_cal: List[Essay],
               anchor_text: str,
               num_candidates: int = 3) -> ScoringRule:
    """
    Generate N candidate rules via SGA, evaluate each on the quality-stratified
    calibration set using Spearman rank correlation (Eq. 5–6), select the best
    candidate, then compute its initial Dcal QWK as the Phase 3 rollback baseline.
    通过 SGA 生成 N 个候选规则，使用 Spearman 秩相关（公式5–6）在质量分层校准集上评估每个候选，
    选择最优候选，然后在 Dcal 上计算初始 QWK 作为阶段3回滚基线。
    """
    print("\n[Phase 2] Cold-Start Initialization")
    candidates = run_sga(doc_content, num_variants=num_candidates)

    best_rule: Optional[ScoringRule] = None
    best_rho = -1.0

    gold_ranks = [e.true_score for e in gradient_essays]   # Y_gold / 真实排名

    for idx, rule in enumerate(candidates):
        records = run_sa(rule, gradient_essays, anchor_text)
        pred_map = {r.essay_id: r.predicted_score for r in records}
        pred_ranks = [pred_map.get(e.essay_id, 0) for e in gradient_essays]

        rho, _ = spearmanr(gold_ranks, pred_ranks)          # Eq. (5) / 公式(5)
        print(f"  Candidate {idx + 1}: Spearman rho = {rho:.4f}")

        if rho > best_rho:
            best_rho = rho
            best_rule = rule

    print(f"[Phase 2] Best candidate (rho = {best_rho:.4f}) selected as R0.")

    # Compute initial Dcal QWK — must match Phase 3 rollback metric
    # 计算初始 Dcal QWK —— 必须与阶段3回滚指标一致
    print(f"[Phase 2] Computing initial Dcal QWK for R0 ({len(d_cal)} essays)...")
    d_cal_records = run_sa(best_rule, d_cal, anchor_text)
    qwk = calculate_qwk(d_cal, d_cal_records)
    best_rule.qwk_score = qwk
    best_rule.rule_id = "v0"
    print(f"[Phase 2] Initial Dcal QWK = {qwk:.4f}")

    _save_rule(best_rule)
    return best_rule


# ──────────────────────────────────────────────────────────────────
# Phase 3 — RAG-Enhanced Iterative Optimization
# 阶段3 — RAG 增强的迭代优化
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
    training_pairs: List[Tuple[Essay, List[Essay]]],
    d_cal: List[Essay],
    anchor_text: str,
    start_iteration: int = 1,
) -> ScoringRule:
    """
    Defect-driven RAG-enhanced optimization loop (Algorithm 1, Phase 3).
    缺陷驱动的 RAG 增强优化循环（算法1，阶段3）。

    Rollback decisions are made on Dcal (100 randomly sampled essays),
    NOT on the hold-out test set, matching Algorithm 1 line 30 in the paper.
    回滚决策在 Dcal（100 篇随机抽样作文）上进行，而非保留测试集，对应论文算法1第30行。

    For each iteration / 每轮迭代:
      1. Sample a mini-batch of (original, augmented) pairs
         采样一个（原始，增强）样本对的小批量
      2. Score with current rule  → detect scoring inversions (Eq. 4)
         用当前规则评分 → 检测评分倒置（公式4）
      3. DDA diagnoses root-cause features
         DDA 诊断根因特征
      4. [Optional] Human validates features
         [可选] 人工验证特征
      5. RAG retrieves top-k features from library
         RAG 从特征库检索 top-k 特征
      6. ROA proposes an updated rule
         ROA 提出更新后的规则
      7. Validate on Dcal → rollback if QWK does not improve by τ
         在 Dcal 上验证 → 若 QWK 未提升 τ 则回滚
    """
    print("\n[Phase 3] Iterative Optimization Started")
    print(f"  Rollback evaluated on Dcal ({len(d_cal)} essays).")
    best_rule = initial_rule

    for i in range(start_iteration, MAX_ITERATIONS + 1):
        print(f"\n  --- Iteration {i}/{MAX_ITERATIONS}  "
              f"(Best Dcal QWK: {best_rule.qwk_score:.4f}) ---")

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

        aug = anomalies[0]  # process first anomaly / 处理第一个异常
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

        # 8. Validate on Dcal (rollback if no improvement — Algorithm 1 line 30)
        # 在 Dcal 上验证（无提升则回滚 —— 算法1第30行）
        print("  Validating updated rule on Dcal...")
        d_cal_records = run_sa(new_rule, d_cal, anchor_text)
        new_qwk = calculate_qwk(d_cal, d_cal_records)
        print(f"  Dcal QWK: {new_qwk:.4f}  (delta = {new_qwk - best_rule.qwk_score:+.4f})")

        if new_qwk > best_rule.qwk_score + ROLLBACK_THRESHOLD:
            best_rule = new_rule
            best_rule.qwk_score = new_qwk
            _save_rule(best_rule, is_final=False)
            print("  Rule accepted.")
        else:
            print("  No improvement — rule rolled back.")

    print(f"\n[Phase 3] Optimization complete.  Final Dcal QWK = {best_rule.qwk_score:.4f}")
    return best_rule


# ──────────────────────────────────────────────────────────────────
# Phase 4 — Cognitive-Aligned Execution / 阶段4 — 认知对齐执行
# ──────────────────────────────────────────────────────────────────

def execute_with_alignment(rule: ScoringRule,
                           test_essays: List[Essay],
                           anchor_text: str):
    """Run Phase 4 if task mode is 'unstructured'.
    若任务模式为 'unstructured' 则执行阶段4。
    """
    from cognitive_alignment import cognitively_aligned_scoring
    from evaluator import calculate_qwk_detailed

    records = cognitively_aligned_scoring(rule, test_essays, anchor_text)
    qwk, details = calculate_qwk_detailed(test_essays, records)
    print(f"\n[Phase 4] Cognitively-Aligned QWK = {qwk:.4f}")
    print(f"  Adjacent Accuracy = {details.get('adjacent_accuracy', 0):.4f}")
    print(f"  MAE               = {details.get('mae', 0):.4f}")
    return records, qwk


# ──────────────────────────────────────────────────────────────────
# Full DD-MARI Pipeline / 完整 DD-MARI 流程
# ──────────────────────────────────────────────────────────────────

def run_dd_mari(resume: bool = False) -> ScoringRule:
    """
    Entry point for the complete DD-MARI pipeline.
    完整 DD-MARI 流程的入口点。

    Parameters / 参数
    ----------
    resume : if True, loads the latest rule from rule_history.json
             and skips cold-start + completed iterations.
             若为 True，从 rule_history.json 加载最新规则，跳过冷启动和已完成的迭代。

    Data split / 数据划分
    ----------
    Dcal    : 100 randomly sampled essays used for Phase 3 rollback decisions only.
              100 篇随机抽样作文，仅用于阶段3回滚决策。
    Holdout : remaining essays (~1,643) used only for cold-start baseline and
              final evaluation — never touched during optimization.
              剩余约1,643篇作文，仅用于冷启动基线和最终评估，优化过程中绝不使用。
    """
    print("=" * 70)
    print("  DD-MARI: Defect-Driven Multi-Agent Rule Induction")
    print("=" * 70)

    # Load data / 加载数据
    print("\n[Data] Loading datasets...")
    gradient_essays = dm.load_gradient_set()
    training_pairs  = dm.load_training_pairs()
    test_essays     = dm.load_test_set()
    doc_content     = dm.load_standard_document()
    anchor_text     = dm.build_anchor_text(gradient_essays)

    # Split test set: Dcal (rollback) + holdout (final eval only) — paper §III-C
    # 划分测试集：Dcal（回滚）+ 保留集（仅最终评估）—— 论文§III-C
    d_cal, holdout_essays = dm.split_test_set(
        test_essays, iter_eval_size=ITER_EVAL_SIZE, seed=ITER_EVAL_SEED
    )

    print(f"  Gradient set    : {len(gradient_essays)} essays")
    print(f"  Training pairs  : {len(training_pairs)} groups")
    print(f"  Dcal (rollback) : {len(d_cal)} essays")
    print(f"  Hold-out (final): {len(holdout_essays)} essays")
    dm.load_feature_library()   # pre-load feature library / 预加载特征库

    # ── Phase 2: Cold Start (Spearman on gradient, baseline QWK on hold-out)
    # 阶段2：冷启动（在梯度集上计算 Spearman，在保留集上计算基线 QWK）
    start_iter = 1
    if resume:
        rule = _load_latest_rule()
        if rule:
            print(f"\n[Resume] Loaded checkpoint: {rule.rule_id}  QWK={rule.qwk_score:.4f}")
            try:
                start_iter = int(rule.rule_id.lstrip("v")) + 1
            except ValueError:
                start_iter = 1
        else:
            print("[Resume] No checkpoint found — starting fresh cold start.")
            rule = cold_start(doc_content, gradient_essays, d_cal,
                              anchor_text, num_candidates=3)
    else:
        rule = cold_start(doc_content, gradient_essays, d_cal,
                          anchor_text, num_candidates=3)

    # ── Phase 3: Iterative Optimization on Dcal only
    # 阶段3：仅在 Dcal 上进行迭代优化
    final_rule = iterative_optimization(
        rule, training_pairs, d_cal, anchor_text,
        start_iteration=start_iter,
    )
    _save_rule(final_rule, is_final=True)

    # ── Final Evaluation on full hold-out set (never seen during optimization)
    # 在完整保留集上进行最终评估（优化过程中从未使用）
    print("\n[Final Evaluation] Scoring hold-out test set...")
    holdout_records = run_sa(final_rule, holdout_essays, anchor_text)
    holdout_qwk, details = calculate_qwk_detailed(holdout_essays, holdout_records)
    print(f"[Final Evaluation] Hold-out QWK = {holdout_qwk:.4f}")
    print(f"  Adjacent Accuracy = {details.get('adjacent_accuracy', 0):.4f}")
    print(f"  MAE               = {details.get('mae', 0):.4f}")

    # ── Phase 4: Cognitive Alignment (subjective tasks only, on hold-out set)
    # 阶段4：认知对齐（仅用于主观任务，在保留集上执行）
    if TASK_MODE == "unstructured":
        execute_with_alignment(final_rule, holdout_essays, anchor_text)

    return final_rule
