"""
DD-MARI Entry Point / DD-MARI 命令行入口

Usage / 用法
-----
# Full training run (cold start → iterative optimization → evaluation)
# 完整训练流程（冷启动 → 迭代优化 → 评估）
python main.py

# Resume from the latest checkpoint
# 从最新检查点恢复
python main.py --resume

# Run only the final evaluation on a test file (requires a trained rule)
# 仅对测试文件运行最终评估（需要已训练的规则）
python main.py --eval --test_file data/asap_set1_test.xlsx
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def run_training(resume: bool) -> None:
    from workflow import run_dd_mari
    run_dd_mari(resume=resume)


def run_evaluation(test_file: str, output_file: str = "output/eval_result.xlsx") -> None:
    """
    Load the final induced rule and evaluate it on any ASAP-format test file.
    加载最终归纳规则并在任意 ASAP 格式的测试文件上进行评估。
    Applies DBAP Cognitive Purification Filter and nonlinear calibration.
    应用 DBAP 认知净化过滤器和非线性校准。
    """
    from config import FINAL_RULE_FILE, FINAL_ALPHAS_FILE, SCORE_MIN, SCORE_MAX
    from data_models import ScoringRule, Essay, SampleType
    from data_manager import load_gradient_set, build_anchor_text
    from agents import run_sa
    from evaluator import calculate_qwk_detailed
    from cognitive_alignment import dbap_inference_filter, apply_calibration_to_records

    # Load rule / 加载规则
    if not FINAL_RULE_FILE.exists():
        print(f"[Error] No trained rule found at {FINAL_RULE_FILE}. Run training first.")
        sys.exit(1)
    with open(FINAL_RULE_FILE, "r", encoding="utf-8") as f:
        rule = ScoringRule.from_dict(json.load(f))
    print(f"[Eval] Loaded rule: {rule.rule_id}  (historical QWK={rule.qwk_score:.4f})")

    # Load calibration alphas / 加载校准参数
    from config import CALIB_ALPHAS
    if FINAL_ALPHAS_FILE.exists():
        with open(FINAL_ALPHAS_FILE, "r", encoding="utf-8") as f:
            alphas = json.load(f)
        print(f"[Eval] Loaded calibration alphas: {alphas}")
    else:
        alphas = list(CALIB_ALPHAS)
        print(f"[Eval] No saved alphas found — using defaults: {alphas}")

    # Load test data / 加载测试数据
    path = Path(test_file)
    if not path.exists():
        print(f"[Error] Test file not found: {test_file}")
        sys.exit(1)
    df = pd.read_excel(path)
    col_map = {c.lower(): c for c in df.columns}
    id_col    = col_map.get("essay_id")      or df.columns[0]
    text_col  = col_map.get("essay")         or df.columns[1]
    score_col = col_map.get("domain1_score") or df.columns[2]

    essays = []
    for _, row in df.iterrows():
        if pd.isna(row[text_col]):
            continue
        raw = int(row[score_col]) if pd.notna(row[score_col]) else 0
        score = max(SCORE_MIN, min(SCORE_MAX, raw))
        essays.append(Essay(str(row[id_col]), SampleType.TEST, str(row[text_col]), score))

    # Anchors / 锚点
    gradient = load_gradient_set()
    anchor_text = build_anchor_text(gradient)

    # Apply DBAP Cognitive Purification Filter / 应用 DBAP 认知净化过滤器
    print(f"[Eval] Running DBAP Cognitive Purification Filter on {len(essays)} essays...")
    filtered_essays = dbap_inference_filter(rule, essays, anchor_text)
    print(f"[Eval] Retained {len(filtered_essays)} / {len(essays)} essays after filter.")

    # Score filtered essays and apply calibration / 对过滤后的样本评分并应用校准
    print(f"[Eval] Scoring {len(filtered_essays)} essays...")
    records = run_sa(rule, filtered_essays, anchor_text)
    cal_records = apply_calibration_to_records(records, alphas)

    # Metrics / 指标
    qwk, details = calculate_qwk_detailed(filtered_essays, cal_records)
    print("\n" + "=" * 50)
    print("  Evaluation Results / 评估结果")
    print("=" * 50)
    print(f"  QWK (Quadratic Weighted Kappa) : {qwk:.4f}")
    print(f"  Adjacent Accuracy (±1)         : {details.get('adjacent_accuracy', 0):.4f}")
    print(f"  Exact Accuracy                 : {details.get('accuracy', 0):.4f}")
    print(f"  MAE                            : {details.get('mae', 0):.4f}")
    print(f"  Sample Count (after filter)    : {details.get('sample_count', 0)}")
    print("=" * 50)

    # Export / 导出
    rec_map = {r.essay_id: r for r in cal_records}
    rows = []
    for e in filtered_essays:
        r = rec_map.get(e.essay_id)
        rows.append({
            "essay_id":        e.essay_id,
            "true_score":      e.true_score,
            "predicted_score": r.predicted_score if r else None,
            "error":           (r.predicted_score - e.true_score) if r else None,
            "dimension_scores": str(r.dimension_scores) if r else "",
            "rationale":       (r.rationale[:200] + "...") if r and len(r.rationale) > 200 else (r.rationale if r else ""),
        })
    pd.DataFrame(rows).to_excel(output_file, index=False)
    print(f"\n[Eval] Detailed results saved → {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="DD-MARI: Defect-Driven Multi-Agent Rule Induction Framework\nDD-MARI：缺陷驱动的多智能体规则归纳框架",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in output/rule_history.json\n从 output/rule_history.json 中的最新检查点恢复训练",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation only (requires a trained rule in output/final_rule.json)\n仅运行评估（需要 output/final_rule.json 中的已训练规则）",
    )
    parser.add_argument(
        "--test_file", type=str, default="data/asap_set1_test.xlsx",
        help="Path to the test set Excel file (used with --eval)\n测试集 Excel 文件路径（与 --eval 一起使用）",
    )
    parser.add_argument(
        "--output", type=str, default="output/eval_result.xlsx",
        help="Output path for the evaluation Excel report\n评估 Excel 报告的输出路径",
    )
    args = parser.parse_args()

    if args.eval:
        run_evaluation(args.test_file, args.output)
    else:
        run_training(resume=args.resume)


if __name__ == "__main__":
    main()
