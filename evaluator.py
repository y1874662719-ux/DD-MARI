"""
DD-MARI Evaluator / DD-MARI 评估器
Computes Quadratic Weighted Kappa (QWK) and auxiliary metrics.
计算二次加权 Kappa（QWK）及辅助指标。
QWK is the primary metric used in the ASAP benchmark.
QWK 是 ASAP 基准测试的主要评估指标。
"""

import math
from typing import List, Dict, Tuple

import numpy as np

from config import SCORE_MIN, SCORE_MAX
from data_models import Essay, EvaluationRecord


def calculate_qwk(essays: List[Essay], records: List[EvaluationRecord]) -> float:
    """Return QWK; call calculate_qwk_detailed for additional metrics.
    返回 QWK 值；调用 calculate_qwk_detailed 获取更多指标。
    """
    qwk, _ = calculate_qwk_detailed(essays, records)
    return qwk


def calculate_qwk_detailed(
    essays: List[Essay],
    records: List[EvaluationRecord],
) -> Tuple[float, Dict]:
    """
    Compute Quadratic Weighted Kappa and auxiliary metrics.
    计算二次加权 Kappa 及辅助指标。

    Returns / 返回
    -------
    (qwk, details_dict)
      details_dict keys / 键: qwk, accuracy, mae, exact_match_rate, sample_count
    """
    record_map = {r.essay_id: r.predicted_score for r in records}

    y_true, y_pred = [], []
    for essay in essays:
        if essay.essay_id in record_map:
            y_true.append(essay.true_score)
            y_pred.append(record_map[essay.essay_id])

    n = len(y_true)
    if n == 0:
        return 0.0, {"qwk": 0.0, "accuracy": 0.0, "mae": 0.0,
                     "exact_match_rate": 0.0, "sample_count": 0}

    num_labels = SCORE_MAX - SCORE_MIN + 1
    label_range = range(SCORE_MIN, SCORE_MAX + 1)

    # Build weight matrix / 构建权重矩阵
    W = np.zeros((num_labels, num_labels))
    for i, vi in enumerate(label_range):
        for j, vj in enumerate(label_range):
            W[i, j] = (vi - vj) ** 2 / (num_labels - 1) ** 2

    # Observed and expected matrices / 观测矩阵与期望矩阵
    O = np.zeros((num_labels, num_labels))
    for yt, yp in zip(y_true, y_pred):
        i = yt - SCORE_MIN
        j = yp - SCORE_MIN
        if 0 <= i < num_labels and 0 <= j < num_labels:
            O[i, j] += 1

    hist_true = np.array([y_true.count(v) for v in label_range], dtype=float)
    hist_pred = np.array([y_pred.count(v) for v in label_range], dtype=float)
    E = np.outer(hist_true, hist_pred) / n

    O_norm = O / n
    E_norm = E / n if E.sum() > 0 else E

    num = np.sum(W * O_norm)
    den = np.sum(W * E_norm)
    qwk = 1.0 - num / den if den > 1e-10 else 0.0

    mae = float(np.mean([abs(yt - yp) for yt, yp in zip(y_true, y_pred)]))
    exact_matches = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    adjacent_matches = sum(abs(yt - yp) <= 1 for yt, yp in zip(y_true, y_pred))

    return qwk, {
        "qwk": round(qwk, 4),
        "accuracy": round(exact_matches / n, 4),
        "adjacent_accuracy": round(adjacent_matches / n, 4),
        "mae": round(mae, 4),
        "sample_count": n,
    }
