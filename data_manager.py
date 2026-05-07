"""
DD-MARI Data Manager / DD-MARI 数据管理器
Handles loading, preprocessing, and serving data for all pipeline phases.
负责所有流程阶段的数据加载、预处理和提供。

Expected data directory layout (data/):
预期数据目录结构（data/）：
  asap_set1_gradient.xlsx   — quality-stratified calibration set   (10–20 essays, diverse scores)
                              质量分层校准集（10–20 篇，分数多样）
  asap_set1_training.xlsx   — training pool                        (10 gold + 30 hard-negatives)
                              训练池（10 篇金标准 + 30 篇困难负样本）
  asap_set1_test.xlsx       — hold-out test set                    (e.g. 1643 essays)
                              保留测试集（约 1643 篇）
  Set1_standard.docx        — official ASAP Set 1 scoring document
                              ASAP Set 1 官方评分标准文档

Excel column names (case-insensitive) / Excel 列名（不区分大小写）：
  essay_id | essay | domain1_score
"""

import json
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import docx

from config import (
    GRADIENT_DATA_FILE, TRAINING_DATA_FILE, TEST_DATA_FILE,
    STANDARD_DOC_FILE, FEATURE_LIB_FILE, SCORE_MIN, SCORE_MAX,
)
from data_models import Essay, SampleType, FeatureEntry


# ──────────────────────────────────────────────────────────────────
# Column-name resolver (tolerant of different naming conventions)
# 列名解析器（兼容不同命名规范）
# ──────────────────────────────────────────────────────────────────
def _resolve_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cm = {c.lower(): c for c in df.columns}
    id_col    = cm.get("essay_id") or cm.get("sample_id") or cm.get("id") or df.columns[0]
    text_col  = cm.get("essay")    or cm.get("essay_content") or cm.get("content") or df.columns[1]
    score_col = cm.get("domain1_score") or cm.get("score") or cm.get("true_score") or df.columns[2]
    return id_col, text_col, score_col


def _df_to_essays(df: pd.DataFrame, sample_type: SampleType) -> List[Essay]:
    """
    Convert a DataFrame to a list of Essay objects.
    将 DataFrame 转换为 Essay 对象列表。

    Scores are read as-is and clamped to [SCORE_MIN, SCORE_MAX].
    The ASAP Set 1 combined score (sum of two raters, range 2–12) is used directly.
    分数直接读取并截断至 [SCORE_MIN, SCORE_MAX]。
    ASAP Set 1 综合分（两位评分者之和，范围 2–12）直接使用，无需缩放。
    """
    id_col, text_col, score_col = _resolve_cols(df)

    essays = []
    for _, row in df.iterrows():
        if pd.isna(row[text_col]):
            continue
        raw_score = int(row[score_col]) if pd.notna(row[score_col]) else 0
        score = max(SCORE_MIN, min(SCORE_MAX, raw_score))
        essays.append(Essay(
            essay_id=str(row[id_col]),
            sample_type=sample_type,
            essay_content=str(row[text_col]),
            true_score=score,
        ))
    return essays


# ──────────────────────────────────────────────────────────────────
# Public data-loading API / 公共数据加载接口
# ──────────────────────────────────────────────────────────────────

def load_gradient_set() -> List[Essay]:
    """Quality-stratified calibration set used for Spearman-guided cold start.
    用于 Spearman 引导冷启动的质量分层校准集。
    """
    df = pd.read_excel(GRADIENT_DATA_FILE)
    return _df_to_essays(df, SampleType.GRADIENT)


def load_training_pairs() -> List[Tuple[Essay, List[Essay]]]:
    """
    Training pool: groups of (original, [augmented_1, ..., augmented_k]).
    训练池：由（原始样本，[增强样本_1, ..., 增强样本_k]）构成的组。
    Each group is one contrastive sample unit.
    每组为一个对比样本单元。

    Pairing logic / 配对逻辑:
      - If 'group_id' column exists → use it as the group key
        若存在 'group_id' 列 → 直接用作组键
      - Else if 'sample_type' column exists → use it for gold/negative detection;
        group key = numeric segment of essay_id (e.g. orig_3 → "3", neg_3_1 → "3")
        若存在 'sample_type' 列 → 用于金/负样本判断；组键取 essay_id 中的数字段
      - Else → fully infer from essay_id string (legacy fallback)
        否则 → 完全从 essay_id 字符串推断（兼容旧格式）
    """
    df = pd.read_excel(TRAINING_DATA_FILE)
    id_col, text_col, score_col = _resolve_cols(df)
    cm = {c.lower(): c for c in df.columns}

    has_group_col   = "group_id"    in cm
    has_stype_col   = "sample_type" in cm

    pairs: Dict[str, Tuple[Optional[Essay], List[Essay]]] = {}

    for _, row in df.iterrows():
        if pd.isna(row[text_col]):
            continue
        raw_score = int(row[score_col]) if pd.notna(row[score_col]) else 0
        score = max(SCORE_MIN, min(SCORE_MAX, raw_score))
        essay_id = str(row[id_col])

        if has_group_col:
            # Explicit group_id column / 显式 group_id 列
            gid    = str(row[cm["group_id"]])
            is_aug = str(row.get(cm.get("sample_type", ""), "")).lower() in (
                         "aug", "augmented", "negative")
        elif has_stype_col:
            # sample_type column present; derive group from numeric part of essay_id
            # 存在 sample_type 列；从 essay_id 的数字段提取组号
            # e.g. orig_3 → parts[1]="3", neg_3_1 → parts[1]="3"
            is_aug = str(row[cm["sample_type"]]).lower() in (
                         "aug", "augmented", "negative")
            parts = essay_id.split("_")
            gid   = parts[1] if len(parts) >= 2 else parts[0]
        else:
            # Legacy fallback: infer everything from essay_id string
            # 兼容旧格式：完全从 essay_id 字符串推断
            is_aug = any(k in essay_id.lower() for k in ("aug", "enhanced", "negative", "neg"))
            parts  = essay_id.split("_")
            gid    = parts[1] if len(parts) >= 2 else parts[0]

        if gid not in pairs:
            pairs[gid] = (None, [])

        essay = Essay(
            essay_id=essay_id,
            sample_type=SampleType.NEGATIVE if is_aug else SampleType.GOLD,
            essay_content=str(row[text_col]),
            true_score=score,
        )
        if is_aug:
            pairs[gid][1].append(essay)
        else:
            pairs[gid] = (essay, pairs[gid][1])

    return [(orig, augs) for orig, augs in pairs.values() if orig is not None and augs]


def load_test_set() -> List[Essay]:
    """Hold-out test set for final QWK evaluation.
    用于最终 QWK 评估的保留测试集。
    """
    df = pd.read_excel(TEST_DATA_FILE)
    return _df_to_essays(df, SampleType.TEST)


def load_standard_document() -> str:
    """Read the official scoring guidelines Word document as plain text.
    将官方评分标准 Word 文档读取为纯文本。
    """
    doc = docx.Document(STANDARD_DOC_FILE)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def build_anchor_text(gradient_essays: List[Essay]) -> str:
    """
    Select one representative anchor essay per score band and format
    them as a reference string for the Scoring Agent.
    从每个分段选取一篇代表性锚样本，格式化为评分智能体的参考字符串。
    """
    target_scores = [SCORE_MIN, 4, 6, 8, 10, SCORE_MAX]
    anchors: Dict[int, str] = {}

    sorted_by_length = sorted(gradient_essays, key=lambda e: len(e.essay_content))
    for t in target_scores:
        match = next((e for e in sorted_by_length if e.true_score == t), None)
        if match:
            content = match.essay_content
            if len(content) > 600:          # truncate very long anchors to save tokens / 截断过长锚样本以节省 token
                content = content[:300] + "\n...[OMITTED]...\n" + content[-200:]
            anchors[t] = content

    return "\n".join(
        f"===[ Anchor {score} pts ]===\n{text}" for score, text in sorted(anchors.items())
    )


def split_test_set(
    test_essays: List[Essay],
    iter_eval_size: int = 100,
    seed: int = 42,
) -> Tuple[List[Essay], List[Essay]]:
    """
    Randomly split the test set into Dcal (for Phase 3 rollback) and the hold-out
    test set (for final evaluation only), matching paper §III-C.
    将测试集随机划分为 Dcal（用于阶段3回滚）和保留测试集（仅用于最终评估），对应论文§III-C。

    Returns / 返回
    -------
    (d_cal, holdout)
      d_cal   : iter_eval_size randomly sampled essays for rollback decisions
                随机抽取 iter_eval_size 篇作文用于回滚决策
      holdout : remaining essays for final evaluation only
                剩余作文仅用于最终评估
    """
    shuffled = test_essays.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    d_cal   = shuffled[:iter_eval_size]
    holdout = shuffled[iter_eval_size:]
    print(f"[Data] Test set split → Dcal: {len(d_cal)} essays (rollback) | "
          f"Hold-out: {len(holdout)} essays (final eval)")
    return d_cal, holdout


def sample_batch(pairs: List[Tuple[Essay, List[Essay]]], size: int = 1) -> List[Tuple[Essay, List[Essay]]]:
    """Randomly draw a mini-batch from the training pair pool.
    从训练样本对池中随机抽取一个小批量。
    """
    return random.sample(pairs, min(size, len(pairs)))


# ──────────────────────────────────────────────────────────────────
# Feature Library — persistent RAG knowledge base
# 特征库 —— 持久化 RAG 知识库
# ──────────────────────────────────────────────────────────────────

_feature_library: List[FeatureEntry] = []


def load_feature_library() -> List[FeatureEntry]:
    global _feature_library
    if FEATURE_LIB_FILE.exists():
        with open(FEATURE_LIB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _feature_library = [FeatureEntry(**d) for d in data]
    return _feature_library


def save_features(new_features: List[FeatureEntry]) -> None:
    global _feature_library
    _feature_library.extend(new_features)
    with open(FEATURE_LIB_FILE, "w", encoding="utf-8") as f:
        json.dump([feat.to_dict() for feat in _feature_library], f, indent=2, ensure_ascii=False)


def retrieve_features(defect_report: str, top_k: int = 5) -> str:
    """
    RAG retrieval — returns top-k features most relevant to the current defect report
    via cosine similarity on TF-IDF vectors (lightweight, no GPU needed).
    RAG 检索 —— 通过 TF-IDF 余弦相似度返回与当前缺陷报告最相关的 top-k 特征（轻量级，无需 GPU）。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    lib = load_feature_library()
    if not lib:
        return "No features in library yet."

    descriptions = [f.description for f in lib]
    corpus = [defect_report] + descriptions

    try:
        vec = TfidfVectorizer().fit_transform(corpus)
        sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        selected = [lib[i] for i in top_idx]
    except Exception:
        selected = lib[:top_k]

    lines = [f"- [{f.feature_type.upper()}] {f.description}" for f in selected]
    return "\n".join(lines)
