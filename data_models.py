"""
DD-MARI Data Models / DD-MARI 数据模型
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime


class SampleType(str, Enum):
    GOLD      = "gold"        # high-quality anchor samples / 高质量锚样本
    NEGATIVE  = "negative"    # proximally-ranked hard negatives / 近邻分数的困难负样本
    GRADIENT  = "gradient"    # quality-stratified calibration set / 质量分层校准集
    TEST      = "test"        # hold-out evaluation set / 保留评估集


@dataclass
class Essay:
    """A single essay sample with its ground-truth score.
    单篇作文样本及其真实评分。
    """
    essay_id:      str
    sample_type:   SampleType
    essay_content: str
    true_score:    int          # ground-truth score on the 2–12 scale / 2–12 分制的真实评分
    parent_id:     Optional[str] = None  # for augmented samples, ID of the original / 增强样本对应的原始样本 ID


@dataclass
class ScoringRule:
    """A natural-language scoring rubric induced by the SGA / ROA.
    由 SGA / ROA 归纳生成的自然语言评分规则。
    """
    rule_id:      str
    rule_content: str           # full rubric text (human-readable) / 完整规则文本（人类可读）
    dimensions:   List[str]     # e.g. ["Ideas & Elaboration", "Organization", ...] / 评分维度列表
    qwk_score:    float = 0.0   # QWK on the calibration / test set (filled after evaluation) / 校准集/测试集上的 QWK（评估后填充）
    created_at:   str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ScoringRule":
        return ScoringRule(**d)


@dataclass
class EvaluationRecord:
    """Output produced by the Scoring Agent for a single essay.
    评分智能体对单篇作文的输出结果。
    """
    essay_id:         str
    predicted_score:  int
    dimension_scores: Dict[str, int]   # per-dimension breakdown / 各维度分数明细
    true_score:       int
    rationale:        str              # chain-of-thought justification / 链式思维推理过程
    is_anomaly:       bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureEntry:
    """A defect or quality signal extracted by the DDA and stored in the feature library.
    由 DDA 提取并存入特征库的缺陷或质量信号。
    """
    feature_id:    str
    description:   str
    feature_type:  str    # "defect" | "advantage" / "缺陷" | "优势"
    source_id:     str    # essay_id that triggered this feature / 触发该特征的样本 ID

    def to_dict(self) -> dict:
        return asdict(self)
