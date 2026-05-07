"""
DD-MARI Agents / DD-MARI 智能体模块
Implements the four independent agents described in Section III of the paper:
实现论文第三节描述的四个独立智能体：
  SGA  – Standard Generation Agent     (Phase 2: cold-start rule generation)
         标准生成智能体（阶段2：冷启动规则生成）
  SA   – Scoring Agent                 (Phases 2 & 3: rule execution)
         评分智能体（阶段2和3：规则执行）
  DDA  – Defect Diagnosis Agent        (Phase 3: root-cause analysis)
         缺陷诊断智能体（阶段3：根因分析）
  ROA  – Rule Optimization Agent       (Phase 3: rubric refinement)
         规则优化智能体（阶段3：评分标准精化）

Prompt templates follow the paper's Appendix (Figures 8–11) exactly.
提示词模板严格遵循论文附录（图8–11）。
"""

import json
import re
import concurrent.futures
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME,
    SCORING_THREADS, SCORE_MIN, SCORE_MAX,
)
from data_models import Essay, ScoringRule, EvaluationRecord, FeatureEntry


# ──────────────────────────────────────────────────────────────────
# LLM factory helpers / LLM 工厂辅助函数
# ──────────────────────────────────────────────────────────────────

def _scorer_llm() -> ChatOpenAI:
    """Deterministic LLM used by the Scoring Agent (temp = 0).
    评分智能体使用的确定性 LLM（温度参数 = 0）。
    """
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,
    )


def _reasoner_llm() -> ChatOpenAI:
    """Creative LLM used by SGA / DDA / ROA (temp = 0.7).
    SGA / DDA / ROA 使用的创造性 LLM（温度参数 = 0.7）。
    """
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.7,
    )


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response.
    从 LLM 响应中提取第一个 JSON 对象。
    Handles markdown code fences and trailing text from verbose LLMs.
    处理 LLM 在 JSON 前后附加 markdown 代码块或多余文字的情况。
    """
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return json.loads(text)


def _parse_json_list(text: str) -> list:
    """Extract the first JSON array from an LLM response.
    从 LLM 响应中提取第一个 JSON 数组。
    """
    text = re.sub(r"```json|```", "", text).strip()
    if "[" in text:
        return json.loads(text[text.find("[") : text.rfind("]") + 1])
    return json.loads(text)


# ══════════════════════════════════════════════════════════════════
# Phase 2 — Standard Generation Agent (SGA)    [Fig. 8 in paper]
# 阶段2 — 标准生成智能体（SGA）                  [论文图8]
# ══════════════════════════════════════════════════════════════════

SGA_SYSTEM_PROMPT = """You are a senior expert in educational measurement.
Task: Read the official document content and design a structured scoring rubric.

[Core Principle]: The scoring must be Holistic and Dimension-based.
Hard rules targeting specific content are strictly prohibited.

[Official Dimensions Limit]:
Based on the "Typical elements" in the document, your rules must and can only include \
the following four dimensions:
1. Ideas & Elaboration  (Corresponding to document: details, reasons, support, elaboration, list-like)
2. Organization         (Corresponding to document: organization, structure)
3. Fluency & Language   (Corresponding to document: fluent, transitional language, awkward, fragmented)
4. Audience Awareness   (Corresponding to document: awareness of audience)

[Score Design Requirements]:
1. Target total score range: 2-12 points.
2. Please allocate points for each dimension, ensuring the sum equals 12 points.
3. Key: For each score in each dimension, you can quote or paraphrase the descriptions \
from Score 1 to Score 6 in the official document (e.g., "unelaborated", "somewhat developed").

Output format (JSON List):
[
  {
    "variation_name": "Standard Dimension Version",
    "dimensions": ["Ideas & Elaboration", "Organization", "Fluency & Language", "Audience Awareness"],
    "rule_content": "1. Ideas & Elaboration (0-4): \\n   - 4 Points: Fully elaborated reasons with specific details...\\n   - 2 Points: Only general reasons with unelaborated or list-like details...\\n..."
  }
]"""

SGA_USER_PROMPT = "Official document content:\n{doc_content}"


def run_sga(doc_content: str, num_variants: int = 3) -> List[ScoringRule]:
    """
    Standard Generation Agent — generates N candidate scoring rubrics
    from the official guidelines document.
    标准生成智能体 —— 从官方评分标准文档生成 N 个候选评分规则。

    Args / 参数:
        doc_content:  text of the official scoring document / 官方评分文档的文本内容
        num_variants: number of rubric variants to generate (N in paper) / 生成的规则变体数（论文中的 N）

    Returns / 返回:
        list of ScoringRule candidates / 候选 ScoringRule 列表
    """
    print(f"[SGA] Generating {num_variants} initial rule candidates from the official document...")
    llm = _reasoner_llm()
    system = SGA_SYSTEM_PROMPT + f"\n\nGenerate exactly {num_variants} distinct variants."
    user = SGA_USER_PROMPT.format(doc_content=doc_content)

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    data = _parse_json_list(resp.content)

    rules = []
    for i, item in enumerate(data):
        rules.append(ScoringRule(
            rule_id=f"init_v{i + 1}",
            rule_content=item["rule_content"],
            dimensions=item.get("dimensions", ["Ideas & Elaboration", "Organization",
                                               "Fluency & Language", "Audience Awareness"]),
        ))
    print(f"[SGA] Generated {len(rules)} candidate rules.")
    return rules


# ══════════════════════════════════════════════════════════════════
# Phases 2 & 3 — Scoring Agent (SA)           [Fig. 9 in paper]
# 阶段2和3 — 评分智能体（SA）                   [论文图9]
# ══════════════════════════════════════════════════════════════════

SA_SYSTEM_PROMPT = """You are a professional essay scorer.
Scoring Logic: Analytic Scoring. The evaluation should be based on a Holistic Impression, \
rather than searching for specific keywords.

[Scoring Rubric]:
{rule_content}

[Calibration Anchors]:
{anchor_text}

[Scoring Steps]:
1. Dimension Scoring: Score each dimension separately according to the rubric.
   Total Score = Dimension 1 Score + Dimension 2 Score + Dimension 3 Score + Dimension 4 Score.
2. You will see numerous tags starting with @ in the essay, such as @CAPS1, @ORGANIZATION2, @NUM1, etc.
   These are privacy desensitization markers of the dataset. It is strictly prohibited to deduct points
   because the essay contains these tags. When evaluating "language conventions," you must completely
   ignore these tags.

Please output your thought process (Rationale) first, and finally output the JSON result."""

SA_USER_PROMPT = """Essay to be evaluated:
{essay_content}

Please begin your analysis, and output only a JSON at the end:
{{
  "rationale": "Scoring rationale based on holistic impression...",
  "dimension_scores": {{ "Dimension Name": Score, ... }},
  "total_score": Total Score (Integer)
}}"""


def _score_single(rule: ScoringRule, essay: Essay, anchor_text: str) -> EvaluationRecord:
    """Score a single essay using the current rule.
    使用当前规则对单篇作文进行评分。
    """
    llm = _scorer_llm()
    system = SA_SYSTEM_PROMPT.format(
        rule_content=rule.rule_content,
        anchor_text=anchor_text,
    )
    user = SA_USER_PROMPT.format(essay_content=essay.essay_content)
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = _parse_json(resp.content)
        return EvaluationRecord(
            essay_id=essay.essay_id,
            predicted_score=int(data["total_score"]),
            dimension_scores=data.get("dimension_scores", {}),
            true_score=essay.true_score,
            rationale=data.get("rationale", ""),
        )
    except Exception as e:
        print(f"[SA] Scoring error for {essay.essay_id}: {e}")
        return EvaluationRecord(essay.essay_id, SCORE_MIN, {}, essay.true_score, "Error")


def run_sa(rule: ScoringRule, essays: List[Essay], anchor_text: str) -> List[EvaluationRecord]:
    """
    Scoring Agent — evaluates a batch of essays using the given rule.
    评分智能体 —— 使用给定规则对一批作文进行评估。
    Runs in parallel for efficiency.
    并行执行以提高效率。

    Args / 参数:
        rule:        current scoring rule / 当前评分规则
        essays:      list of Essay samples to score / 待评分的 Essay 样本列表
        anchor_text: formatted calibration anchor string / 格式化的校准锚点字符串

    Returns / 返回:
        list of EvaluationRecord results / EvaluationRecord 结果列表
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=SCORING_THREADS) as pool:
        futures = [pool.submit(_score_single, rule, e, anchor_text) for e in essays]
        return [f.result() for f in futures]


# ══════════════════════════════════════════════════════════════════
# Phase 3 — Defect Diagnosis Agent (DDA)      [Fig. 10 in paper]
# 阶段3 — 缺陷诊断智能体（DDA）                 [论文图10]
# ══════════════════════════════════════════════════════════════════

DDA_SYSTEM_PROMPT = """You are a scoring algorithm diagnostician.

[Task Objective]:
Analyze why the score of the Original sample should be higher than that of the Augmented sample.
The model may currently be giving incorrectly high scores (Aug >= Orig).
We need to identify the regression in writing quality of the augmented sample.

[Strict Constraints]:
1. Content Attribution Prohibited: Never state "because the original sample mentioned a certain
   example" or "the augmented sample missed a sentence."
2. Sentence Structure Extraction Prohibited: Never extract specific sentence structures.
3. Mandatory Use of Official Dimension Terminology: Your analysis must be grounded in the
   degradation of the following official concepts:
   - Elaboration: Did it regress to "list-like details"? Did it become "vague details"?
   - Organization: Did the structure become "fragmented"? Were logical connections lost?
   - Fluency: Does it read "awkward"? Did transitions become abrupt?
   - Audience Awareness: Was "awareness of audience" lost?

Output JSON:
{{
  "analysis": "The original sample demonstrated full elaboration in the [Dimension], while the augmented sample degenerated into list-like details, despite having a similar word count...",
  "extracted_features": [
    {{
      "type": "defect",
      "description": "The augmented sample exhibits 'list-like' characteristics in the Elaboration dimension, lacking in-depth support, rather than a lack of specific arguments."
    }},
    {{
      "type": "advantage",
      "description": "The original sample demonstrated 'sophisticated transitional language' in the Fluency dimension, with tight logical connections between paragraphs."
    }}
  ],
  "optimization_suggestion": "It is recommended to tighten the definition of 'Adequate Support', explicitly excluding list-like discussion styles."
}}"""

DDA_USER_PROMPT = """[Current Scoring Rubric]:
{rule_content}

[Original Sample (True Score: {orig_score})]:
{orig_content}

[Augmented Sample (True Score: {aug_score})]:
{aug_content}

Please compare the Holistic Quality Gap between the two."""


def run_dda(rule: ScoringRule,
            orig: Essay, aug: Essay) -> Dict:
    """
    Defect Diagnosis Agent — identifies root-cause feature tags from a
    scoring inversion between a positive and negative sample pair.
    缺陷诊断智能体 —— 从正负样本对的评分倒置中识别根因特征标签。

    Args / 参数:
        rule: current rule that produced the inversion / 产生倒置的当前规则
        orig: the higher-quality (positive) essay / 较高质量的正样本作文
        aug:  the lower-quality (negative) essay / 较低质量的负样本作文

    Returns / 返回:
        dict with keys / 包含以下键的字典: "analysis", "extracted_features", "optimization_suggestion"
    """
    print(f"[DDA] Diagnosing inversion: orig={orig.true_score} vs aug={aug.true_score}")
    llm = _reasoner_llm()
    user = DDA_USER_PROMPT.format(
        rule_content=rule.rule_content,
        orig_score=orig.true_score,
        orig_content=orig.essay_content,
        aug_score=aug.true_score,
        aug_content=aug.essay_content,
    )
    try:
        resp = llm.invoke([SystemMessage(content=DDA_SYSTEM_PROMPT), HumanMessage(content=user)])
        content = re.sub(r"```json|```", "", resp.content).strip()
        return json.loads(content)
    except Exception as e:
        print(f"[DDA] Analysis error: {e}")
        return {"analysis": "Error", "extracted_features": [], "optimization_suggestion": ""}


# ══════════════════════════════════════════════════════════════════
# Phase 3 — Rule Optimization Agent (ROA)     [Fig. 11 in paper]
# 阶段3 — 规则优化智能体（ROA）                 [论文图11]
# ══════════════════════════════════════════════════════════════════

ROA_SYSTEM_PROMPT = """You are a Rubric design expert.
Objective: Fine-tune the Descriptors of the scoring rules to define score bands more precisely.

[Core Principles]:
1. Refine, Don't Add: Do not add rules like "deduct points if X appears."
   Your task is to modify the definitions of existing scores.
2. Holistic View: The rules must focus on the overall quality of the essay
   (e.g., logical density, degree of elaboration), rather than local features.
3. Descriptor Calibration:
   - If an augmented sample (low quality) received a high score, it indicates that the
     description for the high-score band is too broad.
   - Task: Please modify the description of the corresponding dimension to give it more
     Discriminative Power.
   - Example: Change "supported by details" to "supported by fully elaborated details, and non list-like."

[Expert Feedback]: {expert_feedback}
[Known Abstract Defects (from Feature Library)]: {feature_lib_text}

Please output the complete revised rule text."""

ROA_USER_PROMPT = """Current Rule:
{current_rule}

Problem Diagnosis:
{analysis}

Identified Dimension Degradations:
{extracted_features}

Please output the optimized rule:"""


def run_roa(current_rule: ScoringRule,
            dda_analysis: Dict,
            feature_lib_text: str,
            expert_feedback: str = "Approve and execute the optimization.") -> ScoringRule:
    """
    Rule Optimization Agent — refines the current rubric based on DDA findings
    and RAG-retrieved historical features.
    规则优化智能体 —— 基于 DDA 分析结果和 RAG 检索的历史特征精化当前评分标准。

    Args / 参数:
        current_rule:    rule to be refined / 待精化的当前规则
        dda_analysis:    output dict from run_dda() / run_dda() 的输出字典
        feature_lib_text: top-k features retrieved from the feature library / 从特征库检索的 top-k 特征
        expert_feedback:  optional human-in-the-loop instruction / 可选的人机协作指令

    Returns / 返回:
        a new ScoringRule with updated rule_content / 包含更新后规则内容的新 ScoringRule
    """
    print("[ROA] Optimizing rule based on defect diagnosis...")
    llm = _reasoner_llm()
    system = ROA_SYSTEM_PROMPT.format(
        expert_feedback=expert_feedback,
        feature_lib_text=feature_lib_text or "No features in library yet.",
    )
    user = ROA_USER_PROMPT.format(
        current_rule=current_rule.rule_content,
        analysis=dda_analysis.get("analysis", ""),
        extracted_features=dda_analysis.get("extracted_features", []),
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return ScoringRule(
        rule_id="temp",
        rule_content=resp.content.strip(),
        dimensions=current_rule.dimensions,
    )
