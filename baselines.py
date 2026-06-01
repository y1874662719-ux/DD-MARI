"""
Baseline implementations for comparison with DD-MARI on ASAP Set 1.
All baselines use DeepSeek as the backbone, evaluated on the full test set (N=1762).

Baselines:
  1. G-Eval          — direct rubric-grounded scoring (Liu et al., 2023)
  2. Prometheus-DA   — reference-anchored rubric scoring (Kim et al., 2024)
  3. Reflexion       — 3-round self-reflection scoring  (Shinn et al., 2023)
"""

import re
import json
import time
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, SCORE_MIN, SCORE_MAX
from data_models import Essay, EvaluationRecord, SampleType, ScoringRule
from evaluator import calculate_qwk_detailed


# ──────────────────────────────────────────────────────────────────
# ASAP Set 1 official rubric (from Set1_standard.docx)
# ──────────────────────────────────────────────────────────────────
ASAP_RUBRIC = """\
Prompt: Write a letter to your local newspaper stating your opinion on the effects \
computers have on people. Persuade the readers to agree with you.

Scoring Rubric — score on a continuous scale from 2 (lowest) to 12 (highest).
Any integer from 2 to 12 is valid; odd numbers (3, 5, 7, 9, 11) are equally valid as even numbers.

Score 2–3 (Very Poor): Undeveloped. Takes a position but offers only very minimal support.
  - Few or vague details; awkward and fragmented; difficult to understand; no audience awareness.

Score 4–5 (Poor): Under-developed. May or may not take a clear position.
  - Only general reasons with unelaborated/list-like details; little or no organization;
    awkward/confused or simplistic; little audience awareness.

Score 6–7 (Developing): Minimally-developed. Takes a position but with inadequate support.
  - Reasons with minimal elaboration and mostly general details; some organization;
    awkward in parts with few transitions; some awareness of audience.

Score 7–8 (Adequate): Somewhat-developed. Takes a position with adequate support.
  - Adequately elaborated reasons with a mix of general and specific details;
    satisfactory organization; somewhat fluent with some transitional language;
    adequate awareness of audience.

Score 8–10 (Good): Developed. Takes a clear position with reasonably persuasive support.
  - Moderately well elaborated reasons with mostly specific details; generally strong
    organization; moderately fluent with transitional language throughout;
    consistent awareness of audience.

Score 10–12 (Excellent): Well-developed. Takes a clear and thoughtful position with \
persuasive support.
  - Fully elaborated reasons with specific details; strong organization; fluent with
    sophisticated transitional language; heightened awareness of audience.
"""


# ──────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────

def _make_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=temperature,
    )


# ── Gemini LLM factory (reads GEMINI_* env vars) ──────────────────
import os as _os
_GEMINI_API_KEY   = _os.environ.get("GEMINI_API_KEY")
_GEMINI_BASE_URL  = _os.environ.get("GEMINI_BASE_URL",  "https://api.v3.cm/v1")
_GEMINI_MODEL     = _os.environ.get("GEMINI_MODEL_NAME", "gemini-3.1-pro-preview")


def _make_gemini_llm(temperature: float = 0.0) -> ChatOpenAI:
    if not _GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to .env:\n"
            "  GEMINI_API_KEY=your_key\n"
            "  GEMINI_BASE_URL=https://api.v3.cm/v1\n"
            "  GEMINI_MODEL_NAME=gemini-3.1-pro-preview"
        )
    return ChatOpenAI(
        model=_GEMINI_MODEL,
        openai_api_base=_GEMINI_BASE_URL,
        openai_api_key=_GEMINI_API_KEY,
        temperature=temperature,
    )


def _extract_score(text: str) -> int:
    """Parse integer score from LLM response; try JSON first, then regex."""
    try:
        clean = re.sub(r"```(?:json)?", "", text).strip()
        s, e = clean.find("{"), clean.rfind("}") + 1
        if s >= 0 and e > s:
            obj = json.loads(clean[s:e])
            for key in ("score", "final_score", "total_score", "combined_score"):
                if key in obj:
                    return max(SCORE_MIN, min(SCORE_MAX, int(float(obj[key]))))
    except Exception:
        pass
    for pat in (r"final[_\s]score[:\s]+(\d+)", r"score[:\s]+(\d+)", r"\b(1[0-2]|[2-9])\b"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if SCORE_MIN <= v <= SCORE_MAX:
                return v
    return (SCORE_MIN + SCORE_MAX) // 2


def _compute_qwk(essays: List[Essay], score_map: Dict[str, int]):
    records = [
        EvaluationRecord(
            essay_id=e.essay_id,
            predicted_score=score_map[e.essay_id],
            dimension_scores={},
            true_score=e.true_score,
            rationale="",
        )
        for e in essays if e.essay_id in score_map
    ]
    filtered = [e for e in essays if e.essay_id in score_map]
    return calculate_qwk_detailed(filtered, records)


# ──────────────────────────────────────────────────────────────────
# Baseline 1: G-Eval
# Liu et al. (2023) "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
# ──────────────────────────────────────────────────────────────────

_PRIVACY_NOTE = (
    "IMPORTANT: The essays may contain privacy desensitization markers such as "
    "@CAPS1, @CAPS2, @NUM1, @ORGANIZATION1, @PERSON1, etc. "
    "These are dataset anonymization tags, NOT spelling or grammar errors. "
    "You must NOT deduct any points because an essay contains these markers. "
    "When evaluating language conventions or fluency, ignore these tags entirely."
)

_GEVAL_SYS = (
    "You are an expert essay rater for the ASAP educational benchmark. "
    "Score the essay strictly following the rubric. "
    + _PRIVACY_NOTE + " "
    "Respond with JSON only: {\"score\": <integer 2-12>}"
)

_GEVAL_USER = """\
{rubric}

[Calibration Anchors — essays with known scores to help you calibrate the 2-12 scale]:
{anchor_text}

Evaluate the student essay below and assign a score from 2 to 12.
Any integer from 2 to 12 is valid — do NOT restrict yourself to even numbers only.

Essay:
\"\"\"
{essay}
\"\"\"

Respond with JSON only: {{"score": <integer 2-12>}}"""


def _geval_one(llm: ChatOpenAI, essay: Essay, anchor_text: str) -> Tuple[str, int]:
    resp = llm.invoke([
        SystemMessage(_GEVAL_SYS),
        HumanMessage(_GEVAL_USER.format(
            rubric=ASAP_RUBRIC,
            anchor_text=anchor_text,
            essay=essay.essay_content[:3000],
        )),
    ])
    return essay.essay_id, _extract_score(resp.content)


def run_geval(essays: List[Essay], anchor_text: str, max_workers: int = 8) -> Dict[str, int]:
    """G-Eval: direct rubric-grounded scoring with DeepSeek."""
    llm = _make_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_geval_one, llm, e, anchor_text): e for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 200 == 0:
                print(f"    G-Eval progress: {i+1}/{len(essays)}")
    return score_map


def run_geval_gemini(essays: List[Essay], anchor_text: str, max_workers: int = 8) -> Dict[str, int]:
    """G-Eval with Gemini backbone (same prompt, different model)."""
    llm = _make_gemini_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_geval_one, llm, e, anchor_text): e for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 200 == 0:
                print(f"    G-Eval (Gemini) progress: {i+1}/{len(essays)}")
    return score_map


# ──────────────────────────────────────────────────────────────────
# Baseline 2: Prometheus-style Direct Assessment
# Kim et al. (2024) "Prometheus: Inducing Fine-grained Evaluation Capability"
# ──────────────────────────────────────────────────────────────────

_PROM_SYS = (
    "You are a rigorous essay evaluator. Score essays by comparing them against "
    "provided reference essays and applying the rubric. "
    + _PRIVACY_NOTE + " "
    "Respond with JSON only: {\"score\": <integer 2-12>}"
)

_PROM_USER = """\
{rubric}

[Calibration Anchors — essays with known scores to help you calibrate the 2-12 scale]:
{anchor_text}

Below are two additional reference essays at the extremes:

=== REFERENCE A (Score 12/12 — Excellent) ===
{ref_high}

=== REFERENCE B (Score 4/12 — Below Average) ===
{ref_low}

=== Essay to Score ===
{essay}

Instructions:
1. Use the anchor essays above to calibrate your sense of the 2-12 scale.
2. Assign a score from 2 to 12. Any integer is valid — odd numbers (7, 9, etc.) are equally valid.

Respond with JSON only: {{"score": <integer 2-12>}}"""


def _prometheus_one(
    llm: ChatOpenAI, essay: Essay, anchor_text: str, ref_high: str, ref_low: str
) -> Tuple[str, int]:
    resp = llm.invoke([
        SystemMessage(_PROM_SYS),
        HumanMessage(_PROM_USER.format(
            rubric=ASAP_RUBRIC,
            anchor_text=anchor_text,
            ref_high=ref_high[:1000],
            ref_low=ref_low[:800],
            essay=essay.essay_content[:2500],
        )),
    ])
    return essay.essay_id, _extract_score(resp.content)


def run_prometheus(
    essays: List[Essay], anchor_text: str, ref_high: str, ref_low: str,
    max_workers: int = 8
) -> Dict[str, int]:
    """Prometheus-style: reference-anchored rubric scoring with DeepSeek."""
    llm = _make_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_prometheus_one, llm, e, anchor_text, ref_high, ref_low): e
                   for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 200 == 0:
                print(f"    Prometheus progress: {i+1}/{len(essays)}")
    return score_map


# ──────────────────────────────────────────────────────────────────
# Baseline 3: Reflexion
# Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"
# ──────────────────────────────────────────────────────────────────

_RFX_R1_SYS = (
    "You are an essay scorer. Score the essay using the rubric. "
    + _PRIVACY_NOTE + " "
    "Respond with JSON: {\"score\": <integer 2-12>, \"rationale\": \"<one sentence>\"}"
)
_RFX_R1_USER = """\
{rubric}

[Calibration Anchors — essays with known scores to help you calibrate the 2-12 scale]:
{anchor_text}

Score this student essay on a scale from 2 to 12.
Any integer from 2 to 12 is valid — odd numbers are equally valid as even numbers.

Essay:
\"\"\"
{essay}
\"\"\"
Respond: {{"score": <int 2-12>, "rationale": "<one sentence>"}}"""

_RFX_R2_SYS = (
    "You are reflecting on a previous essay score to check for scoring biases. "
    "Respond with JSON: {\"reflection\": \"<your reflection>\"}"
)
_RFX_R2_USER = """\
You scored an essay {score}/12 with this rationale: "{rationale}"

Reflect on potential scoring errors:
1. Did essay LENGTH influence your score more than content quality?
2. Is your score consistent with the rubric level descriptions?
3. Did you overlook any quality indicators (e.g., rhetorical devices, paragraph structure)?

Respond: {{"reflection": "<your reflection>"}}"""

_RFX_R3_SYS = (
    "You are finalizing an essay score after self-reflection. "
    "Respond with JSON: {\"final_score\": <integer 2-12>}"
)
_RFX_R3_USER = """\
{rubric}

Essay:
\"\"\"
{essay}
\"\"\"

Your initial score: {score}/12
After reflection: {reflection}

Provide your final score based on the rubric and your reflection.
Respond: {{"final_score": <int>}}"""


def _reflexion_one(llm: ChatOpenAI, essay: Essay, anchor_text: str) -> Tuple[str, int]:
    text = essay.essay_content[:3000]

    # Round 1: initial score + rationale
    r1 = llm.invoke([
        SystemMessage(_RFX_R1_SYS),
        HumanMessage(_RFX_R1_USER.format(
            rubric=ASAP_RUBRIC, anchor_text=anchor_text, essay=text,
        )),
    ])
    try:
        clean = re.sub(r"```(?:json)?", "", r1.content).strip()
        obj1 = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        score1 = max(SCORE_MIN, min(SCORE_MAX, int(float(obj1.get("score", 7)))))
        rationale = str(obj1.get("rationale", ""))[:300]
    except Exception:
        score1 = _extract_score(r1.content)
        rationale = ""

    # Round 2: reflection
    r2 = llm.invoke([
        SystemMessage(_RFX_R2_SYS),
        HumanMessage(_RFX_R2_USER.format(score=score1, rationale=rationale)),
    ])
    try:
        clean = re.sub(r"```(?:json)?", "", r2.content).strip()
        obj2 = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        reflection = str(obj2.get("reflection", ""))[:500]
    except Exception:
        reflection = r2.content[:500]

    # Round 3: final score
    r3 = llm.invoke([
        SystemMessage(_RFX_R3_SYS),
        HumanMessage(_RFX_R3_USER.format(
            rubric=ASAP_RUBRIC, essay=text, score=score1, reflection=reflection,
        )),
    ])
    return essay.essay_id, _extract_score(r3.content)


def run_reflexion(essays: List[Essay], anchor_text: str, max_workers: int = 4) -> Dict[str, int]:
    """Reflexion: 3-round self-reflection scoring with DeepSeek."""
    llm = _make_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_reflexion_one, llm, e, anchor_text): e for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 100 == 0:
                print(f"    Reflexion progress: {i+1}/{len(essays)}")
    return score_map


# ──────────────────────────────────────────────────────────────────
# Baseline 4: Zero-Shot
# Direct scoring with rubric and anchors, no chain-of-thought
# ──────────────────────────────────────────────────────────────────

_ZS_SYS = (
    "You are an essay scorer for the ASAP benchmark. "
    "Use the rubric and calibration anchors to assign a score. "
    + _PRIVACY_NOTE + " "
    "Respond with JSON only: {\"score\": <integer 2-12>}"
)

_ZS_USER = """\
{rubric}

[Calibration Anchors]:
{anchor_text}

Essay:
\"\"\"
{essay}
\"\"\"

Assign a score from 2 to 12. Any integer is valid.
Respond with JSON only: {{"score": <integer 2-12>}}"""


def _zeroshot_one(llm: ChatOpenAI, essay: Essay, anchor_text: str) -> Tuple[str, int]:
    resp = llm.invoke([
        SystemMessage(_ZS_SYS),
        HumanMessage(_ZS_USER.format(
            rubric=ASAP_RUBRIC,
            anchor_text=anchor_text,
            essay=essay.essay_content[:3000],
        )),
    ])
    return essay.essay_id, _extract_score(resp.content)


def run_zeroshot(essays: List[Essay], anchor_text: str, max_workers: int = 8) -> Dict[str, int]:
    """Zero-Shot: rubric + anchors, direct scoring, no CoT."""
    llm = _make_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_zeroshot_one, llm, e, anchor_text): e for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 200 == 0:
                print(f"    Zero-Shot progress: {i+1}/{len(essays)}")
    return score_map


# ──────────────────────────────────────────────────────────────────
# Baseline 5: Few-Shot CoT
# Rubric + anchors + 3 labeled examples + chain-of-thought reasoning
# ──────────────────────────────────────────────────────────────────

_FSC_SYS = (
    "You are an expert essay rater for the ASAP educational benchmark. "
    "Score essays step-by-step using the rubric and examples provided. "
    + _PRIVACY_NOTE + " "
    "Respond with JSON only at the end: {\"score\": <integer 2-12>}"
)

_FSC_USER = """\
{rubric}

[Calibration Anchors]:
{anchor_text}

[Scored Examples — showing how to apply the rubric]:
{examples}

Now evaluate the following essay. Reason step-by-step across Ideas & Elaboration,
Organization, Fluency & Language, and Audience Awareness. Then assign a final score.

Essay:
\"\"\"
{essay}
\"\"\"

Think step by step, then respond with JSON only: {{"score": <integer 2-12>}}"""


def _build_fewshot_examples_text(gradient_essays: List[Essay]) -> str:
    """Pick one essay each at scores ~4, ~8, ~12 from the gradient set."""
    lines = []
    for target in [4, 8, 12]:
        matches = sorted(gradient_essays, key=lambda e: abs(e.true_score - target))
        e = matches[0]
        snippet = e.essay_content[:500]
        if len(e.essay_content) > 500:
            snippet += "\n...[truncated]..."
        lines.append(
            f"--- Example (Score {e.true_score}/12) ---\n{snippet}\n"
            f"Score: {e.true_score}/12\n"
        )
    return "\n".join(lines)


def _fewshot_one(
    llm: ChatOpenAI, essay: Essay, anchor_text: str, examples_text: str
) -> Tuple[str, int]:
    resp = llm.invoke([
        SystemMessage(_FSC_SYS),
        HumanMessage(_FSC_USER.format(
            rubric=ASAP_RUBRIC,
            anchor_text=anchor_text,
            examples=examples_text,
            essay=essay.essay_content[:2500],
        )),
    ])
    return essay.essay_id, _extract_score(resp.content)


def run_fewshot(
    essays: List[Essay], anchor_text: str, gradient_essays: List[Essay],
    max_workers: int = 8
) -> Dict[str, int]:
    """Few-Shot CoT: rubric + anchors + 3 labeled examples + step-by-step reasoning."""
    examples_text = _build_fewshot_examples_text(gradient_essays)
    llm = _make_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fewshot_one, llm, e, anchor_text, examples_text): e
                   for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 200 == 0:
                print(f"    Few-Shot CoT progress: {i+1}/{len(essays)}")
    return score_map


# ──────────────────────────────────────────────────────────────────
# Baseline 6: Self-Refine
# Madaan et al. (2023) "Self-Refine: Iterative Refinement with Self-Feedback"
# score → rubric-consistency feedback → refined score
# (distinct from Reflexion which focuses on cognitive/length bias)
# ──────────────────────────────────────────────────────────────────

_SR_R1_SYS = (
    "You are an essay scorer. Score the essay using the rubric. "
    + _PRIVACY_NOTE + " "
    "Respond with JSON: {\"score\": <integer 2-12>, \"rationale\": \"<one sentence>\"}"
)
_SR_R1_USER = """\
{rubric}

[Calibration Anchors]:
{anchor_text}

Score this essay from 2 to 12. Any integer is valid — odd numbers are equally valid.

Essay:
\"\"\"
{essay}
\"\"\"

Respond: {{"score": <int 2-12>, "rationale": "<one sentence>"}}"""

_SR_R2_SYS = (
    "You are a scoring quality reviewer. Check whether the assigned score is "
    "consistent with the rubric criteria and calibration anchors. "
    "Respond with JSON: {\"feedback\": \"<your assessment>\"}"
)
_SR_R2_USER = """\
{rubric}

[Calibration Anchors]:
{anchor_text}

The essay below was scored {score}/12 with rationale: "{rationale}"

Essay:
\"\"\"
{essay}
\"\"\"

Review questions:
1. Does the score match the rubric band description for {score}/12?
2. Compared to the calibration anchors, is {score}/12 appropriate for this essay?
3. Is there a more fitting score band given the essay's actual quality?

Respond: {{"feedback": "<assessment and suggested correction if any>"}}"""

_SR_R3_SYS = (
    "You are finalizing an essay score after a quality review. "
    "Respond with JSON: {\"final_score\": <integer 2-12>}"
)
_SR_R3_USER = """\
{rubric}

Essay:
\"\"\"
{essay}
\"\"\"

Initial score: {score}/12
Reviewer feedback: {feedback}

Provide your final score. Adjust if the reviewer identified an error.
Respond: {{"final_score": <int>}}"""


def _selfrefine_one(llm: ChatOpenAI, essay: Essay, anchor_text: str) -> Tuple[str, int]:
    text = essay.essay_content[:3000]

    # Round 1: initial score + rationale
    r1 = llm.invoke([
        SystemMessage(_SR_R1_SYS),
        HumanMessage(_SR_R1_USER.format(
            rubric=ASAP_RUBRIC, anchor_text=anchor_text, essay=text,
        )),
    ])
    try:
        clean = re.sub(r"```(?:json)?", "", r1.content).strip()
        obj1 = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        score1 = max(SCORE_MIN, min(SCORE_MAX, int(float(obj1.get("score", 7)))))
        rationale = str(obj1.get("rationale", ""))[:300]
    except Exception:
        score1 = _extract_score(r1.content)
        rationale = ""

    # Round 2: rubric-consistency feedback
    r2 = llm.invoke([
        SystemMessage(_SR_R2_SYS),
        HumanMessage(_SR_R2_USER.format(
            rubric=ASAP_RUBRIC, anchor_text=anchor_text,
            score=score1, rationale=rationale, essay=text,
        )),
    ])
    try:
        clean = re.sub(r"```(?:json)?", "", r2.content).strip()
        obj2 = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        feedback = str(obj2.get("feedback", ""))[:500]
    except Exception:
        feedback = r2.content[:500]

    # Round 3: refined final score
    r3 = llm.invoke([
        SystemMessage(_SR_R3_SYS),
        HumanMessage(_SR_R3_USER.format(
            rubric=ASAP_RUBRIC, essay=text, score=score1, feedback=feedback,
        )),
    ])
    return essay.essay_id, _extract_score(r3.content)


def run_selfrefine(essays: List[Essay], anchor_text: str, max_workers: int = 4) -> Dict[str, int]:
    """Self-Refine: initial score → rubric-consistency feedback → refined score."""
    llm = _make_llm(temperature=0.0)
    score_map: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_selfrefine_one, llm, e, anchor_text): e for e in essays}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            eid, score = fut.result()
            score_map[eid] = score
            if (i + 1) % 100 == 0:
                print(f"    Self-Refine progress: {i+1}/{len(essays)}")
    return score_map


# ──────────────────────────────────────────────────────────────────
# Baseline 8: DD-MARI v2 — custom length-aware system prompt
# Same rule + calibration pipeline, but scoring agent uses a length-
# heuristic prompt supplied by the user (Chinese, explicit char thresholds)
# ──────────────────────────────────────────────────────────────────

_V2_SA_SYSTEM = """\
你是一个专业的作文评分员。
评分逻辑：**Analytic Scoring (维度分析法)**。评价应基于**整体观感 (Holistic Impression)**，而非寻找具体的关键词。

【评分标准 Rubric】：
{rule_content}

【校准锚点 Anchors】：
{anchor_text}

【注意事项】：
1. 低分文章（2-5分）有一个特点，就是文章偏短，一般在800个字符以下；并且超过1000字符的文章总分最少要评6分或7分！对于篇幅较长(超过1200字符)的文章，可以稍微忽略一些字符拼写错误
2. 高分文章（11-12分）有一个特点。就是文章篇幅较长，一般都在1800字符以上，如果字符数没有达到1800以上，绝对不能评到11-12分！
3. 中等文章（8-10分）的作文一般字符数一般在1200-1800字符之间

【打分步骤】：
1. **Holistic Comparison (整体对比)**：
   - 将待测文章与 2分 4分 6分(及格线) 8分和 10分(优秀线) 锚点进行**整体质量**对比。
   - 它的**Elaboration (展开度)** 是否比 6分锚点更像 "list-like" ？
   - 它的**Fluency (流畅度)** 是否达到了 10分锚点的 "sophisticated" 水平？
2. **Dimension Scoring (维度打分)**：根据规则对每个维度分别打分，总分 = 维度一得分+维度二得分+维度三得分+维度四得分
3. 你会在文章中看到大量以 @ 开头的标签，例如 **@CAPS1, @ORGANIZATION2, @NUM1** 等。这是数据集的隐私脱敏标记，**严禁**因为文章包含这些标签而扣分。在评估"语言规范"时，必须完全忽略这些标签。"""

_V2_SA_USER = """\
待评估作文：
{essay_content}

请先输出你的分析思路（Rationale），最后仅输出如下 JSON：
{{
  "rationale": "基于整体观感的评分理由...",
  "dimension_scores": {{"维度名称": 分数, ...}},
  "total_score": 总分（整数）
}}"""


def _score_single_v2(rule: ScoringRule, essay: Essay, anchor_text: str) -> EvaluationRecord:
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_base=LLM_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,
    )
    system = _V2_SA_SYSTEM.format(
        rule_content=rule.rule_content,
        anchor_text=anchor_text,
    )
    user = _V2_SA_USER.format(essay_content=essay.essay_content)
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        import re as _re, json as _json
        text = _re.sub(r"```(?:json)?", "", resp.content).strip()
        s, e = text.find("{"), text.rfind("}") + 1
        data = _json.loads(text[s:e])
        return EvaluationRecord(
            essay_id=essay.essay_id,
            predicted_score=max(SCORE_MIN, min(SCORE_MAX, int(data["total_score"]))),
            dimension_scores=data.get("dimension_scores", {}),
            true_score=essay.true_score,
            rationale=data.get("rationale", ""),
        )
    except Exception as ex:
        print(f"[V2-SA] Error for {essay.essay_id}: {ex}")
        return EvaluationRecord(essay.essay_id, SCORE_MIN, {}, essay.true_score, "Error")


def run_dd_mari_v2(essays: List[Essay], anchor_text: str) -> Dict[str, int]:
    """DD-MARI v2: same final rule + calibration, but length-aware Chinese prompt."""
    from cognitive_alignment import apply_calibration_to_records

    for candidate in [
        Path("output/final_rule.json"),
        Path("score_demo/data/rules/final_rule.json"),
    ]:
        if candidate.exists():
            rule_path = candidate
            break
    else:
        raise FileNotFoundError("final_rule.json not found.")
    with open(rule_path, "r", encoding="utf-8") as f:
        rule = ScoringRule.from_dict(json.load(f))

    alphas_path = Path("output/final_alphas.json")
    if alphas_path.exists():
        with open(alphas_path, "r", encoding="utf-8") as f:
            alphas = json.load(f)
    else:
        from config import CALIB_ALPHAS
        alphas = CALIB_ALPHAS

    print(f"  [DD-MARI v2] Scoring {len(essays)} essays with length-aware prompt...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_score_single_v2, rule, e, anchor_text): e for e in essays}
        records = []
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            records.append(fut.result())
            if (i + 1) % 200 == 0:
                print(f"    DD-MARI v2 progress: {i+1}/{len(essays)}")

    calibrated = apply_calibration_to_records(records, alphas)
    return {r.essay_id: r.predicted_score for r in calibrated}


# ──────────────────────────────────────────────────────────────────
# Baseline 7: DD-MARI (Full Test Set, no CDAP filter)
# Final optimized rule + cognitive calibration, evaluated on all 1762 essays
# ──────────────────────────────────────────────────────────────────

def run_dd_mari_full(essays: List[Essay], anchor_text: str) -> Dict[str, int]:
    """DD-MARI final rule + calibration on the full test set (no CDAP filter)."""
    from agents import run_sa
    from cognitive_alignment import apply_calibration_to_records

    for candidate in [
        Path("output/final_rule.json"),
        Path("score_demo/data/rules/final_rule.json"),
    ]:
        if candidate.exists():
            rule_path = candidate
            break
    else:
        raise FileNotFoundError(
            "final_rule.json not found in output/ or score_demo/data/rules/. "
            "Run main.py to train DD-MARI first."
        )
    print(f"  [DD-MARI] Loading rule from {rule_path}")
    with open(rule_path, "r", encoding="utf-8") as f:
        rule = ScoringRule.from_dict(json.load(f))

    alphas_path = Path("output/final_alphas.json")
    if alphas_path.exists():
        with open(alphas_path, "r", encoding="utf-8") as f:
            alphas = json.load(f)
        print(f"  [DD-MARI] Loaded alphas: {alphas}")
    else:
        from config import CALIB_ALPHAS
        alphas = CALIB_ALPHAS
        print(f"  [DD-MARI] final_alphas.json not found — using default {alphas}")

    print(f"  [DD-MARI] Scoring {len(essays)} essays with final rule + calibration (no CDAP)...")
    records = run_sa(rule, essays, anchor_text)
    calibrated = apply_calibration_to_records(records, alphas)
    return {r.essay_id: r.predicted_score for r in calibrated}


# ──────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────

def _load_essays() -> List[Essay]:
    df = pd.read_excel("data/asap_set1_test.xlsx")
    col = {c.lower(): c for c in df.columns}
    id_col    = col.get("essay_id")    or df.columns[0]
    text_col  = col.get("essay")       or df.columns[1]
    score_col = col.get("domain1_score") or df.columns[2]
    essays = []
    for _, row in df.iterrows():
        if pd.isna(row[text_col]):
            continue
        raw = int(row[score_col]) if pd.notna(row[score_col]) else 0
        score = max(SCORE_MIN, min(SCORE_MAX, raw))
        essays.append(Essay(str(row[id_col]), SampleType.TEST, str(row[text_col]), score))
    return essays


def _load_references() -> Tuple[str, str]:
    df = pd.read_excel("data/asap_set1_gradient.xlsx")
    col = {c.lower(): c for c in df.columns}
    text_col  = col.get("essay")         or df.columns[1]
    score_col = col.get("domain1_score") or df.columns[2]
    df_s = df.sort_values(score_col, ascending=False)
    return str(df_s.iloc[0][text_col]), str(df_s.iloc[-1][text_col])


def _build_anchor_text() -> str:
    """Build the same anchor text used by DD-MARI's Scoring Agent."""
    from data_manager import load_gradient_set, build_anchor_text
    return build_anchor_text(load_gradient_set())


def _save_checkpoint(name: str, score_map: Dict[str, int]) -> None:
    Path("output").mkdir(exist_ok=True)
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    path = Path(f"output/checkpoint_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(score_map, f)
    print(f"  Checkpoint saved → {path.name}")


def _load_checkpoint(name: str) -> Dict[str, int]:
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    path = Path(f"output/checkpoint_{safe}.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    print("=" * 60)
    print("  Baseline Evaluation — ASAP Set 1 Full Test Set")
    print("=" * 60)

    essays = _load_essays()
    print(f"  Test essays: {len(essays)}")

    ref_high, ref_low = _load_references()
    anchor_text = _build_anchor_text()
    print(f"  Anchor text: {len(anchor_text)} chars")

    from data_manager import load_gradient_set
    gradient_essays = load_gradient_set()
    print(f"  Gradient set: {len(gradient_essays)} essays")

    all_results: Dict[str, dict] = {}
    total_baselines = 9

    # ── 1. G-Eval ────────────────────────────────────────────────
    name = "G-Eval (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[1/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        geval_map = cached
    else:
        print(f"\n[1/{total_baselines}] {name}...")
        t0 = time.time()
        geval_map = run_geval(essays, anchor_text)
        _save_checkpoint(name, geval_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    qwk, det = _compute_qwk(essays, geval_map)
    print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
          f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
    all_results[name] = {"qwk": qwk, "map": geval_map, **det}

    # ── 2. Prometheus-style ───────────────────────────────────────
    name = "Prometheus-DA (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[2/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        prom_map = cached
    else:
        print(f"\n[2/{total_baselines}] {name}...")
        t0 = time.time()
        prom_map = run_prometheus(essays, anchor_text, ref_high, ref_low)
        _save_checkpoint(name, prom_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    qwk, det = _compute_qwk(essays, prom_map)
    print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
          f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
    all_results[name] = {"qwk": qwk, "map": prom_map, **det}

    # ── 3. Reflexion ──────────────────────────────────────────────
    name = "Reflexion (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[3/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        reflex_map = cached
    else:
        print(f"\n[3/{total_baselines}] {name}...")
        t0 = time.time()
        reflex_map = run_reflexion(essays, anchor_text)
        _save_checkpoint(name, reflex_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    qwk, det = _compute_qwk(essays, reflex_map)
    print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
          f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
    all_results[name] = {"qwk": qwk, "map": reflex_map, **det}

    # ── 4. Zero-Shot ──────────────────────────────────────────────
    name = "Zero-Shot (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[4/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        zs_map = cached
    else:
        print(f"\n[4/{total_baselines}] {name}...")
        t0 = time.time()
        zs_map = run_zeroshot(essays, anchor_text)
        _save_checkpoint(name, zs_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    qwk, det = _compute_qwk(essays, zs_map)
    print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
          f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
    all_results[name] = {"qwk": qwk, "map": zs_map, **det}

    # ── 5. Few-Shot CoT ───────────────────────────────────────────
    name = "Few-Shot CoT (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[5/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        fsc_map = cached
    else:
        print(f"\n[5/{total_baselines}] {name}...")
        t0 = time.time()
        fsc_map = run_fewshot(essays, anchor_text, gradient_essays)
        _save_checkpoint(name, fsc_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    qwk, det = _compute_qwk(essays, fsc_map)
    print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
          f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
    all_results[name] = {"qwk": qwk, "map": fsc_map, **det}

    # ── 6. Self-Refine ────────────────────────────────────────────
    name = "Self-Refine (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[6/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        sr_map = cached
    else:
        print(f"\n[6/{total_baselines}] {name}...")
        t0 = time.time()
        sr_map = run_selfrefine(essays, anchor_text)
        _save_checkpoint(name, sr_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    qwk, det = _compute_qwk(essays, sr_map)
    print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
          f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
    all_results[name] = {"qwk": qwk, "map": sr_map, **det}

    # ── 7. DD-MARI Full (no CDAP filter) ─────────────────────────
    name = "DD-MARI Full (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[7/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        ddmari_map = cached
    else:
        rule_path = (
            Path("output/final_rule.json")
            if Path("output/final_rule.json").exists()
            else Path("score_demo/data/rules/final_rule.json")
        )
        if not rule_path.exists():
            print(f"\n[7/{total_baselines}] {name} — SKIPPED (output/final_rule.json not found)")
            ddmari_map = None
        else:
            print(f"\n[7/{total_baselines}] {name}...")
            t0 = time.time()
            ddmari_map = run_dd_mari_full(essays, anchor_text)
            _save_checkpoint(name, ddmari_map)
            print(f"  Done ({time.time()-t0:.0f}s)")
    if ddmari_map is not None:
        qwk, det = _compute_qwk(essays, ddmari_map)
        print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
              f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
        all_results[name] = {"qwk": qwk, "map": ddmari_map, **det}

    # ── 8. DD-MARI v2 (length-aware Chinese prompt) ───────────────
    name = "DD-MARI v2 Prompt (DeepSeek)"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[8/8] {name} — loaded from checkpoint ({len(cached)} essays)")
        v2_map = cached
    else:
        rule_path = (
            Path("output/final_rule.json")
            if Path("output/final_rule.json").exists()
            else Path("score_demo/data/rules/final_rule.json")
        )
        if not rule_path.exists():
            print(f"\n[8/8] {name} — SKIPPED (final_rule.json not found)")
            v2_map = None
        else:
            print(f"\n[8/8] {name}...")
            t0 = time.time()
            v2_map = run_dd_mari_v2(essays, anchor_text)
            _save_checkpoint(name, v2_map)
            print(f"  Done ({time.time()-t0:.0f}s)")
    if v2_map is not None:
        qwk, det = _compute_qwk(essays, v2_map)
        print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
              f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
        all_results[name] = {"qwk": qwk, "map": v2_map, **det}

    # ── 9. G-Eval Gemini ─────────────────────────────────────────
    name = f"G-Eval ({_GEMINI_MODEL})"
    cached = _load_checkpoint(name)
    if len(cached) >= len(essays) * 0.99:
        print(f"\n[9/{total_baselines}] {name} — loaded from checkpoint ({len(cached)} essays)")
        geval_gemini_map = cached
    elif not _GEMINI_API_KEY:
        print(f"\n[9/{total_baselines}] {name} — SKIPPED (GEMINI_API_KEY not set in .env)")
        geval_gemini_map = None
    else:
        print(f"\n[9/{total_baselines}] {name}...")
        t0 = time.time()
        geval_gemini_map = run_geval_gemini(essays, anchor_text)
        _save_checkpoint(name, geval_gemini_map)
        print(f"  Done ({time.time()-t0:.0f}s)")
    if geval_gemini_map is not None:
        qwk, det = _compute_qwk(essays, geval_gemini_map)
        print(f"  QWK={qwk:.4f}  AA={det.get('adjacent_accuracy',0):.4f}  "
              f"MAE={det.get('mae',0):.4f}  N={det.get('sample_count',0)}")
        all_results[name] = {"qwk": qwk, "map": geval_gemini_map, **det}

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results (Full Test Set)")
    print("=" * 60)
    print(f"  {'Method':<32} {'QWK':>6}  {'AA':>6}  {'MAE':>6}")
    print("  " + "-" * 56)
    for n, r in all_results.items():
        print(f"  {n:<32} {r['qwk']:>6.4f}  "
              f"{r.get('adjacent_accuracy',0):>6.4f}  {r.get('mae',0):>6.4f}")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────
    Path("output").mkdir(exist_ok=True)

    rows = []
    for method, r in all_results.items():
        for essay in essays:
            rows.append({
                "method": method,
                "essay_id": essay.essay_id,
                "true_score": essay.true_score,
                "predicted_score": r["map"].get(essay.essay_id),
                "error": (r["map"].get(essay.essay_id, 0) - essay.true_score)
                         if essay.essay_id in r["map"] else None,
            })
    pd.DataFrame(rows).to_excel("output/baseline_detail.xlsx", index=False)

    summary = [
        {"Method": k, "Paradigm": "Prompting", "Train Size": 0,
         "Interpretability": "High",
         "QWK": round(v["qwk"], 4),
         "Adjacent Accuracy": round(v.get("adjacent_accuracy", 0), 4),
         "MAE": round(v.get("mae", 0), 4),
         "Sample Count": v.get("sample_count", 0)}
        for k, v in all_results.items()
    ]
    pd.DataFrame(summary).to_excel("output/baseline_summary.xlsx", index=False)
    print(f"\nDetailed  → output/baseline_detail.xlsx")
    print(f"Summary   → output/baseline_summary.xlsx")


if __name__ == "__main__":
    main()
