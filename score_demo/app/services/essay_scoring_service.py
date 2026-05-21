from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.essay_scoring import DimensionReport, EssayScoreResponse
from app.services.essay_rule_loader import EssayRule, EssayRuleLoader
from app.services.providers.llm_provider import OpenAICompatibleLLMProvider


class EssayScoringService:
    def __init__(
        self,
        rule_path: str | Path | None = None,
        provider: OpenAICompatibleLLMProvider | None = None,
    ) -> None:
        self.rule_path = Path(rule_path) if rule_path is not None else settings.resolved_rule_path
        self.provider = provider or OpenAICompatibleLLMProvider()

    def score_essay(self, essay: str) -> EssayScoreResponse:
        essay_text = essay.strip()
        if not essay_text:
            raise ValueError("Essay text is required.")

        rule = EssayRuleLoader(self.rule_path).load()
        if not self.provider.enabled:
            raise RuntimeError("The scoring model is not configured.")

        data = self.provider.complete_json(
            self._build_system_prompt(rule),
            self._build_user_prompt(rule, essay_text),
        )
        if not data:
            last_error = getattr(self.provider, "last_error", "") or "empty_response"
            raise RuntimeError(f"The scoring model did not return valid JSON: {last_error}")

        return self._normalize_response(rule, data)

    def _build_system_prompt(self, rule: EssayRule) -> str:
        dimensions = ", ".join(rule.dimensions)
        return (
            "You are an expert English essay rater. Score only according to the supplied rubric. "
            "Return valid JSON only. Do not include markdown or extra commentary. "
            f"The required dimensions are: {dimensions}."
        )

    def _build_user_prompt(self, rule: EssayRule, essay: str) -> str:
        return (
            f"Rubric:\n{rule.rule_content}\n\n"
            "Score the essay below. Each dimension must be an integer from 0 to 3. "
            "The total score must be the sum of the four dimension scores, from 0 to 12.\n\n"
            "Return this exact JSON shape:\n"
            "{\n"
            '  "dimension_scores": {"Ideas & Elaboration": 0, "Organization": 0, '
            '"Fluency & Language": 0, "Audience Awareness": 0},\n'
            '  "dimension_reports": [\n'
            '    {"dimension": "Ideas & Elaboration", "score": 0, "analysis": "...", '
            '"evidence": ["..."], "suggestions": ["..."]}\n'
            "  ]\n"
            "}\n\n"
            f"Essay:\n{essay}"
        )

    def _normalize_response(self, rule: EssayRule, data: dict[str, Any]) -> EssayScoreResponse:
        raw_scores = data.get("dimension_scores") or {}
        raw_reports = data.get("dimension_reports") or []

        reports_by_dimension = {
            str(item.get("dimension", "")).strip(): item
            for item in raw_reports
            if isinstance(item, dict)
        }
        dimension_scores: dict[str, int] = {}
        dimension_reports: list[DimensionReport] = []

        for dimension in rule.dimensions:
            report_data = reports_by_dimension.get(dimension, {})
            raw_score = report_data.get("score", raw_scores.get(dimension, 0))
            score = self._clamp_score(raw_score)
            dimension_scores[dimension] = score
            dimension_reports.append(
                DimensionReport(
                    dimension=dimension,
                    score=score,
                    analysis=self._text(report_data.get("analysis")),
                    evidence=self._text_list(report_data.get("evidence")),
                    suggestions=self._text_list(report_data.get("suggestions")),
                )
            )

        total_score = sum(dimension_scores.values())
        return EssayScoreResponse(
            total_score=total_score,
            dimension_scores=dimension_scores,
            dimension_reports=dimension_reports,
            rule_id=rule.rule_id,
        )

    @staticmethod
    def _clamp_score(value: Any) -> int:
        try:
            score = int(value)
        except (TypeError, ValueError):
            score = 0
        return max(0, min(3, score))

    @staticmethod
    def _text(value: Any) -> str:
        text = str(value or "").strip()
        return text or "No analysis was provided by the model."

    @staticmethod
    def _text_list(value: Any) -> list[str]:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = [str(value).strip()] if str(value or "").strip() else []
        return items
