import json
import tempfile
import unittest
from pathlib import Path

from app.services.essay_rule_loader import EssayRuleLoader
from app.services.essay_scoring_service import EssayScoringService


class FakeProvider:
    enabled = True
    last_error = ""

    def complete_json(self, system_prompt: str, user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return {
            "dimension_scores": {
                "Ideas & Elaboration": 3,
                "Organization": 2,
                "Fluency & Language": 2,
                "Audience Awareness": 1,
            },
            "dimension_reports": [
                {
                    "dimension": "Ideas & Elaboration",
                    "score": 3,
                    "analysis": "The essay develops its main idea with specific examples.",
                    "evidence": ["Specific reasons are explained instead of merely listed."],
                    "suggestions": ["Keep connecting each example to the central claim."],
                },
                {
                    "dimension": "Organization",
                    "score": 2,
                    "analysis": "The essay has a clear sequence but transitions are basic.",
                    "evidence": ["Ideas move in a mostly logical order."],
                    "suggestions": ["Use smoother transitions between body paragraphs."],
                },
                {
                    "dimension": "Fluency & Language",
                    "score": 2,
                    "analysis": "Sentences are readable with some repetition.",
                    "evidence": ["Several sentence openings repeat."],
                    "suggestions": ["Vary sentence structures."],
                },
                {
                    "dimension": "Audience Awareness",
                    "score": 1,
                    "analysis": "The essay only occasionally addresses the reader.",
                    "evidence": ["Most of the essay is written as personal narration."],
                    "suggestions": ["Use inclusive language or direct reader engagement."],
                },
            ],
        }


class EssayScoringTests(unittest.TestCase):
    def test_rule_loader_reads_dimensions_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            rule_path = Path(tmp) / "rule.json"
            rule_path.write_text(
                json.dumps(
                    {
                        "rule_id": "v17",
                        "rule_content": "Score each essay from 0 to 3 in each dimension.",
                        "dimensions": ["Ideas & Elaboration", "Organization"],
                    }
                ),
                encoding="utf-8",
            )

            rule = EssayRuleLoader(rule_path).load()

        self.assertEqual(rule.rule_id, "v17")
        self.assertEqual(rule.dimensions, ["Ideas & Elaboration", "Organization"])
        self.assertIn("Score each essay", rule.rule_content)

    def test_score_essay_returns_four_dimension_scores_and_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            rule_path = Path(tmp) / "rule.json"
            rule_path.write_text(
                json.dumps(
                    {
                        "rule_id": "v17",
                        "rule_content": "Score the essay.",
                        "dimensions": [
                            "Ideas & Elaboration",
                            "Organization",
                            "Fluency & Language",
                            "Audience Awareness",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = EssayScoringService(rule_path=rule_path, provider=FakeProvider())

            result = service.score_essay(
                "Dear local newspaper, computers help students learn because they give fast access to examples."
            )

        self.assertEqual(result.total_score, 8)
        self.assertEqual(result.dimension_scores["Ideas & Elaboration"], 3)
        self.assertEqual(len(result.dimension_reports), 4)
        self.assertEqual(result.dimension_reports[0].dimension, "Ideas & Elaboration")

    def test_score_essay_rejects_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            rule_path = Path(tmp) / "rule.json"
            rule_path.write_text(
                json.dumps(
                    {
                        "rule_id": "v17",
                        "rule_content": "Score the essay.",
                        "dimensions": [
                            "Ideas & Elaboration",
                            "Organization",
                            "Fluency & Language",
                            "Audience Awareness",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = EssayScoringService(rule_path=rule_path, provider=FakeProvider())

            with self.assertRaisesRegex(ValueError, "Essay text is required"):
                service.score_essay("  ")


if __name__ == "__main__":
    unittest.main()
