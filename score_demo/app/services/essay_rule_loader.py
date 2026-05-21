from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EssayRule:
    rule_id: str
    rule_content: str
    dimensions: list[str]


class EssayRuleLoader:
    def __init__(self, rule_path: str | Path) -> None:
        self.rule_path = Path(rule_path)

    def load(self) -> EssayRule:
        if not self.rule_path.exists():
            raise FileNotFoundError(f"Scoring rule file was not found: {self.rule_path}")

        data = json.loads(self.rule_path.read_text(encoding="utf-8"))
        rule_id = str(data.get("rule_id", "")).strip()
        rule_content = str(data.get("rule_content", "")).strip()
        dimensions = [str(item).strip() for item in data.get("dimensions", []) if str(item).strip()]

        if not rule_content:
            raise ValueError("Scoring rule content is empty.")
        if not dimensions:
            raise ValueError("Scoring rule dimensions are empty.")

        return EssayRule(rule_id=rule_id, rule_content=rule_content, dimensions=dimensions)
