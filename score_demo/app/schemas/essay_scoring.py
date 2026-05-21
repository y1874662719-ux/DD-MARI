from pydantic import BaseModel, Field


class EssayScoreRequest(BaseModel):
    essay: str = Field(..., description="English essay text to score")


class DimensionReport(BaseModel):
    dimension: str
    score: int = Field(ge=0, le=3)
    analysis: str
    evidence: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class EssayScoreResponse(BaseModel):
    total_score: int = Field(ge=0, le=12)
    dimension_scores: dict[str, int] = Field(default_factory=dict)
    dimension_reports: list[DimensionReport] = Field(default_factory=list)
    rule_id: str = ""
