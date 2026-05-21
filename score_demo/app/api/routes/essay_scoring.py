from fastapi import APIRouter, HTTPException

from app.schemas.essay_scoring import EssayScoreRequest, EssayScoreResponse
from app.services.essay_scoring_service import EssayScoringService

router = APIRouter(prefix="/essay", tags=["essay-scoring"])
service = EssayScoringService()


@router.post("/score", response_model=EssayScoreResponse)
def score_essay(payload: EssayScoreRequest) -> EssayScoreResponse:
    try:
        return service.score_essay(payload.essay)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
