"""Longitudinal engagement patterns and SHAP risk driver insights endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from mei.api.models import EngagementPatternsResponse, RiskDriversResponse
from mei.api.store import RunStore

router = APIRouter(tags=["Insights"])


def get_store() -> RunStore:
    return RunStore.get_instance()


@router.get("/api/patterns", response_model=EngagementPatternsResponse)
def get_patterns(store: RunStore = Depends(get_store)) -> EngagementPatternsResponse:
    """Return hidden Markov states, transition dynamics, behavioral patterns, trajectories, and risk rates."""
    try:
        data = store.get_patterns()
        return EngagementPatternsResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed loading engagement patterns: {e}"
        )


@router.get("/api/drivers", response_model=RiskDriversResponse)
def get_drivers(store: RunStore = Depends(get_store)) -> RiskDriversResponse:
    """Return global feature rankings, grouped importance, fold consistency, and web point clouds."""
    try:
        data = store.get_drivers()
        return RiskDriversResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed loading explainability risk drivers: {e}"
        )
