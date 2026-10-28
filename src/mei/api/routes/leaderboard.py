"""Model leaderboard, category bests, detail, curves, and comparison endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from mei.api.models import (
    CompareModelsResponse,
    LeaderboardResponse,
    ModelCurvesResponse,
    ModelDetailResponse,
)
from mei.api.store import RunStore

router = APIRouter(tags=["Leaderboard"])


def get_store() -> RunStore:
    return RunStore.get_instance()


@router.get("/api/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard(
    track: str = Query(
        "classification", description="Track: 'classification' or 'regression'"
    ),
    store: RunStore = Depends(get_store),
) -> LeaderboardResponse:
    """Return model leaderboard with real metrics, generalization gaps, and status flags."""
    try:
        return store.get_leaderboard(track=track)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed loading leaderboard for track '{track}': {e}",
        )


@router.get("/api/categories/best")
def get_categories_best(
    track: str = Query(
        "classification", description="Track: 'classification' or 'regression'"
    ),
    store: RunStore = Depends(get_store),
) -> Dict[str, Any]:
    """Return best performing model per architectural category."""
    try:
        return store.get_best_per_category(track=track)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed loading best models for track '{track}': {e}",
        )


@router.get("/api/models/{model_id}", response_model=ModelDetailResponse)
def get_model_detail(
    model_id: str,
    track: str = Query(
        "classification", description="Track: 'classification' or 'regression'"
    ),
    store: RunStore = Depends(get_store),
) -> ModelDetailResponse:
    """Return comprehensive metric breakdown, hyperparameters, and split details for a model."""
    try:
        data = store.get_model_detail(model_id=model_id, track=track)
        return ModelDetailResponse(**data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in active run for track '{track}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed loading model detail for '{model_id}': {e}"
        )


@router.get("/api/models/{model_id}/curves", response_model=ModelCurvesResponse)
def get_model_curves(
    model_id: str,
    track: str = Query(
        "classification", description="Track: 'classification' or 'regression'"
    ),
    store: RunStore = Depends(get_store),
) -> ModelCurvesResponse:
    """Return ROC/PR/calibration curves, deep learning epoch curves, or boosting iteration losses."""
    try:
        data = store.get_model_curves(model_id=model_id, track=track)
        return ModelCurvesResponse(**data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Curves for model '{model_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed loading curves for '{model_id}': {e}"
        )


@router.get("/api/compare", response_model=CompareModelsResponse)
def compare_models(
    ids: str = Query(
        ..., description="Comma-separated list of up to 4 model IDs to compare"
    ),
    track: str = Query(
        "classification", description="Track: 'classification' or 'regression'"
    ),
    store: RunStore = Depends(get_store),
) -> CompareModelsResponse:
    """Side-by-side metric and configuration comparison for up to 4 models."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if not id_list:
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'ids' must contain at least one model ID",
        )
    try:
        data = store.compare_models(model_ids=id_list, track=track)
        return CompareModelsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed comparing models: {e}")
