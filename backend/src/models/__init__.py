"""
Model Zoo — Academic Intelligence Framework (IT23164680).

Importing this package triggers all @register decorators,
populating ModelRegistry with every model across Categories A-F.

Usage:
    from backend.src.models import ModelRegistry
    model = ModelRegistry.build("lstm")
"""

from __future__ import annotations

# ── Import all model modules to trigger @register decorators ──────────────────
# Order matters only for readability; all imports are side-effect registrations.
from backend.src.models import tabular_models  # Categories A, B, C, F
from backend.src.models import mlp_tabular     # Category D
from backend.src.models import sequence_models  # Category E

# ── Re-export key symbols ─────────────────────────────────────────────────────
from backend.src.models.base import (
    BaseAcademicModel,
    ModelRegistry,
    CATEGORIES,
    register,
)

__all__ = [
    "BaseAcademicModel",
    "ModelRegistry",
    "CATEGORIES",
    "register",
    "tabular_models",
    "mlp_tabular",
    "sequence_models",
]
