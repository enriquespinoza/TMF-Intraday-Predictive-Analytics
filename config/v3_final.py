"""
Final frozen V3 model specification.

Feature research is complete.

Final V3 consists only of:

    Technical V1
    +
    PAYEMS-A

All Treasury, inflation, and growth/activity candidate families
were evaluated and rejected.

No feature additions, removals, transformations, or hyperparameter
changes are permitted after this freeze based on 2026 performance.

2026 is the locked out-of-sample evaluation year.
"""

from __future__ import annotations

from config.payems_candidates import (
    PAYEMS_A_FEATURES,
)

from src.models.directional_classifier import (
    SELECTED_TECHNICAL_FEATURES,
)


FINAL_V3_NAME = "V3_FINAL"


FINAL_V3_TECHNICAL_FEATURES = tuple(
    SELECTED_TECHNICAL_FEATURES
)


FINAL_V3_MACRO_FEATURES = tuple(
    PAYEMS_A_FEATURES
)


FINAL_V3_FEATURES = (
    FINAL_V3_TECHNICAL_FEATURES
    + FINAL_V3_MACRO_FEATURES
)


FINAL_V3_MODEL = {
    "model_type":
        "logistic_regression",

    "features":
        FINAL_V3_FEATURES,

    "technical_features":
        FINAL_V3_TECHNICAL_FEATURES,

    "macro_features":
        FINAL_V3_MACRO_FEATURES,
}


DEVELOPMENT_END = "2025-12-31"

LOCKED_OOS_YEAR = 2026

FEATURE_RESEARCH_COMPLETE = True