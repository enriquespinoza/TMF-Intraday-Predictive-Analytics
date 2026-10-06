import pandas as pd

from config.payems_candidates import (
    PAYEMS_A_FEATURES,
)

from config.v3_final import (
    FINAL_V3_FEATURES,
    FINAL_V3_NAME,
    LOCKED_OOS_YEAR,
)

from src.analysis.evaluate_v3_locked_2026 import (
    OOS_START,
    TECHNICAL_V1_NAME,
)

from src.models.directional_classifier import (
    SELECTED_TECHNICAL_FEATURES,
)


def test_final_v3_is_still_exactly_frozen():
    assert FINAL_V3_FEATURES == (
        tuple(
            SELECTED_TECHNICAL_FEATURES
        )
        + tuple(
            PAYEMS_A_FEATURES
        )
    )


def test_final_model_name_is_frozen():
    assert FINAL_V3_NAME == "V3_FINAL"


def test_technical_benchmark_is_fixed():
    assert (
        TECHNICAL_V1_NAME
        == "technical_v1"
    )


def test_locked_year_is_2026():
    assert LOCKED_OOS_YEAR == 2026


def test_oos_boundary_is_january_1_2026():
    assert OOS_START == pd.Timestamp(
        "2026-01-01"
    )