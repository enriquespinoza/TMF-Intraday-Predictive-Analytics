from config.payems_candidates import (
    PAYEMS_A_FEATURES,
)

from config.v3_final import (
    FEATURE_RESEARCH_COMPLETE,
    FINAL_V3_FEATURES,
    FINAL_V3_MACRO_FEATURES,
    FINAL_V3_MODEL,
    FINAL_V3_NAME,
    FINAL_V3_TECHNICAL_FEATURES,
    LOCKED_OOS_YEAR,
)

from src.models.directional_classifier import (
    SELECTED_TECHNICAL_FEATURES,
)


def test_final_v3_name():
    assert FINAL_V3_NAME == (
        "V3_FINAL"
    )


def test_final_technical_features_are_frozen_v1():
    assert (
        FINAL_V3_TECHNICAL_FEATURES
        == tuple(
            SELECTED_TECHNICAL_FEATURES
        )
    )


def test_final_macro_features_are_payems_a():
    assert (
        FINAL_V3_MACRO_FEATURES
        == tuple(
            PAYEMS_A_FEATURES
        )
    )


def test_final_feature_set_is_exact():
    assert FINAL_V3_FEATURES == (
        tuple(
            SELECTED_TECHNICAL_FEATURES
        )
        + tuple(
            PAYEMS_A_FEATURES
        )
    )


def test_final_model_type_is_logistic():
    assert (
        FINAL_V3_MODEL[
            "model_type"
        ]
        == "logistic_regression"
    )


def test_2026_is_final_locked_oos_year():
    assert LOCKED_OOS_YEAR == 2026


def test_feature_research_is_closed():
    assert (
        FEATURE_RESEARCH_COMPLETE
        is True
    )