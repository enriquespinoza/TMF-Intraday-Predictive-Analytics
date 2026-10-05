from config.payems_candidates import (
    LOCKED_OOS_YEAR,
    PAYEMS_A_FEATURES,
    PAYEMS_B_FEATURES,
)

from src.analysis.evaluate_payems_candidates import (
    MATCHED_SAMPLE_FEATURES,
    MODEL_SPECS,
)

from src.models.directional_classifier import (
    SELECTED_TECHNICAL_FEATURES,
)


def test_three_frozen_models_are_defined():
    assert set(MODEL_SPECS) == {
        "technical_v1_matched",
        "PAYEMS-A",
        "PAYEMS-B",
    }


def test_technical_control_is_exact_v1():
    assert MODEL_SPECS[
        "technical_v1_matched"
    ] == tuple(
        SELECTED_TECHNICAL_FEATURES
    )


def test_payems_a_adds_only_frozen_a():
    assert MODEL_SPECS[
        "PAYEMS-A"
    ] == (
        tuple(
            SELECTED_TECHNICAL_FEATURES
        )
        + tuple(PAYEMS_A_FEATURES)
    )


def test_payems_b_adds_only_frozen_b():
    assert MODEL_SPECS[
        "PAYEMS-B"
    ] == (
        tuple(
            SELECTED_TECHNICAL_FEATURES
        )
        + tuple(PAYEMS_B_FEATURES)
    )


def test_matched_sample_contains_both_candidates():
    expected = tuple(
        dict.fromkeys(
            tuple(PAYEMS_A_FEATURES)
            + tuple(PAYEMS_B_FEATURES)
        )
    )

    assert (
        MATCHED_SAMPLE_FEATURES
        == expected
    )


def test_2026_remains_locked():
    assert LOCKED_OOS_YEAR == 2026