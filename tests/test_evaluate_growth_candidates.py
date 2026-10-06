from config.growth_candidates import (
    GROWTH_A_FEATURES,
    GROWTH_A_NAME,
    GROWTH_B_FEATURES,
    GROWTH_B_NAME,
    LOCKED_OOS_YEAR,
)

from config.payems_candidates import (
    PAYEMS_A_FEATURES,
)

from src.analysis.evaluate_growth_candidates import (
    MATCHED_SAMPLE_FEATURES,
    MODEL_SPECS,
    V3_BASE_FEATURES,
    V3_BASE_NAME,
)

from src.models.directional_classifier import (
    SELECTED_TECHNICAL_FEATURES,
)


def test_v3_base_is_technical_plus_payems_a():
    assert V3_BASE_FEATURES == (
        tuple(
            SELECTED_TECHNICAL_FEATURES
        )
        + tuple(
            PAYEMS_A_FEATURES
        )
    )


def test_exact_three_models():
    assert set(
        MODEL_SPECS
    ) == {
        V3_BASE_NAME,
        GROWTH_A_NAME,
        GROWTH_B_NAME,
    }


def test_growth_a_added_to_v3_base():
    assert (
        MODEL_SPECS[
            GROWTH_A_NAME
        ]
        == (
            V3_BASE_FEATURES
            + tuple(
                GROWTH_A_FEATURES
            )
        )
    )


def test_growth_b_added_to_v3_base():
    assert (
        MODEL_SPECS[
            GROWTH_B_NAME
        ]
        == (
            V3_BASE_FEATURES
            + tuple(
                GROWTH_B_FEATURES
            )
        )
    )


def test_matched_sample_contains_all_growth_inputs():
    expected = tuple(
        dict.fromkeys(
            tuple(
                PAYEMS_A_FEATURES
            )
            + tuple(
                GROWTH_A_FEATURES
            )
            + tuple(
                GROWTH_B_FEATURES
            )
        )
    )

    assert (
        MATCHED_SAMPLE_FEATURES
        == expected
    )


def test_2026_remains_locked():
    assert (
        LOCKED_OOS_YEAR
        == 2026
    )