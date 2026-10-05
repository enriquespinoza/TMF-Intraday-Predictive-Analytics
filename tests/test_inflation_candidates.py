from config.inflation_candidates import (
    FROZEN_VALIDATION_YEARS,
    INFLATION_A_FEATURES,
    INFLATION_A_NAME,
    INFLATION_B_FEATURES,
    INFLATION_B_NAME,
    INFLATION_CANDIDATES,
    LOCKED_OOS_YEAR,
)


def test_inflation_a_is_frozen():
    assert INFLATION_A_FEATURES == (
        "breakeven_10y",
        "breakeven_10y_change_5d_bp",
    )


def test_inflation_b_is_frozen():
    assert INFLATION_B_FEATURES == (
        "core_cpi_yoy_pct",
        "core_cpi_3m_annualized_pct",
    )


def test_candidate_registry_is_exact():
    assert INFLATION_CANDIDATES == {
        INFLATION_A_NAME:
            INFLATION_A_FEATURES,
        INFLATION_B_NAME:
            INFLATION_B_FEATURES,
    }


def test_validation_years_are_frozen():
    assert FROZEN_VALIDATION_YEARS == (
        2023,
        2024,
        2025,
    )


def test_2026_remains_locked():
    assert LOCKED_OOS_YEAR == 2026