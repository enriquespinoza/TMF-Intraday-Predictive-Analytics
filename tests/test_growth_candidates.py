from config.growth_candidates import (
    FINAL_NEW_MACRO_FAMILY,
    FROZEN_VALIDATION_YEARS,
    GROWTH_A_FEATURES,
    GROWTH_A_NAME,
    GROWTH_B_FEATURES,
    GROWTH_B_NAME,
    GROWTH_CANDIDATES,
    LOCKED_OOS_YEAR,
)


def test_growth_a_is_frozen():
    assert GROWTH_A_FEATURES == (
        "retail_sales_mom_pct",
        "retail_sales_3m_annualized_pct",
    )


def test_growth_b_is_frozen():
    assert GROWTH_B_FEATURES == (
        "industrial_production_mom_pct",
        "industrial_production_3m_annualized_pct",
    )


def test_candidate_registry_is_exact():
    assert GROWTH_CANDIDATES == {
        GROWTH_A_NAME:
            GROWTH_A_FEATURES,
        GROWTH_B_NAME:
            GROWTH_B_FEATURES,
    }


def test_validation_years_are_frozen():
    assert FROZEN_VALIDATION_YEARS == (
        2023,
        2024,
        2025,
    )


def test_2026_remains_locked():
    assert LOCKED_OOS_YEAR == 2026


def test_this_is_final_new_macro_family():
    assert FINAL_NEW_MACRO_FAMILY is True