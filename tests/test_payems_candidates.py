from config.payems_candidates import (
    FROZEN_VALIDATION_YEARS,
    LOCKED_OOS_YEAR,
    PAYEMS_A_FEATURES,
    PAYEMS_B_FEATURES,
    PAYEMS_CANDIDATES,
    PAYEMS_RESEARCH_ONLY_FEATURES,
)


def test_payems_a_is_frozen():
    assert PAYEMS_A_FEATURES == (
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_broad_revision_event",
    )


def test_payems_b_is_frozen():
    assert PAYEMS_B_FEATURES == (
        "payroll_change_1m",
        "payroll_revision_2m",
        "payroll_broad_revision_event",
    )


def test_payems_candidate_registry_is_exact():
    assert set(PAYEMS_CANDIDATES) == {
        "PAYEMS-A",
        "PAYEMS-B",
    }

    assert PAYEMS_CANDIDATES[
        "PAYEMS-A"
    ] == PAYEMS_A_FEATURES

    assert PAYEMS_CANDIDATES[
        "PAYEMS-B"
    ] == PAYEMS_B_FEATURES


def test_standardized_features_are_not_candidate_features():
    candidate_features = (
        set(PAYEMS_A_FEATURES)
        | set(PAYEMS_B_FEATURES)
    )

    assert "payroll_change_z" not in candidate_features
    assert "payroll_revision_z" not in candidate_features


def test_research_only_features_do_not_leak_into_payems_a():
    assert not (
        set(PAYEMS_A_FEATURES)
        & set(PAYEMS_RESEARCH_ONLY_FEATURES)
    )


def test_validation_period_is_frozen():
    assert FROZEN_VALIDATION_YEARS == (
        2023,
        2024,
        2025,
    )

    assert LOCKED_OOS_YEAR == 2026