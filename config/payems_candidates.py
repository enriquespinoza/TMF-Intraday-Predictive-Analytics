"""
Frozen PAYEMS candidate specifications for V3 evaluation.

These feature sets were selected before any PAYEMS candidate was
evaluated on the 2023, 2024, or 2025 validation folds.

Do not modify these specifications based on validation results.

2026 remains locked and must not be used for feature selection,
candidate selection, or model tuning.
"""

from __future__ import annotations


PAYEMS_A_NAME = "PAYEMS-A"
PAYEMS_B_NAME = "PAYEMS-B"


PAYEMS_A_FEATURES = (
    "payroll_change_1m",
    "payroll_revision_total",
    "payroll_broad_revision_event",
)


PAYEMS_B_FEATURES = (
    "payroll_change_1m",
    "payroll_revision_2m",
    "payroll_broad_revision_event",
)


PAYEMS_CANDIDATES = {
    PAYEMS_A_NAME: PAYEMS_A_FEATURES,
    PAYEMS_B_NAME: PAYEMS_B_FEATURES,
}


PAYEMS_RESEARCH_ONLY_FEATURES = (
    "payroll_change_z",
    "payroll_revision_z",
    "payroll_revision_1m",
    "payroll_revised_periods",
    "payroll_initial_periods",
    "days_since_payroll_release",
    "payroll_release_day",
)


FROZEN_VALIDATION_YEARS = (
    2023,
    2024,
    2025,
)


LOCKED_OOS_YEAR = 2026