"""
Frozen growth/activity candidate specifications for V3 evaluation.

These candidate definitions were selected before any growth/activity
candidate was evaluated against TMF returns or the 2023-2025
validation folds.

This is the final new macro family permitted in V3.

Do not modify these specifications based on validation results.

2026 remains locked and must not be used for feature selection,
candidate selection, or model tuning.
"""

from __future__ import annotations


GROWTH_A_NAME = "GROWTH-A"
GROWTH_B_NAME = "GROWTH-B"


# Consumer/activity momentum.
#
# Constructed point-in-time from Advance Retail Sales vintages.
GROWTH_A_FEATURES = (
    "retail_sales_mom_pct",
    "retail_sales_3m_annualized_pct",
)


# Production/activity momentum.
#
# Constructed point-in-time from Industrial Production vintages.
GROWTH_B_FEATURES = (
    "industrial_production_mom_pct",
    "industrial_production_3m_annualized_pct",
)


GROWTH_CANDIDATES = {
    GROWTH_A_NAME: GROWTH_A_FEATURES,
    GROWTH_B_NAME: GROWTH_B_FEATURES,
}


FROZEN_VALIDATION_YEARS = (
    2023,
    2024,
    2025,
)

LOCKED_OOS_YEAR = 2026


# V3 stops introducing new macro families after this evaluation.
FINAL_NEW_MACRO_FAMILY = True