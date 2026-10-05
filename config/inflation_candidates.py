"""
Frozen inflation candidate specifications for V3 evaluation.

These candidate definitions were selected before any inflation
candidate was evaluated against TMF returns or against the
2023-2025 validation folds.

Do not modify these specifications based on validation results.

2026 remains locked and must not be used for feature selection,
candidate selection, or model tuning.
"""

from __future__ import annotations


INFLATION_A_NAME = "INFLATION-A"
INFLATION_B_NAME = "INFLATION-B"


# Market-implied inflation state.
#
# 10-year breakeven inflation is available daily and does not
# require release-calendar or ALFRED vintage reconstruction.
INFLATION_A_FEATURES = (
    "breakeven_10y",
    "breakeven_10y_change_5d_bp",
)


# Realized inflation state.
#
# These features must be reconstructed point-in-time from
# Core CPI vintages and aligned to CPI publication dates.
INFLATION_B_FEATURES = (
    "core_cpi_yoy_pct",
    "core_cpi_3m_annualized_pct",
)


INFLATION_CANDIDATES = {
    INFLATION_A_NAME: INFLATION_A_FEATURES,
    INFLATION_B_NAME: INFLATION_B_FEATURES,
}


FROZEN_VALIDATION_YEARS = (
    2023,
    2024,
    2025,
)

LOCKED_OOS_YEAR = 2026


# Other available inflation variables are intentionally excluded
# from the frozen V3 candidate search.
INFLATION_RESEARCH_ONLY_SERIES = (
    "cpi",
    "pce_price_index",
    "core_pce_price_index",
    "breakeven_5y",
    "forward_inflation_5y5y",
    "real_yield_5y",
    "real_yield_10y",
    "real_yield_20y",
    "real_yield_30y",
)