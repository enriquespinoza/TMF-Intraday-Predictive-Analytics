"""
Build the V3 point-in-time dataset for frozen growth evaluation.

Base dataset:
    Technical features + PAYEMS features

Adds:

GROWTH-A
    retail_sales_mom_pct
    retail_sales_3m_annualized_pct

GROWTH-B
    industrial_production_mom_pct
    industrial_production_3m_annualized_pct

Methodology
-----------
- Growth states are merged strictly backward.
- Same-day growth vintage states are not allowed.
- No future state may enter a market observation.
- 2026 may remain in the generated dataset but stays locked
  downstream for model selection and validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.growth_features import (
    merge_growth_features_to_market,
)


BASE_DATA_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_features.csv"
)

RETAIL_FEATURE_FILE = Path(
    "data/features/"
    "retail_sales_point_in_time_features.csv"
)

INDUSTRIAL_FEATURE_FILE = Path(
    "data/features/"
    "industrial_production_point_in_time_features.csv"
)

OUTPUT_FILE = Path(
    "data/features/"
    "TMF_1d_technical_PAYEMS_growth_features.csv"
)


def load_base_data() -> pd.DataFrame:
    """Load Technical + PAYEMS feature dataset."""

    data = pd.read_csv(
        BASE_DATA_FILE,
        parse_dates=["market_date"],
    )

    if data.empty:
        raise ValueError(
            "Base Technical + PAYEMS dataset is empty."
        )

    if data["market_date"].duplicated().any():
        raise ValueError(
            "Duplicate market dates found in base dataset."
        )

    return (
        data.sort_values("market_date")
        .reset_index(drop=True)
    )


def load_retail_sales_features() -> pd.DataFrame:
    """Load point-in-time retail-sales features."""

    return pd.read_csv(
        RETAIL_FEATURE_FILE,
        parse_dates=[
            "retail_sales_state_date",
            "retail_sales_reference_date",
        ],
    )


def load_industrial_production_features() -> pd.DataFrame:
    """Load point-in-time industrial-production features."""

    return pd.read_csv(
        INDUSTRIAL_FEATURE_FILE,
        parse_dates=[
            "industrial_production_state_date",
            "industrial_production_reference_date",
        ],
    )


def validate_combined_data(
    data: pd.DataFrame,
) -> None:
    """Validate point-in-time integrity."""

    required_columns = {
        "market_date",
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_broad_revision_event",
        "retail_sales_state_date",
        "retail_sales_reference_date",
        "retail_sales_mom_pct",
        "retail_sales_3m_annualized_pct",
        "industrial_production_state_date",
        "industrial_production_reference_date",
        "industrial_production_mom_pct",
        "industrial_production_3m_annualized_pct",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            "Combined V3 growth dataset missing "
            f"columns: {sorted(missing)}"
        )

    if data["market_date"].duplicated().any():
        raise ValueError(
            "Combined V3 growth dataset contains "
            "duplicate market dates."
        )

    retail_violation = (
        data["retail_sales_state_date"].notna()
        & (
            data["retail_sales_state_date"]
            >= data["market_date"]
        )
    )

    if retail_violation.any():
        raise AssertionError(
            "Non-prior retail-sales state detected."
        )

    industrial_violation = (
        data[
            "industrial_production_state_date"
        ].notna()
        & (
            data[
                "industrial_production_state_date"
            ]
            >= data["market_date"]
        )
    )

    if industrial_violation.any():
        raise AssertionError(
            "Non-prior industrial-production "
            "state detected."
        )

    retail_reference_future = (
        data[
            "retail_sales_reference_date"
        ].notna()
        & (
            data[
                "retail_sales_reference_date"
            ]
            > data["market_date"]
        )
    )

    if retail_reference_future.any():
        raise AssertionError(
            "Future retail-sales reference period detected."
        )

    industrial_reference_future = (
        data[
            "industrial_production_reference_date"
        ].notna()
        & (
            data[
                "industrial_production_reference_date"
            ]
            > data["market_date"]
        )
    )

    if industrial_reference_future.any():
        raise AssertionError(
            "Future industrial-production "
            "reference period detected."
        )


def build_v3_growth_dataset(
    base_data: pd.DataFrame,
    retail_sales_features: pd.DataFrame,
    industrial_production_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge both frozen growth families onto V3 base data."""

    combined = (
        merge_growth_features_to_market(
            market_data=base_data,
            growth_features=retail_sales_features,
            series_name="retail_sales",
        )
    )

    combined = (
        merge_growth_features_to_market(
            market_data=combined,
            growth_features=industrial_production_features,
            series_name="industrial_production",
        )
    )

    validate_combined_data(
        combined
    )

    return (
        combined.sort_values(
            "market_date"
        )
        .reset_index(drop=True)
    )


def main() -> None:
    """Build and save the combined growth dataset."""

    base = load_base_data()

    retail = (
        load_retail_sales_features()
    )

    industrial = (
        load_industrial_production_features()
    )

    combined = (
        build_v3_growth_dataset(
            base_data=base,
            retail_sales_features=retail,
            industrial_production_features=industrial,
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    growth_columns = [
        "retail_sales_mom_pct",
        "retail_sales_3m_annualized_pct",
        "industrial_production_mom_pct",
        "industrial_production_3m_annualized_pct",
    ]

    print(
        "Saved:",
        OUTPUT_FILE,
    )

    print(
        "Shape:",
        combined.shape,
    )

    print(
        "Market range:",
        combined["market_date"].min().date(),
        "to",
        combined["market_date"].max().date(),
    )

    print(
        "\nMissing growth values:"
    )

    print(
        combined[
            growth_columns
        ].isna().sum()
    )

    print(
        "\nNon-prior state violations:"
    )

    print(
        "Retail sales:",
        (
            combined["retail_sales_state_date"]
            >= combined["market_date"]
        ).sum(),
    )

    print(
        "Industrial production:",
        (
            combined[
                "industrial_production_state_date"
            ]
            >= combined["market_date"]
        ).sum(),
    )


if __name__ == "__main__":
    main()