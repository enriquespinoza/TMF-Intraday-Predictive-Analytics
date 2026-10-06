import pandas as pd
import pytest

from src.features.build_v3_growth_dataset import (
    build_v3_growth_dataset,
)


def make_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-05-15",
                        "2025-05-16",
                    ]
                ),
            "payroll_change_1m":
                [100.0, 100.0],
            "payroll_revision_total":
                [10.0, 10.0],
            "payroll_broad_revision_event":
                [0, 0],
        }
    )


def make_retail() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "retail_sales_state_date":
                pd.to_datetime(
                    ["2025-05-15"]
                ),
            "retail_sales_reference_date":
                pd.to_datetime(
                    ["2025-04-01"]
                ),
            "retail_sales_level":
                [700000.0],
            "retail_sales_mom_pct":
                [0.5],
            "retail_sales_3m_annualized_pct":
                [2.0],
        }
    )


def make_industrial() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "industrial_production_state_date":
                pd.to_datetime(
                    ["2025-05-15"]
                ),
            "industrial_production_reference_date":
                pd.to_datetime(
                    ["2025-04-01"]
                ),
            "industrial_production_level":
                [103.0],
            "industrial_production_mom_pct":
                [0.2],
            "industrial_production_3m_annualized_pct":
                [1.5],
        }
    )


def test_combined_dataset_preserves_market_rows():
    result = build_v3_growth_dataset(
        base_data=make_base(),
        retail_sales_features=make_retail(),
        industrial_production_features=make_industrial(),
    )

    assert len(result) == 2


def test_same_day_growth_states_are_not_allowed():
    result = build_v3_growth_dataset(
        base_data=make_base(),
        retail_sales_features=make_retail(),
        industrial_production_features=make_industrial(),
    )

    assert pd.isna(
        result.loc[
            0,
            "retail_sales_mom_pct",
        ]
    )

    assert pd.isna(
        result.loc[
            0,
            "industrial_production_mom_pct",
        ]
    )


def test_next_day_receives_growth_states():
    result = build_v3_growth_dataset(
        base_data=make_base(),
        retail_sales_features=make_retail(),
        industrial_production_features=make_industrial(),
    )

    assert result.loc[
        1,
        "retail_sales_mom_pct",
    ] == pytest.approx(
        0.5
    )

    assert result.loc[
        1,
        "industrial_production_mom_pct",
    ] == pytest.approx(
        0.2
    )


def test_reference_periods_are_prior_to_market_date():
    result = build_v3_growth_dataset(
        base_data=make_base(),
        retail_sales_features=make_retail(),
        industrial_production_features=make_industrial(),
    )

    row = result.iloc[1]

    assert (
        row[
            "retail_sales_reference_date"
        ]
        < row["market_date"]
    )

    assert (
        row[
            "industrial_production_reference_date"
        ]
        < row["market_date"]
    )