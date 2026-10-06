import pandas as pd
import pytest

from src.features.build_v3_inflation_dataset import (
    build_v3_inflation_dataset,
)


def test_combined_dataset_uses_backward_states():
    base = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-01-10",
                        "2025-01-13",
                    ]
                ),
            "payroll_change_1m":
                [100.0, 100.0],
            "payroll_revision_total":
                [10.0, 10.0],
            "payroll_revision_2m":
                [5.0, 5.0],
            "payroll_broad_revision_event":
                [0, 0],
        }
    )

    breakeven = pd.DataFrame(
        {
            "breakeven_state_date":
                pd.to_datetime(
                    [
                        "2025-01-09",
                        "2025-01-13",
                    ]
                ),
            "breakeven_10y":
                [2.30, 2.35],
            "breakeven_10y_change_5d_bp":
                [5.0, 8.0],
        }
    )

    cpi = pd.DataFrame(
        {
            "inflation_state_date":
                pd.to_datetime(
                    [
                        "2025-01-08",
                    ]
                ),
            "core_cpi_reference_date":
                pd.to_datetime(
                    [
                        "2024-12-01",
                    ]
                ),
            "core_cpi_yoy_pct":
                [3.0],
            "core_cpi_3m_annualized_pct":
                [2.8],
        }
    )

    result = (
        build_v3_inflation_dataset(
            base_data=base,
            core_cpi_features=cpi,
            breakeven_features=breakeven,
        )
    )

    assert len(result) == 2

    assert result.loc[
        0,
        "breakeven_10y",
    ] == pytest.approx(
        2.30
    )

    assert result.loc[
        1,
        "breakeven_10y",
    ] == pytest.approx(
        2.35
    )

    assert result[
        "core_cpi_yoy_pct"
    ].tolist() == [
        3.0,
        3.0,
    ]


def test_combined_dataset_preserves_market_rows():
    base = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-01-09",
                        "2025-01-10",
                    ]
                ),
            "payroll_change_1m":
                [100.0, 100.0],
            "payroll_revision_total":
                [10.0, 10.0],
            "payroll_revision_2m":
                [5.0, 5.0],
            "payroll_broad_revision_event":
                [0, 0],
        }
    )

    breakeven = pd.DataFrame(
        {
            "breakeven_state_date":
                pd.to_datetime(
                    ["2025-01-10"]
                ),
            "breakeven_10y":
                [2.30],
            "breakeven_10y_change_5d_bp":
                [5.0],
        }
    )

    cpi = pd.DataFrame(
        {
            "inflation_state_date":
                pd.to_datetime(
                    ["2025-01-08"]
                ),
            "core_cpi_reference_date":
                pd.to_datetime(
                    ["2024-12-01"]
                ),
            "core_cpi_yoy_pct":
                [3.0],
            "core_cpi_3m_annualized_pct":
                [2.8],
        }
    )

    result = (
        build_v3_inflation_dataset(
            base_data=base,
            core_cpi_features=cpi,
            breakeven_features=breakeven,
        )
    )

    assert len(result) == len(base)

    assert pd.isna(
        result.loc[
            0,
            "breakeven_10y",
        ]
    )


def test_same_day_macro_state_is_allowed():
    base = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    ["2025-02-12"]
                ),
            "payroll_change_1m":
                [100.0],
            "payroll_revision_total":
                [10.0],
            "payroll_revision_2m":
                [5.0],
            "payroll_broad_revision_event":
                [0],
        }
    )

    breakeven = pd.DataFrame(
        {
            "breakeven_state_date":
                pd.to_datetime(
                    ["2025-02-12"]
                ),
            "breakeven_10y":
                [2.40],
            "breakeven_10y_change_5d_bp":
                [4.0],
        }
    )

    cpi = pd.DataFrame(
        {
            "inflation_state_date":
                pd.to_datetime(
                    ["2025-02-12"]
                ),
            "core_cpi_reference_date":
                pd.to_datetime(
                    ["2025-01-01"]
                ),
            "core_cpi_yoy_pct":
                [3.1],
            "core_cpi_3m_annualized_pct":
                [2.9],
        }
    )

    result = (
        build_v3_inflation_dataset(
            base_data=base,
            core_cpi_features=cpi,
            breakeven_features=breakeven,
        )
    )

    assert result.loc[
        0,
        "breakeven_10y",
    ] == pytest.approx(
        2.40
    )

    assert result.loc[
        0,
        "core_cpi_yoy_pct",
    ] == pytest.approx(
        3.1
    )