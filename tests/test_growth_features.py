import numpy as np
import pandas as pd
import pytest

from src.features.growth_features import (
    build_growth_feature_history,
    calculate_3m_annualized_pct,
    calculate_mom_pct,
    exact_reference_value,
    merge_growth_features_to_market,
)


def test_mom_calculation():
    result = calculate_mom_pct(
        102.0,
        100.0,
    )

    assert result == pytest.approx(
        2.0
    )


def test_three_month_annualized_calculation():
    result = calculate_3m_annualized_pct(
        103.0,
        100.0,
    )

    expected = (
        (1.03 ** 4)
        - 1
    ) * 100

    assert result == pytest.approx(
        expected
    )


def test_exact_reference_does_not_substitute():
    state = pd.DataFrame(
        {
            "reference_date":
                pd.to_datetime(
                    [
                        "2025-01-01",
                        "2025-03-01",
                    ]
                ),
            "value_as_known":
                [
                    100.0,
                    103.0,
                ],
        }
    )

    result = exact_reference_value(
        state,
        pd.Timestamp(
            "2025-02-01"
        ),
    )

    assert np.isnan(result)


def test_revision_changes_feature_only_after_revision_date():
    data = pd.DataFrame(
        {
            "series_name": [
                "retail_sales",
                "retail_sales",
                "retail_sales",
                "retail_sales",
                "retail_sales",
            ],
            "series_id": [
                "RSAFS",
                "RSAFS",
                "RSAFS",
                "RSAFS",
                "RSAFS",
            ],
            "reference_date":
                pd.to_datetime(
                    [
                        "2025-01-01",
                        "2025-02-01",
                        "2025-03-01",
                        "2025-04-01",
                        "2025-03-01",
                    ]
                ),
            "vintage_date":
                pd.to_datetime(
                    [
                        "2025-02-15",
                        "2025-03-15",
                        "2025-04-15",
                        "2025-05-15",
                        "2025-06-01",
                    ]
                ),
            "value_as_known": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
        }
    )

    history = (
        build_growth_feature_history(
            data,
            "retail_sales",
        )
    )

    before_revision = history.loc[
        history[
            "retail_sales_state_date"
        ]
        == pd.Timestamp(
            "2025-05-15"
        )
    ].iloc[0]

    after_revision = history.loc[
        history[
            "retail_sales_state_date"
        ]
        == pd.Timestamp(
            "2025-06-01"
        )
    ].iloc[0]

    assert (
        before_revision[
            "retail_sales_reference_date"
        ]
        == pd.Timestamp(
            "2025-04-01"
        )
    )

    assert (
        after_revision[
            "retail_sales_reference_date"
        ]
        == pd.Timestamp(
            "2025-04-01"
        )
    )

    # Revision to March changes the 3-month-state
    # calculations only once the revision is known.
    assert (
        before_revision[
            "retail_sales_mom_pct"
        ]
        != after_revision[
            "retail_sales_mom_pct"
        ]
    )

def test_market_merge_does_not_allow_same_day_state():
    market = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-05-15",
                        "2025-05-16",
                    ]
                )
        }
    )

    growth = pd.DataFrame(
        {
            "retail_sales_state_date":
                pd.to_datetime(
                    [
                        "2025-05-15",
                    ]
                ),
            "retail_sales_reference_date":
                pd.to_datetime(
                    [
                        "2025-04-01",
                    ]
                ),
            "retail_sales_level":
                [103.0],
            "retail_sales_mom_pct":
                [1.0],
            "retail_sales_3m_annualized_pct":
                [4.0],
        }
    )

    result = (
        merge_growth_features_to_market(
            market_data=market,
            growth_features=growth,
            series_name="retail_sales",
        )
    )

    assert pd.isna(
        result.loc[
            0,
            "retail_sales_mom_pct",
        ]
    )

    assert result.loc[
        1,
        "retail_sales_mom_pct",
    ] == pytest.approx(
        1.0
    )


def test_industrial_production_columns():
    data = pd.DataFrame(
        {
            "series_name": [
                "industrial_production",
            ] * 4,
            "series_id": [
                "INDPRO",
            ] * 4,
            "reference_date":
                pd.to_datetime(
                    [
                        "2025-01-01",
                        "2025-02-01",
                        "2025-03-01",
                        "2025-04-01",
                    ]
                ),
            "vintage_date":
                pd.to_datetime(
                    [
                        "2025-02-15",
                        "2025-03-15",
                        "2025-04-15",
                        "2025-05-15",
                    ]
                ),
            "value_as_known": [
                100.0,
                100.5,
                101.0,
                101.5,
            ],
        }
    )

    result = (
        build_growth_feature_history(
            data,
            "industrial_production",
        )
    )

    assert (
        "industrial_production_mom_pct"
        in result.columns
    )

    assert (
        "industrial_production_3m_annualized_pct"
        in result.columns
    )
    