import numpy as np
import pandas as pd
import pytest

from src.features.inflation_features import (
    build_core_cpi_feature_history,
    calculate_3m_annualized_pct,
    calculate_yoy_pct,
    core_cpi_features_as_of,
    exact_reference_value,
    merge_core_cpi_features_to_market,
)


def make_vintage_data(
    records,
):
    return pd.DataFrame(
        [
            {
                "series_name": "core_cpi",
                "series_id": "CPILFESL",
                "reference_date":
                    reference_date,
                "vintage_date":
                    vintage_date,
                "value_as_known":
                    value,
            }
            for (
                reference_date,
                vintage_date,
                value,
            ) in records
        ]
    )


def test_yoy_calculation():
    result = calculate_yoy_pct(
        current_value=110.0,
        year_ago_value=100.0,
    )

    assert result == pytest.approx(
        10.0
    )


def test_three_month_annualized_calculation():
    result = (
        calculate_3m_annualized_pct(
            current_value=102.0,
            three_month_ago_value=100.0,
        )
    )

    expected = (
        (1.02 ** 4) - 1
    ) * 100

    assert result == pytest.approx(
        expected
    )


def test_exact_reference_value_does_not_substitute_missing_month():
    state = pd.DataFrame(
        {
            "reference_date":
                pd.to_datetime(
                    [
                        "2025-08-01",
                        "2025-09-01",
                        "2025-11-01",
                    ]
                ),
            "value_as_known": [
                328.0,
                330.0,
                331.0,
            ],
        }
    )

    value = exact_reference_value(
        state,
        pd.Timestamp(
            "2025-10-01"
        ),
    )

    assert np.isnan(value)


def test_missing_october_does_not_break_november_three_month_feature():
    data = make_vintage_data(
        [
            (
                "2024-11-01",
                "2024-12-11",
                320.0,
            ),
            (
                "2025-08-01",
                "2025-09-11",
                329.0,
            ),
            (
                "2025-09-01",
                "2025-10-24",
                330.0,
            ),
            (
                "2025-10-01",
                "2025-12-18",
                np.nan,
            ),
            (
                "2025-11-01",
                "2025-12-18",
                331.0,
            ),
        ]
    )

    features = core_cpi_features_as_of(
        data,
        "2025-12-18",
    )

    expected_yoy = (
        (331.0 / 320.0) - 1
    ) * 100

    expected_3m = (
        (
            (331.0 / 329.0) ** 4
        )
        - 1
    ) * 100

    assert (
        features[
            "core_cpi_reference_date"
        ]
        == pd.Timestamp("2025-11-01")
    )

    assert features[
        "core_cpi_yoy_pct"
    ] == pytest.approx(
        expected_yoy
    )

    assert features[
        "core_cpi_3m_annualized_pct"
    ] == pytest.approx(
        expected_3m
    )


def test_revision_changes_feature_only_after_revision_date():
    data = make_vintage_data(
        [
            (
                "2024-01-01",
                "2024-02-13",
                100.0,
            ),
            (
                "2025-01-01",
                "2025-02-12",
                104.0,
            ),
            (
                "2024-01-01",
                "2025-02-20",
                101.0,
            ),
        ]
    )

    before = core_cpi_features_as_of(
        data,
        "2025-02-19",
    )

    after = core_cpi_features_as_of(
        data,
        "2025-02-20",
    )

    expected_before = (
        (104.0 / 100.0) - 1
    ) * 100

    expected_after = (
        (104.0 / 101.0) - 1
    ) * 100

    assert before[
        "core_cpi_yoy_pct"
    ] == pytest.approx(
        expected_before
    )

    assert after[
        "core_cpi_yoy_pct"
    ] == pytest.approx(
        expected_after
    )


def test_feature_history_uses_vintage_dates():
    data = make_vintage_data(
        [
            (
                "2024-01-01",
                "2024-02-13",
                100.0,
            ),
            (
                "2025-01-01",
                "2025-02-12",
                104.0,
            ),
            (
                "2024-01-01",
                "2025-02-20",
                101.0,
            ),
        ]
    )

    history = (
        build_core_cpi_feature_history(
            data
        )
    )

    assert list(
        history[
            "inflation_state_date"
        ]
    ) == list(
        pd.to_datetime(
            [
                "2024-02-13",
                "2025-02-12",
                "2025-02-20",
            ]
        )
    )


def test_market_merge_is_backward_only():
    features = pd.DataFrame(
        {
            "inflation_state_date":
                pd.to_datetime(
                    [
                        "2025-01-10",
                        "2025-02-12",
                    ]
                ),
            "core_cpi_reference_date":
                pd.to_datetime(
                    [
                        "2024-12-01",
                        "2025-01-01",
                    ]
                ),
            "core_cpi_yoy_pct": [
                3.0,
                2.8,
            ],
            "core_cpi_3m_annualized_pct": [
                3.2,
                2.5,
            ],
        }
    )

    market = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-02-11",
                        "2025-02-12",
                        "2025-02-13",
                    ]
                )
        }
    )

    merged = (
        merge_core_cpi_features_to_market(
            market,
            features,
        )
    )

    assert (
        merged.loc[
            0,
            "inflation_state_date",
        ]
        == pd.Timestamp(
            "2025-01-10"
        )
    )

    assert (
        merged.loc[
            1,
            "inflation_state_date",
        ]
        == pd.Timestamp(
            "2025-02-12"
        )
    )

    assert (
        merged.loc[
            2,
            "inflation_state_date",
        ]
        == pd.Timestamp(
            "2025-02-12"
        )
    )