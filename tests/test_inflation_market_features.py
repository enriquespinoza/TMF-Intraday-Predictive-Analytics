import pandas as pd
import pytest

from src.features.inflation_market_features import (
    build_breakeven_features,
    merge_breakeven_features_to_market,
    prepare_breakeven_10y,
)


def make_breakeven_data(
    dates,
    values,
):
    return pd.DataFrame(
        {
            "reference_date": dates,
            "value": values,
            "series_name":
                "breakeven_10y",
            "series_id":
                "T10YIE",
            "category":
                "inflation_compensation",
            "source":
                "FRED",
            "frequency":
                "daily",
            "units":
                "percent",
        }
    )


def test_prepare_breakeven_data():
    data = make_breakeven_data(
        [
            "2025-01-03",
            "2025-01-02",
        ],
        [
            2.30,
            2.20,
        ],
    )

    result = prepare_breakeven_10y(
        data
    )

    assert (
        result.iloc[0][
            "reference_date"
        ]
        == pd.Timestamp(
            "2025-01-02"
        )
    )


def test_five_day_change_converts_to_basis_points():
    data = make_breakeven_data(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
        [
            2.20,
            2.21,
            2.22,
            2.23,
            2.24,
            2.30,
        ],
    )

    result = build_breakeven_features(
        data
    )

    assert result.iloc[-1][
        "breakeven_10y_change_5d_bp"
    ] == pytest.approx(
        10.0
    )


def test_holiday_missing_value_does_not_count_as_observation():
    data = make_breakeven_data(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
        ],
        [
            2.20,
            None,
            2.21,
            2.22,
            2.23,
            2.24,
            2.30,
        ],
    )

    result = build_breakeven_features(
        data
    )

    # Valid observations are:
    # 2.20, 2.21, 2.22, 2.23, 2.24, 2.30
    #
    # 2.30 - 2.20 = 0.10 percentage points
    # = 10 basis points.
    assert result.iloc[-1][
        "breakeven_10y_change_5d_bp"
    ] == pytest.approx(
        10.0
    )


def test_first_five_valid_rows_have_no_change():
    data = make_breakeven_data(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ],
        [
            2.20,
            2.21,
            2.22,
            2.23,
            2.24,
        ],
    )

    result = build_breakeven_features(
        data
    )

    assert (
        result[
            "breakeven_10y_change_5d_bp"
        ].isna().all()
    )


def test_market_merge_uses_latest_known_observation():
    features = pd.DataFrame(
        {
            "breakeven_state_date":
                pd.to_datetime(
                    [
                        "2025-01-10",
                        "2025-01-13",
                    ]
                ),
            "breakeven_10y": [
                2.30,
                2.35,
            ],
            "breakeven_10y_change_5d_bp": [
                5.0,
                8.0,
            ],
        }
    )

    market = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-01-10",
                        "2025-01-11",
                        "2025-01-13",
                    ]
                )
        }
    )

    merged = (
        merge_breakeven_features_to_market(
            market,
            features,
        )
    )

    assert merged.loc[
        0,
        "breakeven_10y",
    ] == pytest.approx(
        2.30
    )

    assert merged.loc[
        1,
        "breakeven_10y",
    ] == pytest.approx(
        2.30
    )

    assert merged.loc[
        2,
        "breakeven_10y",
    ] == pytest.approx(
        2.35
    )


def test_market_merge_never_uses_future_state():
    features = pd.DataFrame(
        {
            "breakeven_state_date":
                pd.to_datetime(
                    [
                        "2025-01-10",
                    ]
                ),
            "breakeven_10y": [
                2.30,
            ],
            "breakeven_10y_change_5d_bp": [
                5.0,
            ],
        }
    )

    market = pd.DataFrame(
        {
            "market_date":
                pd.to_datetime(
                    [
                        "2025-01-09",
                    ]
                )
        }
    )

    merged = (
        merge_breakeven_features_to_market(
            market,
            features,
        )
    )

    assert pd.isna(
        merged.loc[
            0,
            "breakeven_10y",
        ]
    )