import numpy as np
import pandas as pd

from src.features.technical_features import (
    build_technical_features,
)


def create_sample_data(rows: int = 50) -> pd.DataFrame:
    """Create deterministic OHLCV data for testing."""

    close = np.linspace(
        30,
        40,
        rows,
    )

    return pd.DataFrame(
        {
            "open": close - 0.10,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": np.linspace(
                1_000_000,
                2_000_000,
                rows,
            ),
        }
    )


def test_feature_columns_created():
    df = create_sample_data()

    result = build_technical_features(df)

    expected_columns = {
        "return_1d",
        "return_5d",
        "ema_9",
        "ema_21",
        "ema_spread_pct",
        "rsi_14",
        "momentum_5d_pct",
        "volatility_20d",
        "relative_volume",
        "distance_from_20d_high",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_row_count_preserved():
    df = create_sample_data()

    result = build_technical_features(df)

    assert len(result) == len(df)


def test_ema_spread_positive_in_uptrend():
    df = create_sample_data()

    result = build_technical_features(df)

    assert (
        result["ema_spread_pct"].iloc[-1]
        > 0
    )


def test_relative_volume_valid():
    df = create_sample_data()

    result = build_technical_features(df)

    valid = (
        result["relative_volume"]
        .dropna()
    )

    assert (valid > 0).all()
    