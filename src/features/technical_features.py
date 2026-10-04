"""Technical feature engineering for TMF predictive research."""

import numpy as np
import pandas as pd

from src.analysis.forward_returns import calculate_rsi


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create continuous technical features from OHLCV market data."""

    data = df.copy()

    # Returns
    data["return_1d"] = data["close"].pct_change()
    data["return_3d"] = data["close"].pct_change(3)
    data["return_5d"] = data["close"].pct_change(5)
    data["return_10d"] = data["close"].pct_change(10)

    # Exponential moving averages
    data["ema_9"] = (
        data["close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["ema_21"] = (
        data["close"]
        .ewm(span=21, adjust=False)
        .mean()
    )

    # Continuous trend measurements
    data["ema_spread_pct"] = (
        data["ema_9"] / data["ema_21"] - 1
    )

    data["close_vs_ema9"] = (
        data["close"] / data["ema_9"] - 1
    )

    data["close_vs_ema21"] = (
        data["close"] / data["ema_21"] - 1
    )

    # RSI
    data["rsi_14"] = calculate_rsi(
        data["close"],
        length=14,
    )

    data["rsi_centered"] = (
        data["rsi_14"] - 50
    ) / 50

    # Momentum
    data["momentum_5d_pct"] = (
        data["close"] /
        data["close"].shift(5)
        - 1
    )

    data["momentum_10d_pct"] = (
        data["close"] /
        data["close"].shift(10)
        - 1
    )

    # Realized volatility
    data["volatility_5d"] = (
        data["return_1d"]
        .rolling(5)
        .std()
    )

    data["volatility_20d"] = (
        data["return_1d"]
        .rolling(20)
        .std()
    )

    # Daily trading range
    data["range_pct"] = (
        data["high"] - data["low"]
    ) / data["close"]

    # Volume
    data["volume_avg_20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["relative_volume"] = (
        data["volume"] /
        data["volume_avg_20"]
    )

    data["log_volume"] = np.log1p(
        data["volume"]
    )

    # Position relative to recent price extremes
    data["rolling_high_20"] = (
        data["close"]
        .rolling(20)
        .max()
    )

    data["distance_from_20d_high"] = (
        data["close"] /
        data["rolling_high_20"]
        - 1
    )

    data["rolling_low_20"] = (
        data["close"]
        .rolling(20)
        .min()
    )

    data["distance_from_20d_low"] = (
        data["close"] /
        data["rolling_low_20"]
        - 1
    )

    return data