"""Forward-return analysis for the TMF directional baseline.

Reproduces the logic used by TMF_Directional_V2_STUDY.ts and measures
TMF returns following each technical-score observation.

This module does NOT execute trades. Its purpose is to determine whether
the technical score contains information about future TMF returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FAST_EMA = 9
SLOW_EMA = 21
RSI_LENGTH = 14
MOMENTUM_LENGTH = 5
VOLUME_LENGTH = 20

BULLISH_RSI = 55
BEARISH_RSI = 45

FORWARD_HORIZONS = (1, 3, 5, 10, 20)


# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------

def calculate_rsi(close: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    """Calculate Wilder-style RSI."""

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------

def build_directional_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create the -5 to +5 TMF directional score."""

    data = df.copy()

    # EMA
    data["ema_fast"] = (
        data["close"]
        .ewm(span=FAST_EMA, adjust=False)
        .mean()
    )

    data["ema_slow"] = (
        data["close"]
        .ewm(span=SLOW_EMA, adjust=False)
        .mean()
    )

    # RSI
    data["rsi"] = calculate_rsi(data["close"])

    # Momentum
    data["momentum"] = (
        data["close"] -
        data["close"].shift(MOMENTUM_LENGTH)
    )

    # Relative volume
    data["avg_volume"] = (
        data["volume"]
        .rolling(VOLUME_LENGTH)
        .mean()
    )

    data["high_volume"] = (
        data["volume"] > data["avg_volume"]
    )

    # --------------------------------------------------------------
    # Component scores
    # --------------------------------------------------------------

    data["trend_score"] = np.select(
        [
            data["ema_fast"] > data["ema_slow"],
            data["ema_fast"] < data["ema_slow"],
        ],
        [1, -1],
        default=0,
    )

    # VWAP score will be added once intraday/session VWAP
    # is available in the dataset.
    if "vwap" in data.columns:
        data["vwap_score"] = np.select(
            [
                data["close"] > data["vwap"],
                data["close"] < data["vwap"],
            ],
            [1, -1],
            default=0,
        )
    else:
        data["vwap_score"] = 0

    data["rsi_score"] = np.select(
        [
            data["rsi"] > BULLISH_RSI,
            data["rsi"] < BEARISH_RSI,
        ],
        [1, -1],
        default=0,
    )

    data["momentum_score"] = np.select(
        [
            data["momentum"] > 0,
            data["momentum"] < 0,
        ],
        [1, -1],
        default=0,
    )

    positive_bar = data["close"] > data["close"].shift(1)
    negative_bar = data["close"] < data["close"].shift(1)

    data["volume_score"] = np.select(
        [
            data["high_volume"] & positive_bar,
            data["high_volume"] & negative_bar,
        ],
        [1, -1],
        default=0,
    )

    # Composite score
    data["directional_score"] = (
        data["trend_score"]
        + data["vwap_score"]
        + data["rsi_score"]
        + data["momentum_score"]
        + data["volume_score"]
    )

    return data


# ---------------------------------------------------------------------
# Forward Returns
# ---------------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate future returns for each configured horizon."""

    data = df.copy()

    for horizon in FORWARD_HORIZONS:
        data[f"forward_return_{horizon}"] = (
            data["close"].shift(-horizon) /
            data["close"] - 1
        )

    return data


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

def summarize_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize future returns by directional score."""

    rows = []

    for score in sorted(
        df["directional_score"].dropna().unique()
    ):
        subset = df[
            df["directional_score"] == score
        ]

        row = {
            "directional_score": int(score),
            "observations": len(subset),
        }

        for horizon in FORWARD_HORIZONS:
            column = f"forward_return_{horizon}"
            valid = subset[column].dropna()

            n = len(valid)

            mean_return = valid.mean()
            median_return = valid.median()
            std_return = valid.std(ddof=1)

            if n > 1:
                standard_error = std_return / np.sqrt(n)
                ci_95_low = (
                    mean_return - 1.96 * standard_error
                )
                ci_95_high = (
                    mean_return + 1.96 * standard_error
                )
            else:
                standard_error = np.nan
                ci_95_low = np.nan
                ci_95_high = np.nan

            row[f"n_{horizon}"] = n
            row[f"mean_return_{horizon}"] = mean_return
            row[f"median_return_{horizon}"] = median_return
            row[f"std_return_{horizon}"] = std_return
            row[f"se_return_{horizon}"] = standard_error
            row[f"ci95_low_{horizon}"] = ci_95_low
            row[f"ci95_high_{horizon}"] = ci_95_high

            row[f"positive_rate_{horizon}"] = (
                (valid > 0).mean()
                if n
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_forward_return_analysis(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete directional-score analysis."""

    scored = build_directional_score(df)

    scored = add_forward_returns(scored)

    summary = summarize_forward_returns(scored)

    return scored, summary