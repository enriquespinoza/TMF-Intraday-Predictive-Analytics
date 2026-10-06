"""
Diagnose the role of EMA21 in the V4 TMF trading system.

Questions
---------
1. Does being below EMA21 actually identify worse future TMF outcomes?
2. Is a one-day break different from a persistent breakdown?
3. Does EMA21 add useful information when the raw exposure governor
   is otherwise GREEN?
4. Does blocking every below-EMA day discard attractive reversals?

This is diagnostic research only.

No Risk Governor thresholds are changed by this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_daily.csv"
)

SUMMARY_FILE = Path(
    "reports/"
    "V4_ema21_risk_diagnostic.csv"
)

STREAK_FILE = Path(
    "reports/"
    "V4_ema21_streak_diagnostic.csv"
)


def add_ema_streak(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count consecutive sessions below EMA21.

    Above-EMA observations receive streak = 0.
    """

    output = data.copy()

    below = (
        output["below_ema21"]
        .astype(bool)
    )

    groups = (
        below != below.shift()
    ).cumsum()

    streak = (
        below.groupby(groups)
        .cumcount()
        + 1
    )

    output[
        "below_ema21_streak"
    ] = np.where(
        below,
        streak,
        0,
    )

    return output


def add_forward_path_metrics(
    data: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Add future maximum adverse/favorable excursion.

    Measurements are based on future closes relative to today's
    close.

    Current-day information is not used as a future observation.
    """

    output = data.copy()

    future_returns = []

    for step in range(
        1,
        horizon + 1,
    ):
        future_close = (
            output["close"]
            .shift(-step)
        )

        future_return = (
            future_close
            / output["close"]
            - 1.0
        )

        future_returns.append(
            future_return
        )

    future_matrix = pd.concat(
        future_returns,
        axis=1,
    )

    output[
        "forward_5d_mae"
    ] = future_matrix.min(
        axis=1
    )

    output[
        "forward_5d_mfe"
    ] = future_matrix.max(
        axis=1
    )

    return output


def classify_streak(
    streak: int,
) -> str:
    """
    Diagnostic bins only.

    These are not proposed trading thresholds.
    """

    if streak == 0:
        return "above_ema"

    if streak == 1:
        return "below_1_day"

    if streak <= 3:
        return "below_2_to_3_days"

    if streak <= 5:
        return "below_4_to_5_days"

    return "below_6plus_days"


def summarize(
    data: pd.DataFrame,
    label: str,
) -> dict:
    """Summarize forward behavior for one sample."""

    return {
        "sample":
            label,

        "n":
            len(data),

        "mean_forward_return_1":
            data[
                "forward_return_1"
            ].mean(),

        "mean_forward_return_5":
            data[
                "forward_return_5"
            ].mean(),

        "median_forward_return_5":
            data[
                "forward_return_5"
            ].median(),

        "forward_5d_positive_rate":
            (
                data[
                    "forward_return_5"
                ]
                > 0
            ).mean(),

        "mean_forward_5d_mae":
            data[
                "forward_5d_mae"
            ].mean(),

        "median_forward_5d_mae":
            data[
                "forward_5d_mae"
            ].median(),

        "mean_forward_5d_mfe":
            data[
                "forward_5d_mfe"
            ].mean(),
    }


def main() -> None:
    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    data = (
        data
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    data = add_ema_streak(
        data
    )

    data = add_forward_path_metrics(
        data
    )

    data[
        "ema_streak_bucket"
    ] = data[
        "below_ema21_streak"
    ].map(
        classify_streak
    )

    valid = data[
        "forward_return_5"
    ].notna()

    data = (
        data.loc[
            valid
        ]
        .copy()
    )

    rows = []

    rows.append(
        summarize(
            data.loc[
                ~data[
                    "below_ema21"
                ].astype(bool)
            ],
            "ALL_ABOVE_EMA21",
        )
    )

    rows.append(
        summarize(
            data.loc[
                data[
                    "below_ema21"
                ].astype(bool)
            ],
            "ALL_BELOW_EMA21",
        )
    )

    raw_green = (
        data[
            "raw_exposure_state"
        ]
        == "GREEN"
    )

    rows.append(
        summarize(
            data.loc[
                raw_green
                & ~data[
                    "below_ema21"
                ].astype(bool)
            ],
            "RAW_GREEN_ABOVE_EMA21",
        )
    )

    rows.append(
        summarize(
            data.loc[
                raw_green
                & data[
                    "below_ema21"
                ].astype(bool)
            ],
            "RAW_GREEN_BELOW_EMA21",
        )
    )

    summary = pd.DataFrame(
        rows
    )

    streak_summary = (
        data.groupby(
            "ema_streak_bucket",
            as_index=False,
        )
        .agg(
            n=(
                "market_date",
                "size",
            ),

            mean_forward_return_1=(
                "forward_return_1",
                "mean",
            ),

            mean_forward_return_5=(
                "forward_return_5",
                "mean",
            ),

            median_forward_return_5=(
                "forward_return_5",
                "median",
            ),

            forward_5d_positive_rate=(
                "forward_return_5",
                lambda x:
                    (x > 0).mean(),
            ),

            mean_forward_5d_mae=(
                "forward_5d_mae",
                "mean",
            ),

            median_forward_5d_mae=(
                "forward_5d_mae",
                "median",
            ),

            mean_forward_5d_mfe=(
                "forward_5d_mfe",
                "mean",
            ),
        )
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    streak_summary.to_csv(
        STREAK_FILE,
        index=False,
    )

    print("=" * 78)
    print(
        "V4 EMA21 — RISK ROLE DIAGNOSTIC"
    )
    print("=" * 78)

    print(
        "\nABOVE / BELOW EMA21"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nBELOW-EMA PERSISTENCE"
    )

    print(
        streak_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nInterpretation guide:"
    )

    print(
        "Negative MAE means adverse movement "
        "after the observation."
    )

    print(
        "Positive MFE means favorable movement "
        "after the observation."
    )

    print(
        "RAW_GREEN_BELOW_EMA21 isolates days "
        "where market-risk signals were otherwise safe."
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {STREAK_FILE}"
    )


if __name__ == "__main__":
    main()