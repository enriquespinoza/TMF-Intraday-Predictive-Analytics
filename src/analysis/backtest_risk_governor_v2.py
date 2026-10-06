"""
Historical evaluation of V4 Risk Governor V2.

This evaluates V2 independently from the future opportunity engine.

V2 differs from V1 by:

- separating entry blocking from exposure control
- excluding EMA21/data staleness from exposure reduction
- applying chronological recovery hysteresis
- supporting an intraday emergency interface

No intraday emergency signals are supplied in this historical EOD
evaluation.

State observed at close t controls exposure during trading day t+1.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.risk_governor_v2 import (
    ExposureState,
    GovernorMemory,
    RiskInputsV2,
    evaluate_risk_v2,
)


INPUT_FILE = Path(
    "data/features/"
    "TMF_1d_risk_governor_features.csv"
)

V1_SUMMARY_FILE = Path(
    "reports/"
    "V4_risk_governor_overlay_summary.csv"
)

DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_daily.csv"
)

STATE_SUMMARY_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_state_summary.csv"
)

OVERLAY_SUMMARY_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_overlay_summary.csv"
)

REASON_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_reason_frequency.csv"
)


TRANSACTION_COST_BPS = 5.0


REQUIRED_NUMERIC = [
    "tmf_return_1d_pct",
    "change_30y_1d_bp",
    "change_30y_5d_bp",
    "real_yield_10y_change_5d_bp",
    "volatility_ratio_5d_20d",
]


def maximum_drawdown(
    returns: pd.Series,
) -> float:
    """Calculate maximum drawdown."""

    wealth = (
        1.0
        + returns.fillna(0.0)
    ).cumprod()

    running_peak = (
        wealth.cummax()
    )

    drawdown = (
        wealth
        / running_peak
        - 1.0
    )

    return float(
        drawdown.min()
    )


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized compound return."""

    clean = returns.dropna()

    if clean.empty:
        return np.nan

    growth = (
        1.0 + clean
    ).prod()

    years = (
        len(clean)
        / periods_per_year
    )

    if (
        growth <= 0
        or years <= 0
    ):
        return np.nan

    return float(
        growth ** (
            1.0 / years
        )
        - 1.0
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility."""

    clean = returns.dropna()

    if clean.empty:
        return np.nan

    return float(
        clean.std(ddof=1)
        * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized zero-risk-free Sharpe."""

    clean = returns.dropna()

    std = clean.std(ddof=1)

    if (
        clean.empty
        or std == 0
        or pd.isna(std)
    ):
        return np.nan

    return float(
        clean.mean()
        / std
        * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio."""

    clean = returns.dropna()

    downside = clean.loc[
        clean < 0
    ]

    if downside.empty:
        return np.nan

    downside_std = (
        downside.std(ddof=1)
    )

    if (
        downside_std == 0
        or pd.isna(downside_std)
    ):
        return np.nan

    return float(
        clean.mean()
        / downside_std
        * np.sqrt(periods_per_year)
    )


def calmar_ratio(
    annual_return: float,
    max_drawdown: float,
) -> float:
    """Calculate Calmar ratio."""

    if (
        pd.isna(annual_return)
        or max_drawdown >= 0
    ):
        return np.nan

    return float(
        annual_return
        / abs(max_drawdown)
    )


def make_inputs(
    row: pd.Series,
) -> RiskInputsV2:
    """Convert one historical row to V2 inputs."""

    return RiskInputsV2(
        tmf_return_1d_pct=float(
            row["tmf_return_1d_pct"]
        ),

        yield_30y_change_1d_bp=float(
            row["change_30y_1d_bp"]
        ),

        yield_30y_change_5d_bp=float(
            row["change_30y_5d_bp"]
        ),

        real_yield_10y_change_5d_bp=float(
            row[
                "real_yield_10y_change_5d_bp"
            ]
        ),

        volatility_ratio_5d_20d=float(
            row[
                "volatility_ratio_5d_20d"
            ]
        ),

        below_ema21=bool(
            row["below_ema21"]
        ),

        critical_data_stale=bool(
            row["critical_data_stale"]
        ),

        # Market-state test only.
        position_open=False,

        drawdown_from_entry_pct=None,
    )


def evaluate_chronologically(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply Risk Governor V2 sequentially.

    GovernorMemory is carried from one market day to the next.
    """

    output = (
        data.copy()
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    memory = GovernorMemory()

    records = []

    for _, row in output.iterrows():

        decision, memory = (
            evaluate_risk_v2(
                inputs=make_inputs(row),
                memory=memory,
            )
        )

        records.append(
            {
                "raw_exposure_state":
                    decision.raw_exposure_state.value,

                "exposure_state":
                    decision.exposure_state.value,

                "entry_blocked":
                    decision.entry_blocked,

                "new_entries_allowed":
                    decision.new_entries_allowed,

                "max_position_multiplier":
                    decision.max_position_multiplier,

                "exit_required":
                    decision.exit_required,

                "cooldown_required":
                    decision.cooldown_required,

                "yellow_warning_count":
                    decision.yellow_warning_count,

                "recovery_streak":
                    decision.recovery_streak,

                "exposure_reasons":
                    " | ".join(
                        decision.exposure_reasons
                    ),

                "entry_block_reasons":
                    " | ".join(
                        decision.entry_block_reasons
                    ),
            }
        )

    decisions = pd.DataFrame(
        records
    )

    return pd.concat(
        [
            output.reset_index(drop=True),
            decisions,
        ],
        axis=1,
    )


def add_state_run_lengths(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate consecutive effective-state duration."""

    output = data.copy()

    groups = (
        output["exposure_state"]
        != output[
            "exposure_state"
        ].shift()
    ).cumsum()

    output[
        "state_run_length"
    ] = (
        output.groupby(groups)
        .cumcount()
        + 1
    )

    return output


def add_overlay_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply effective V2 exposure on the following market day.
    """

    output = data.copy()

    output[
        "strategy_exposure"
    ] = (
        output[
            "max_position_multiplier"
        ]
        .shift(1)
        .fillna(0.0)
    )

    output[
        "turnover"
    ] = (
        output[
            "strategy_exposure"
        ]
        .diff()
        .abs()
    )

    output[
        "turnover"
    ] = output[
        "turnover"
    ].fillna(
        output[
            "strategy_exposure"
        ].abs()
    )

    cost_rate = (
        TRANSACTION_COST_BPS
        / 10000.0
    )

    output[
        "transaction_cost"
    ] = (
        output["turnover"]
        * cost_rate
    )

    output[
        "overlay_return_gross"
    ] = (
        output[
            "strategy_exposure"
        ]
        * output[
            "return_1d"
        ]
    )

    output[
        "overlay_return_net"
    ] = (
        output[
            "overlay_return_gross"
        ]
        - output[
            "transaction_cost"
        ]
    )

    output[
        "always_long_return"
    ] = output[
        "return_1d"
    ]

    output[
        "overlay_equity_net"
    ] = (
        1.0
        + output[
            "overlay_return_net"
        ].fillna(0.0)
    ).cumprod()

    return output


def summarize_performance(
    name: str,
    returns: pd.Series,
    exposure: pd.Series,
    turnover: pd.Series,
) -> dict:
    """Summarize portfolio performance."""

    annual_return = annualized_return(
        returns
    )

    max_dd = maximum_drawdown(
        returns
    )

    return {
        "strategy":
            name,

        "observations":
            int(
                returns.notna().sum()
            ),

        "total_return":
            float(
                (
                    1.0
                    + returns.fillna(0.0)
                ).prod()
                - 1.0
            ),

        "annualized_return":
            annual_return,

        "annualized_volatility":
            annualized_volatility(
                returns
            ),

        "max_drawdown":
            max_dd,

        "sharpe":
            sharpe_ratio(
                returns
            ),

        "sortino":
            sortino_ratio(
                returns
            ),

        "calmar":
            calmar_ratio(
                annual_return,
                max_dd,
            ),

        "mean_exposure":
            float(
                exposure.mean()
            ),

        "total_turnover":
            float(
                turnover.sum()
            ),

        "exposure_changes":
            int(
                (
                    turnover > 0
                ).sum()
            ),

        "worst_day":
            float(
                returns.min()
            ),
    }


def build_state_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize effective V2 exposure states."""

    total = len(data)

    summary = (
        data.groupby(
            "exposure_state",
            as_index=False,
        )
        .agg(
            days=(
                "market_date",
                "size",
            ),

            mean_tmf_return_1d_pct=(
                "tmf_return_1d_pct",
                "mean",
            ),

            mean_forward_return_1=(
                "forward_return_1",
                "mean",
            ),

            mean_forward_return_5=(
                "forward_return_5",
                "mean",
            ),

            max_run_length=(
                "state_run_length",
                "max",
            ),

            mean_warning_count=(
                "yellow_warning_count",
                "mean",
            ),
        )
    )

    summary[
        "share_of_days"
    ] = (
        summary["days"]
        / total
    )

    return summary


def build_reason_frequency(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Count exposure and entry-block reasons."""

    counter = Counter()

    for reasons in data[
        "exposure_reasons"
    ]:

        if not reasons:
            continue

        for reason in reasons.split(
            " | "
        ):
            counter[
                "EXPOSURE: " + reason
            ] += 1

    for reasons in data[
        "entry_block_reasons"
    ]:

        if not reasons:
            continue

        for reason in reasons.split(
            " | "
        ):
            counter[
                "ENTRY: " + reason
            ] += 1

    rows = [
        {
            "reason": reason,
            "count": count,
            "share_of_days":
                count / len(data),
        }
        for reason, count
        in counter.most_common()
    ]

    return pd.DataFrame(
        rows
    )


def main() -> None:
    """Run the V2 historical evaluation."""

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    eligible = (
        data[
            REQUIRED_NUMERIC
        ]
        .notna()
        .all(axis=1)
    )

    excluded = int(
        (~eligible).sum()
    )

    data = (
        data.loc[
            eligible
        ]
        .copy()
        .reset_index(drop=True)
    )

    data = (
        evaluate_chronologically(
            data
        )
    )

    data = (
        add_state_run_lengths(
            data
        )
    )

    data = (
        add_overlay_returns(
            data
        )
    )

    state_summary = (
        build_state_summary(
            data
        )
    )

    reason_frequency = (
        build_reason_frequency(
            data
        )
    )

    benchmark_exposure = pd.Series(
        1.0,
        index=data.index,
    )

    zero_turnover = pd.Series(
        0.0,
        index=data.index,
    )

    performance = pd.DataFrame(
        [
            summarize_performance(
                name="always_long_tmf",
                returns=data[
                    "always_long_return"
                ],
                exposure=benchmark_exposure,
                turnover=zero_turnover,
            ),

            summarize_performance(
                name="risk_governor_v2_net",
                returns=data[
                    "overlay_return_net"
                ],
                exposure=data[
                    "strategy_exposure"
                ],
                turnover=data[
                    "turnover"
                ],
            ),
        ]
    )

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    state_summary.to_csv(
        STATE_SUMMARY_FILE,
        index=False,
    )

    performance.to_csv(
        OVERLAY_SUMMARY_FILE,
        index=False,
    )

    reason_frequency.to_csv(
        REASON_FILE,
        index=False,
    )

    print("=" * 76)
    print(
        "V4 RISK GOVERNOR V2 — HISTORICAL EVALUATION"
    )
    print("=" * 76)

    print(
        "\nEvaluated days:",
        len(data),
    )

    print(
        "Excluded warm-up/incomplete days:",
        excluded,
    )

    print(
        "Range:",
        data["market_date"].min().date(),
        "to",
        data["market_date"].max().date(),
    )

    print(
        "\nEFFECTIVE EXPOSURE STATE SUMMARY"
    )

    print(
        state_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nRAW EXPOSURE STATE DISTRIBUTION"
    )

    print(
        data[
            "raw_exposure_state"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(
        "\nEFFECTIVE EXPOSURE STATE DISTRIBUTION"
    )

    print(
        data[
            "exposure_state"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(
        "\nENTRY PERMISSION"
    )

    print(
        data[
            "new_entries_allowed"
        ]
        .value_counts(
            normalize=True
        )
    )

    print(
        "\nPORTFOLIO PERFORMANCE"
    )

    print(
        performance.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nTURNOVER"
    )

    print(
        "Total:",
        data[
            "turnover"
        ].sum(),
    )

    print(
        "Exposure changes:",
        int(
            (
                data[
                    "turnover"
                ] > 0
            ).sum()
        ),
    )

    print(
        "\nRECOVERY HYSTERESIS"
    )

    print(
        "Days with active recovery streak:",
        int(
            (
                data[
                    "recovery_streak"
                ] > 0
            ).sum()
        ),
    )

    print(
        "\nTOP REASONS"
    )

    print(
        reason_frequency.head(
            20
        ).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {DAILY_OUTPUT_FILE}"
    )

    print(
        f"  {STATE_SUMMARY_FILE}"
    )

    print(
        f"  {OVERLAY_SUMMARY_FILE}"
    )

    print(
        f"  {REASON_FILE}"
    )


if __name__ == "__main__":
    main()