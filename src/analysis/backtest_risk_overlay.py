"""
V4 Risk Governor V1 portfolio-overlay backtest.

This evaluates the governor independently of the future entry engine.

Benchmark:
    Always long TMF.

Governor overlay:
    GREEN          -> 1.00 exposure
    YELLOW         -> 0.50 exposure
    RED            -> 0.00 exposure
    EMERGENCY_EXIT -> 0.00 exposure

Important
---------
The risk state calculated at the close of market day t controls
exposure for market day t+1.

This prevents same-day lookahead.

This test answers whether the governor reduces severe drawdowns and
downside exposure. It does not test entry timing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_market_states.csv"
)

DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_overlay_daily.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_overlay_summary.csv"
)


# Research assumption.
#
# Applied per 1.0 unit change in portfolio exposure.
# Example:
#     1.0 -> 0.5 = 0.5 turnover
#
# This is deliberately configurable rather than optimized.
TRANSACTION_COST_BPS = 5.0


STATE_EXPOSURE = {
    "GREEN": 1.00,
    "YELLOW": 0.50,
    "RED": 0.00,
    "EMERGENCY_EXIT": 0.00,
}


def maximum_drawdown(
    returns: pd.Series,
) -> float:
    """Calculate maximum drawdown from daily simple returns."""

    wealth = (
        1.0 + returns.fillna(0.0)
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

    total_growth = (
        1.0 + clean
    ).prod()

    years = (
        len(clean)
        / periods_per_year
    )

    if (
        total_growth <= 0
        or years <= 0
    ):
        return np.nan

    return float(
        total_growth ** (
            1.0 / years
        )
        - 1.0
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized realized volatility."""

    clean = returns.dropna()

    if clean.empty:
        return np.nan

    return float(
        clean.std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Risk-free rate intentionally omitted for the initial
    relative strategy comparison.
    """

    clean = returns.dropna()

    std = clean.std(
        ddof=1
    )

    if (
        clean.empty
        or std == 0
        or pd.isna(std)
    ):
        return np.nan

    return float(
        clean.mean()
        / std
        * np.sqrt(
            periods_per_year
        )
    )


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio."""

    clean = returns.dropna()

    downside = (
        clean.loc[
            clean < 0
        ]
    )

    if downside.empty:
        return np.nan

    downside_std = (
        downside.std(
            ddof=1
        )
    )

    if (
        downside_std == 0
        or pd.isna(
            downside_std
        )
    ):
        return np.nan

    return float(
        clean.mean()
        / downside_std
        * np.sqrt(
            periods_per_year
        )
    )


def calmar_ratio(
    annual_return: float,
    max_drawdown: float,
) -> float:
    """Calculate Calmar ratio."""

    if (
        pd.isna(
            annual_return
        )
        or max_drawdown >= 0
    ):
        return np.nan

    return float(
        annual_return
        / abs(
            max_drawdown
        )
    )


def build_overlay(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Construct lagged governor exposure and strategy returns."""

    output = (
        data.copy()
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    if "return_1d" not in output.columns:
        raise ValueError(
            "Input dataset requires return_1d."
        )

    unknown_states = (
        set(
            output[
                "risk_state"
            ].dropna()
        )
        - set(
            STATE_EXPOSURE
        )
    )

    if unknown_states:
        raise ValueError(
            "Unknown risk states: "
            f"{sorted(unknown_states)}"
        )

    output[
        "state_target_exposure"
    ] = output[
        "risk_state"
    ].map(
        STATE_EXPOSURE
    )

    # State observed at close t controls exposure during t+1.
    output[
        "strategy_exposure"
    ] = output[
        "state_target_exposure"
    ].shift(1)

    # First day has no prior governor decision.
    output[
        "strategy_exposure"
    ] = output[
        "strategy_exposure"
    ].fillna(
        0.0
    )

    output[
        "turnover"
    ] = output[
        "strategy_exposure"
    ].diff().abs()

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
        output[
            "turnover"
        ]
        * cost_rate
    )

    output[
        "always_long_return"
    ] = output[
        "return_1d"
    ]

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
        "always_long_equity"
    ] = (
        1.0
        + output[
            "always_long_return"
        ].fillna(
            0.0
        )
    ).cumprod()

    output[
        "overlay_equity_gross"
    ] = (
        1.0
        + output[
            "overlay_return_gross"
        ].fillna(
            0.0
        )
    ).cumprod()

    output[
        "overlay_equity_net"
    ] = (
        1.0
        + output[
            "overlay_return_net"
        ].fillna(
            0.0
        )
    ).cumprod()

    return output


def summarize_strategy(
    name: str,
    returns: pd.Series,
    exposure: pd.Series,
    turnover: pd.Series,
) -> dict:
    """Calculate portfolio-level performance metrics."""

    annual_return = (
        annualized_return(
            returns
        )
    )

    max_dd = (
        maximum_drawdown(
            returns
        )
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
                    + returns.fillna(
                        0.0
                    )
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

        "worst_day":
            float(
                returns.min()
            ),

        "positive_day_rate":
            float(
                (
                    returns > 0
                ).mean()
            ),
    }


def main() -> None:
    """Run the independent V1 governor overlay backtest."""

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    overlay = (
        build_overlay(
            data
        )
    )

    benchmark_exposure = (
        pd.Series(
            1.0,
            index=overlay.index,
        )
    )

    benchmark_turnover = (
        pd.Series(
            0.0,
            index=overlay.index,
        )
    )

    summary = pd.DataFrame(
        [
            summarize_strategy(
                name="always_long_tmf",
                returns=(
                    overlay[
                        "always_long_return"
                    ]
                ),
                exposure=(
                    benchmark_exposure
                ),
                turnover=(
                    benchmark_turnover
                ),
            ),

            summarize_strategy(
                name="risk_overlay_gross",
                returns=(
                    overlay[
                        "overlay_return_gross"
                    ]
                ),
                exposure=(
                    overlay[
                        "strategy_exposure"
                    ]
                ),
                turnover=(
                    overlay[
                        "turnover"
                    ]
                ),
            ),

            summarize_strategy(
                name="risk_overlay_net",
                returns=(
                    overlay[
                        "overlay_return_net"
                    ]
                ),
                exposure=(
                    overlay[
                        "strategy_exposure"
                    ]
                ),
                turnover=(
                    overlay[
                        "turnover"
                    ]
                ),
            ),
        ]
    )

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    print("=" * 76)
    print(
        "V4 RISK GOVERNOR V1 — PORTFOLIO OVERLAY BACKTEST"
    )
    print("=" * 76)

    print(
        "\nState at close t controls exposure during t+1."
    )

    print(
        "Transaction cost assumption:",
        f"{TRANSACTION_COST_BPS:.1f} bps "
        "per 1.0 turnover",
    )

    print(
        "\nPERFORMANCE SUMMARY"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nEXPOSURE DISTRIBUTION"
    )

    print(
        overlay[
            "strategy_exposure"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(
        "\nTURNOVER"
    )

    print(
        "Total:",
        overlay[
            "turnover"
        ].sum(),
    )

    print(
        "Exposure changes:",
        int(
            (
                overlay[
                    "turnover"
                ]
                > 0
            ).sum()
        ),
    )

    print("\nSaved:")
    print(
        f"  {DAILY_OUTPUT_FILE}"
    )
    print(
        f"  {SUMMARY_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()