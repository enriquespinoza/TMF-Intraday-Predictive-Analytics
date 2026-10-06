"""
Compare Risk Governor V1, V2 raw-state, and V2 effective-state
portfolio overlays.

Purpose
-------
Separate the effects of:

1. Removing EMA21/data-staleness from exposure control.
2. Adding recovery hysteresis.

No thresholds are changed.

All state decisions at close t control exposure during t+1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.backtest_risk_governor_v2 import (
    summarize_performance,
)


V1_FILE = Path(
    "reports/"
    "V4_risk_governor_overlay_daily.csv"
)

V2_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_daily.csv"
)

OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_version_comparison.csv"
)

TRANSACTION_COST_BPS = 5.0

STATE_EXPOSURE = {
    "GREEN": 1.0,
    "YELLOW": 0.5,
    "RED": 0.0,
    "EMERGENCY_EXIT": 0.0,
}


def build_state_overlay(
    data: pd.DataFrame,
    state_column: str,
) -> pd.DataFrame:
    """
    Build a lagged portfolio overlay from one state column.
    """

    output = (
        data.copy()
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    output[
        "target_exposure"
    ] = output[
        state_column
    ].map(
        STATE_EXPOSURE
    )

    if output[
        "target_exposure"
    ].isna().any():
        bad = output.loc[
            output[
                "target_exposure"
            ].isna(),
            state_column,
        ].unique()

        raise ValueError(
            f"Unknown states in {state_column}: "
            f"{bad}"
        )

    # Close t decision controls day t+1.
    output[
        "strategy_exposure"
    ] = (
        output[
            "target_exposure"
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
        "strategy_return_net"
    ] = (
        output[
            "strategy_exposure"
        ]
        * output[
            "return_1d"
        ]
        - output[
            "turnover"
        ]
        * cost_rate
    )

    return output


def main() -> None:
    v1 = pd.read_csv(
        V1_FILE,
        parse_dates=["market_date"],
    )

    v2 = pd.read_csv(
        V2_FILE,
        parse_dates=["market_date"],
    )

    common_dates = sorted(
        set(v1["market_date"])
        & set(v2["market_date"])
    )

    v1 = (
        v1.loc[
            v1["market_date"].isin(
                common_dates
            )
        ]
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    v2 = (
        v2.loc[
            v2["market_date"].isin(
                common_dates
            )
        ]
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    if not v1[
        "market_date"
    ].equals(
        v2["market_date"]
    ):
        raise AssertionError(
            "V1 and V2 dates do not align."
        )

    if not v1[
        "return_1d"
    ].round(12).equals(
        v2[
            "return_1d"
        ].round(12)
    ):
        raise AssertionError(
            "V1 and V2 TMF returns do not align."
        )

    # Rebuild all overlays on identical dates.
    v1_overlay = (
        build_state_overlay(
            v1,
            "risk_state",
        )
    )

    v2_raw_overlay = (
        build_state_overlay(
            v2,
            "raw_exposure_state",
        )
    )

    v2_effective_overlay = (
        build_state_overlay(
            v2,
            "exposure_state",
        )
    )

    benchmark_exposure = pd.Series(
        1.0,
        index=v2.index,
    )

    zero_turnover = pd.Series(
        0.0,
        index=v2.index,
    )

    summaries = []

    summaries.append(
        summarize_performance(
            name="always_long_tmf",
            returns=v2["return_1d"],
            exposure=benchmark_exposure,
            turnover=zero_turnover,
        )
    )

    summaries.append(
        summarize_performance(
            name="risk_governor_v1",
            returns=v1_overlay[
                "strategy_return_net"
            ],
            exposure=v1_overlay[
                "strategy_exposure"
            ],
            turnover=v1_overlay[
                "turnover"
            ],
        )
    )

    summaries.append(
        summarize_performance(
            name="risk_governor_v2_raw",
            returns=v2_raw_overlay[
                "strategy_return_net"
            ],
            exposure=v2_raw_overlay[
                "strategy_exposure"
            ],
            turnover=v2_raw_overlay[
                "turnover"
            ],
        )
    )

    summaries.append(
        summarize_performance(
            name="risk_governor_v2_effective",
            returns=v2_effective_overlay[
                "strategy_return_net"
            ],
            exposure=v2_effective_overlay[
                "strategy_exposure"
            ],
            turnover=v2_effective_overlay[
                "turnover"
            ],
        )
    )

    comparison = pd.DataFrame(
        summaries
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    green_days = (
        v2[
            "exposure_state"
        ]
        == "GREEN"
    )

    green_entry_blocked = (
        green_days
        & v2[
            "entry_blocked"
        ].astype(bool)
    )

    ema_only_block = (
        green_entry_blocked
        & v2[
            "below_ema21"
        ].astype(bool)
        & ~v2[
            "critical_data_stale"
        ].astype(bool)
    )

    print("=" * 78)
    print(
        "V4 RISK GOVERNOR — ARCHITECTURAL ABLATION"
    )
    print("=" * 78)

    print(
        "\nCommon evaluated days:",
        len(v2),
    )

    print(
        "\nPORTFOLIO COMPARISON"
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nSTATE DISTRIBUTIONS"
    )

    state_table = pd.DataFrame(
        {
            "V1":
                v1[
                    "risk_state"
                ].value_counts(
                    normalize=True
                ),

            "V2_RAW":
                v2[
                    "raw_exposure_state"
                ].value_counts(
                    normalize=True
                ),

            "V2_EFFECTIVE":
                v2[
                    "exposure_state"
                ].value_counts(
                    normalize=True
                ),
        }
    ).fillna(0.0)

    print(
        state_table.to_string(
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nENTRY-GATE DIAGNOSTICS"
    )

    print(
        "Effective GREEN days:",
        int(
            green_days.sum()
        ),
    )

    print(
        "GREEN days still entry-blocked:",
        int(
            green_entry_blocked.sum()
        ),
    )

    print(
        "Share of GREEN days blocked:",
        (
            green_entry_blocked.sum()
            / green_days.sum()
            if green_days.sum()
            else 0.0
        ),
    )

    print(
        "GREEN days blocked with EMA21 "
        "as blocker:",
        int(
            ema_only_block.sum()
        ),
    )

    print(
        "\nInterpretation:"
    )

    print(
        "V1 -> V2_RAW isolates removal of "
        "EMA/stale-data from exposure."
    )

    print(
        "V2_RAW -> V2_EFFECTIVE isolates "
        "recovery hysteresis."
    )

    print(
        "\nSaved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()