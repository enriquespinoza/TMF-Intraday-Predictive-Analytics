"""
Historical market-state backtest for V4 Risk Governor V1.

This does NOT simulate trades yet.

It asks:

- How frequently is each risk state triggered?
- What conditions cause defensive states?
- Does the governor spend too much time blocking exposure?
- How persistent are defensive states?
- What does subsequent TMF behavior look like after each state?

Position-level drawdown rules are deliberately disabled in this
stage by using position_open=False.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from src.risk.risk_governor import (
    RiskInputs,
    evaluate_risk,
)


DATA_FILE = Path(
    "data/features/"
    "TMF_1d_risk_governor_features.csv"
)

OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_market_states.csv"
)

SUMMARY_FILE = Path(
    "reports/"
    "V4_risk_governor_state_summary.csv"
)

REASONS_FILE = Path(
    "reports/"
    "V4_risk_governor_reason_frequency.csv"
)


REQUIRED_NUMERIC = [
    "tmf_return_1d_pct",
    "change_30y_1d_bp",
    "change_30y_5d_bp",
    "real_yield_10y_change_5d_bp",
    "volatility_ratio_5d_20d",
]


def evaluate_row(
    row: pd.Series,
):
    """Evaluate one historical market day."""

    inputs = RiskInputs(
        tmf_return_1d_pct=float(
            row[
                "tmf_return_1d_pct"
            ]
        ),

        yield_30y_change_1d_bp=float(
            row[
                "change_30y_1d_bp"
            ]
        ),

        yield_30y_change_5d_bp=float(
            row[
                "change_30y_5d_bp"
            ]
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
            row[
                "below_ema21"
            ]
        ),

        critical_data_stale=bool(
            row[
                "critical_data_stale"
            ]
        ),

        # Market-state analysis only.
        position_open=False,
        drawdown_from_entry_pct=None,
    )

    return evaluate_risk(
        inputs
    )


def add_run_lengths(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate consecutive duration of each risk state."""

    output = data.copy()

    group = (
        output[
            "risk_state"
        ]
        != output[
            "risk_state"
        ].shift()
    ).cumsum()

    output[
        "state_run_length"
    ] = (
        output.groupby(
            group
        )
        .cumcount()
        + 1
    )

    return output


def build_state_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize governor behavior by state."""

    total = len(data)

    summary = (
        data.groupby(
            "risk_state",
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

            median_forward_return_5=(
                "forward_return_5",
                "median",
            ),

            max_run_length=(
                "state_run_length",
                "max",
            ),

            mean_yellow_warning_count=(
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
    """Count individual governor trigger reasons."""

    counter = Counter()

    for reasons in data[
        "risk_reasons"
    ]:

        if not reasons:
            continue

        for reason in reasons.split(
            " | "
        ):
            counter[
                reason
            ] += 1

    rows = [
        {
            "reason": reason,
            "count": count,
            "share_of_evaluated_days":
                count / len(data),
        }
        for reason, count
        in counter.most_common()
    ]

    return pd.DataFrame(
        rows
    )


def main() -> None:
    """Run historical market-state evaluation."""

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    # Initial rolling windows can be incomplete.
    # Do not invent values for them.
    eligible = (
        data[
            REQUIRED_NUMERIC
        ]
        .notna()
        .all(axis=1)
    )

    excluded = (
        ~eligible
    ).sum()

    data = (
        data.loc[
            eligible
        ]
        .copy()
        .reset_index(drop=True)
    )

    decisions = [
        evaluate_row(
            row
        )
        for _, row
        in data.iterrows()
    ]

    data[
        "risk_state"
    ] = [
        decision.state.value
        for decision
        in decisions
    ]

    data[
        "new_entries_allowed"
    ] = [
        decision.new_entries_allowed
        for decision
        in decisions
    ]

    data[
        "max_position_multiplier"
    ] = [
        decision.max_position_multiplier
        for decision
        in decisions
    ]

    data[
        "exit_required"
    ] = [
        decision.exit_required
        for decision
        in decisions
    ]

    data[
        "cooldown_required"
    ] = [
        decision.cooldown_required
        for decision
        in decisions
    ]

    data[
        "yellow_warning_count"
    ] = [
        decision.yellow_warning_count
        for decision
        in decisions
    ]

    data[
        "risk_reasons"
    ] = [
        " | ".join(
            decision.reasons
        )
        for decision
        in decisions
    ]

    data = add_run_lengths(
        data
    )

    summary = (
        build_state_summary(
            data
        )
    )

    reasons = (
        build_reason_frequency(
            data
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    reasons.to_csv(
        REASONS_FILE,
        index=False,
    )

    print("=" * 72)
    print("V4 RISK GOVERNOR V1 — MARKET-STATE BACKTEST")
    print("=" * 72)

    print(
        "\nEvaluated days:",
        len(data),
    )

    print(
        "Excluded warm-up/incomplete days:",
        int(excluded),
    )

    print(
        "Range:",
        data[
            "market_date"
        ].min().date(),
        "to",
        data[
            "market_date"
        ].max().date(),
    )

    print("\nSTATE SUMMARY")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
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
        .rename(
            "share"
        )
    )

    print(
        "\nTOP RISK REASONS"
    )

    print(
        reasons.head(
            15
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
        f"  {OUTPUT_FILE}"
    )
    print(
        f"  {SUMMARY_FILE}"
    )
    print(
        f"  {REASONS_FILE}"
    )


if __name__ == "__main__":
    main()