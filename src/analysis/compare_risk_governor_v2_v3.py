"""
Compare Risk Governor V2 and V3.

V3 is allowed to change ONLY entry permission.

The following must remain identical to V2:

- raw exposure state
- effective exposure state
- maximum position multiplier
- exit requirement
- cooldown requirement

Any difference in those fields is an implementation error.

V3 changes only the EMA21 entry gate:

V2:
    any close below EMA21 blocks entry

V3:
    entry blocked only when below-EMA21 streak >= 4 sessions
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.risk.risk_governor_v2 import (
    GovernorMemory,
)

from src.risk.risk_governor_v3 import (
    RiskInputsV3,
    evaluate_risk_v3,
)


INPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_daily.csv"
)

DAILY_OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v3_daily.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v2_v3_entry_comparison.csv"
)

RESTORED_OUTPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v3_restored_entry_days.csv"
)


def add_ema_streak(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate consecutive sessions below EMA21."""

    output = (
        data.copy()
        .sort_values(
            "market_date"
        )
        .reset_index(drop=True)
    )

    below = (
        output[
            "below_ema21"
        ]
        .astype(bool)
    )

    groups = (
        below
        != below.shift()
    ).cumsum()

    streak = (
        below.groupby(groups)
        .cumcount()
        + 1
    )

    output[
        "below_ema21_streak"
    ] = streak.where(
        below,
        0,
    ).astype(int)

    return output


def make_v3_inputs(
    row: pd.Series,
) -> RiskInputsV3:
    """Convert one historical row to V3 inputs."""

    return RiskInputsV3(
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

        below_ema21_streak=int(
            row[
                "below_ema21_streak"
            ]
        ),

        critical_data_stale=bool(
            row[
                "critical_data_stale"
            ]
        ),

        position_open=False,

        drawdown_from_entry_pct=None,
    )


def evaluate_v3_chronologically(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Apply V3 sequentially using GovernorMemory."""

    output = add_ema_streak(
        data
    )

    memory = GovernorMemory()

    records = []

    for _, row in output.iterrows():

        decision, memory = (
            evaluate_risk_v3(
                inputs=make_v3_inputs(
                    row
                ),
                memory=memory,
            )
        )

        records.append(
            {
                "v3_raw_exposure_state":
                    decision.raw_exposure_state.value,

                "v3_exposure_state":
                    decision.exposure_state.value,

                "v3_entry_blocked":
                    decision.entry_blocked,

                "v3_new_entries_allowed":
                    decision.new_entries_allowed,

                "v3_max_position_multiplier":
                    decision.max_position_multiplier,

                "v3_exit_required":
                    decision.exit_required,

                "v3_cooldown_required":
                    decision.cooldown_required,

                "v3_recovery_streak":
                    decision.recovery_streak,

                "v3_entry_block_reasons":
                    " | ".join(
                        decision.entry_block_reasons
                    ),
            }
        )

    return pd.concat(
        [
            output,
            pd.DataFrame(
                records
            ),
        ],
        axis=1,
    )


def assert_exposure_identical(
    data: pd.DataFrame,
) -> None:
    """
    V3 must not alter V2 exposure behavior.
    """

    comparisons = [
        (
            "raw_exposure_state",
            "v3_raw_exposure_state",
        ),
        (
            "exposure_state",
            "v3_exposure_state",
        ),
        (
            "max_position_multiplier",
            "v3_max_position_multiplier",
        ),
        (
            "exit_required",
            "v3_exit_required",
        ),
        (
            "cooldown_required",
            "v3_cooldown_required",
        ),
        (
            "recovery_streak",
            "v3_recovery_streak",
        ),
    ]

    for v2_col, v3_col in comparisons:

        same = (
            data[
                v2_col
            ]
            .reset_index(drop=True)
            .equals(
                data[
                    v3_col
                ]
                .reset_index(drop=True)
            )
        )

        if not same:
            differences = (
                data[
                    data[
                        v2_col
                    ]
                    != data[
                        v3_col
                    ]
                ]
            )

            raise AssertionError(
                f"V2/V3 exposure mismatch in "
                f"{v2_col}. "
                f"Differences: "
                f"{len(differences)}"
            )


def build_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize V2/V3 entry permission."""

    v2_allowed = (
        data[
            "new_entries_allowed"
        ].astype(bool)
    )

    v3_allowed = (
        data[
            "v3_new_entries_allowed"
        ].astype(bool)
    )

    restored = (
        ~v2_allowed
        & v3_allowed
    )

    lost = (
        v2_allowed
        & ~v3_allowed
    )

    total = len(data)

    rows = [
        {
            "metric":
                "evaluated_days",

            "value":
                total,
        },

        {
            "metric":
                "v2_entry_allowed_days",

            "value":
                int(
                    v2_allowed.sum()
                ),
        },

        {
            "metric":
                "v2_entry_allowed_share",

            "value":
                float(
                    v2_allowed.mean()
                ),
        },

        {
            "metric":
                "v3_entry_allowed_days",

            "value":
                int(
                    v3_allowed.sum()
                ),
        },

        {
            "metric":
                "v3_entry_allowed_share",

            "value":
                float(
                    v3_allowed.mean()
                ),
        },

        {
            "metric":
                "entry_days_restored_by_v3",

            "value":
                int(
                    restored.sum()
                ),
        },

        {
            "metric":
                "entry_days_lost_by_v3",

            "value":
                int(
                    lost.sum()
                ),
        },

        {
            "metric":
                "restored_share_of_all_days",

            "value":
                float(
                    restored.mean()
                ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def main() -> None:
    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "market_date",
        ],
    )

    data = (
        evaluate_v3_chronologically(
            data
        )
    )

    assert_exposure_identical(
        data
    )

    v2_allowed = (
        data[
            "new_entries_allowed"
        ].astype(bool)
    )

    v3_allowed = (
        data[
            "v3_new_entries_allowed"
        ].astype(bool)
    )

    restored_mask = (
        ~v2_allowed
        & v3_allowed
    )

    restored = (
        data.loc[
            restored_mask
        ]
        .copy()
    )

    summary = build_summary(
        data
    )

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        DAILY_OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    restored.to_csv(
        RESTORED_OUTPUT_FILE,
        index=False,
    )

    print("=" * 78)
    print(
        "V4 RISK GOVERNOR V2 VS V3 — ENTRY GATE COMPARISON"
    )
    print("=" * 78)

    print(
        "\nExposure behavior:"
    )

    print(
        "PASS — V2 and V3 exposure states, "
        "multipliers, exits, cooldowns, and "
        "hysteresis are identical."
    )

    print(
        "\nENTRY PERMISSION"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print(
        "\nV3 ALLOWED-DAY EMA STREAK DISTRIBUTION"
    )

    allowed_v3 = (
        data.loc[
            data[
                "v3_new_entries_allowed"
            ].astype(bool)
        ]
    )

    print(
        allowed_v3[
            "below_ema21_streak"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nRESTORED V2-BLOCKED DAYS"
    )

    if restored.empty:
        print(
            "None."
        )

    else:
        restored_summary = (
            restored.groupby(
                "below_ema21_streak",
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

                positive_5d_rate=(
                    "forward_return_5",
                    lambda x:
                        (
                            x > 0
                        ).mean(),
                ),
            )
        )

        print(
            restored_summary.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )

    print(
        "\nImportant:"
    )

    print(
        "Forward-return statistics for restored "
        "days are descriptive/in-sample only."
    )

    print(
        "They must not be treated as independent "
        "validation because the EMA threshold was "
        "derived from this historical sample."
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {DAILY_OUTPUT_FILE}"
    )

    print(
        f"  {SUMMARY_OUTPUT_FILE}"
    )

    print(
        f"  {RESTORED_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()