"""
Validate the integrated risk bridge.

Replays Risk Governor V3 with:

position_open = False
drawdown_from_entry_pct = None

and verifies exact agreement with the previously generated
V4_risk_governor_v3_daily.csv outputs.

This must pass before position-aware P&L simulation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.execution.integrated_bridge import (
    build_risk_inputs_v1,
    parse_bool_value,
)

from src.execution.position_lifecycle import (
    PositionMemoryV1,
)

from src.risk.risk_governor_v3 import (
    GovernorMemory,
    evaluate_risk_v3,
)


INPUT_FILE = Path(
    "reports/"
    "V4_risk_governor_v3_daily.csv"
)

OUTPUT_FILE = Path(
    "reports/"
    "V4_integrated_risk_bridge_validation.csv"
)


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

    position = PositionMemoryV1()

    memory = GovernorMemory()

    records = []

    for _, row in data.iterrows():

        risk_inputs, drawdown = (
            build_risk_inputs_v1(
                row=row,
                position=position,
            )
        )

        if drawdown is not None:
            raise AssertionError(
                "Flat replay unexpectedly "
                "produced position drawdown."
            )

        (
            decision,
            memory,
        ) = evaluate_risk_v3(
            inputs=risk_inputs,
            memory=memory,
        )

        expected_raw = str(
            row[
                "v3_raw_exposure_state"
            ]
        )

        expected_effective = str(
            row[
                "v3_exposure_state"
            ]
        )

        expected_entry_blocked = (
            parse_bool_value(
                row[
                    "v3_entry_blocked"
                ]
            )
        )

        expected_new_entries = (
            parse_bool_value(
                row[
                    "v3_new_entries_allowed"
                ]
            )
        )

        expected_exit = (
            parse_bool_value(
                row[
                    "v3_exit_required"
                ]
            )
        )

        expected_cooldown = (
            parse_bool_value(
                row[
                    "v3_cooldown_required"
                ]
            )
        )

        actual_raw = (
            decision.raw_exposure_state.value
        )

        actual_effective = (
            decision.exposure_state.value
        )

        checks = {
            "raw_state_match":
                actual_raw
                == expected_raw,

            "effective_state_match":
                actual_effective
                == expected_effective,

            "entry_blocked_match":
                decision.entry_blocked
                == expected_entry_blocked,

            "new_entries_match":
                decision.new_entries_allowed
                == expected_new_entries,

            "exit_match":
                decision.exit_required
                == expected_exit,

            "cooldown_match":
                decision.cooldown_required
                == expected_cooldown,
        }

        all_match = all(
            checks.values()
        )

        records.append(
            {
                "market_date":
                    row[
                        "market_date"
                    ],

                "expected_raw_state":
                    expected_raw,

                "actual_raw_state":
                    actual_raw,

                "expected_exposure_state":
                    expected_effective,

                "actual_exposure_state":
                    actual_effective,

                **checks,

                "all_match":
                    all_match,
            }
        )

    result = pd.DataFrame(
        records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    check_columns = [
        "raw_state_match",
        "effective_state_match",
        "entry_blocked_match",
        "new_entries_match",
        "exit_match",
        "cooldown_match",
    ]

    print("=" * 78)
    print(
        "V4 INTEGRATED RISK BRIDGE — "
        "FLAT-POSITION REPLAY VALIDATION"
    )
    print("=" * 78)

    print(
        "\nRows evaluated:",
        len(
            result
        ),
    )

    print(
        "\nCHECK RESULTS"
    )

    for column in check_columns:

        matches = int(
            result[
                column
            ].sum()
        )

        print(
            f"{column:28} "
            f"{matches}/{len(result)}"
        )

    failures = result.loc[
        ~result[
            "all_match"
        ]
    ]

    print(
        "\nRows with any mismatch:",
        len(
            failures
        ),
    )

    if not failures.empty:

        print(
            "\nFIRST MISMATCHES"
        )

        print(
            failures.head(
                20
            ).to_string(
                index=False
            )
        )

        raise AssertionError(
            "Integrated bridge does not "
            "reproduce frozen Risk Governor V3."
        )

    print(
        "\nPASS — integrated bridge exactly "
        "reproduces frozen V3 flat-position history."
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()