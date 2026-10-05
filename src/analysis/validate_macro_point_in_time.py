"""Validate point-in-time availability of release-sensitive macro data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.macro_release_calendar import (
    RELEASE_CALENDAR_DIR,
    load_release_calendar,
    prepare_release_series,
)


RAW_MACRO_DIR = Path("data/raw/macro")


EMPLOYMENT_SITUATION_SERIES = [
    "nonfarm_payrolls",
    "unemployment_rate",
    "u6_unemployment_rate",
    "labor_force_participation",
    "part_time_economic_reasons",
    "average_weekly_hours",
    "average_hourly_earnings",
]


def load_raw_observations(
    series_name: str,
    raw_macro_dir: Path = RAW_MACRO_DIR,
) -> pd.DataFrame:
    """Load one raw macro observation file."""

    path = raw_macro_dir / f"{series_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Raw macro series not found: {path}"
        )

    data = pd.read_csv(path)

    required = {
        "reference_date",
        "value",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Raw macro series missing required "
            f"columns: {sorted(missing)}"
        )

    return data


def prepare_point_in_time_series(
    series_name: str,
    raw_macro_dir: Path = RAW_MACRO_DIR,
    calendar_dir: Path = RELEASE_CALENDAR_DIR,
) -> pd.DataFrame:
    """Prepare one release-sensitive series point in time."""

    observations = load_raw_observations(
        series_name=series_name,
        raw_macro_dir=raw_macro_dir,
    )

    calendar = load_release_calendar(
        series_name=series_name,
        calendar_dir=calendar_dir,
    )

    prepared = prepare_release_series(
        observations=observations,
        calendar=calendar,
        series_name=series_name,
    )

    return prepared


def validate_point_in_time_series(
    series_name: str,
    prepared: pd.DataFrame,
) -> dict:
    """Validate temporal ordering of one prepared macro series."""

    observed = prepared.loc[
        prepared["value"].notna()
    ].copy()

    if observed.empty:
        return {
            "series_name": series_name,
            "observations": 0,
            "valid_availability": 0,
            "invalid_availability": 0,
        }

    missing_availability = (
        observed["market_date_available"].isna()
    )

    if missing_availability.any():
        dates = (
            observed.loc[
                missing_availability,
                "reference_date",
            ]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        raise ValueError(
            f"{series_name} has observations "
            "without market availability: "
            f"{dates[:10]}"
        )

    invalid_order = (
        observed["market_date_available"]
        < observed["reference_date"]
    )

    if invalid_order.any():
        dates = (
            observed.loc[
                invalid_order,
                "reference_date",
            ]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        raise ValueError(
            f"{series_name} contains lookahead "
            "availability dates: "
            f"{dates[:10]}"
        )

    return {
        "series_name": series_name,
        "observations": len(observed),
        "valid_availability": len(observed),
        "invalid_availability": 0,
    }


def validate_employment_situation_point_in_time(
    raw_macro_dir: Path = RAW_MACRO_DIR,
    calendar_dir: Path = RELEASE_CALENDAR_DIR,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Validate all Employment Situation series."""

    summaries = []
    prepared_series = {}

    for series_name in EMPLOYMENT_SITUATION_SERIES:
        prepared = prepare_point_in_time_series(
            series_name=series_name,
            raw_macro_dir=raw_macro_dir,
            calendar_dir=calendar_dir,
        )

        summaries.append(
            validate_point_in_time_series(
                series_name=series_name,
                prepared=prepared,
            )
        )

        prepared_series[
            series_name
        ] = prepared

    return (
        pd.DataFrame(summaries),
        prepared_series,
    )


def print_shutdown_window(
    prepared_series: dict[str, pd.DataFrame],
) -> None:
    """Display point-in-time behavior around the 2025 shutdown."""

    print(
        "\n2025 shutdown point-in-time audit"
    )
    print("=" * 88)

    for series_name, data in prepared_series.items():
        window = data.loc[
            (
                data["reference_date"]
                >= pd.Timestamp("2025-09-01")
            )
            & (
                data["reference_date"]
                <= pd.Timestamp("2025-11-01")
            ),
            [
                "reference_date",
                "value",
                "release_date",
                "release_time",
                "release_timezone",
                "market_date_available",
            ],
        ]

        print(
            f"\n{series_name}"
        )

        print(
            window.to_string(
                index=False,
            )
        )


def main() -> None:
    """Run point-in-time validation."""

    summary, prepared_series = (
        validate_employment_situation_point_in_time()
    )

    print(
        "\nEmployment Situation point-in-time validation"
    )
    print("=" * 88)

    print(
        summary.to_string(
            index=False,
        )
    )

    print_shutdown_window(
        prepared_series
    )


if __name__ == "__main__":
    main()
    