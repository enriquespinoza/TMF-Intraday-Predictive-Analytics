"""Validate point-in-time alignment between macro observations and releases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.macro_release_calendar import (
    RELEASE_CALENDAR_DIR,
    load_release_calendar,
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


def load_raw_macro_series(
    series_name: str,
    raw_macro_dir: Path = RAW_MACRO_DIR,
) -> pd.DataFrame:
    """Load one raw FRED macro series."""

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

    data["reference_date"] = pd.to_datetime(
        data["reference_date"],
        errors="raise",
    ).dt.normalize()

    return data


def validate_series_release_coverage(
    series_name: str,
    raw_macro_dir: Path = RAW_MACRO_DIR,
    calendar_dir: Path = RELEASE_CALENDAR_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare non-null observations with known release events.

    Returns
    -------
    aligned:
        All raw observations with release metadata attached where available.

    problems:
        Non-null observations for which no release event exists.
    """

    observations = load_raw_macro_series(
        series_name=series_name,
        raw_macro_dir=raw_macro_dir,
    )

    calendar = load_release_calendar(
        series_name=series_name,
        calendar_dir=calendar_dir,
    )

    calendar_columns = [
        "reference_date",
        "release_date",
        "release_time",
        "release_timezone",
    ]

    aligned = observations.merge(
        calendar[calendar_columns],
        on="reference_date",
        how="left",
        validate="many_to_one",
    )

    problems = aligned.loc[
        aligned["value"].notna()
        & aligned["release_date"].isna()
    ].copy()

    return aligned, problems


def summarize_release_coverage(
    series_name: str,
    aligned: pd.DataFrame,
    problems: pd.DataFrame,
) -> dict:
    """Create a compact validation summary."""

    non_null = int(
        aligned["value"].notna().sum()
    )

    matched = int(
        (
            aligned["value"].notna()
            & aligned["release_date"].notna()
        ).sum()
    )

    unmatched = len(problems)

    return {
        "series_name": series_name,
        "non_null_observations": non_null,
        "matched_release_events": matched,
        "unmatched_observations": unmatched,
        "coverage_rate": (
            matched / non_null
            if non_null
            else float("nan")
        ),
    }


def validate_employment_situation_coverage(
    raw_macro_dir: Path = RAW_MACRO_DIR,
    calendar_dir: Path = RELEASE_CALENDAR_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate release coverage across Employment Situation series."""

    summaries = []
    problem_frames = []

    for series_name in EMPLOYMENT_SITUATION_SERIES:
        aligned, problems = (
            validate_series_release_coverage(
                series_name=series_name,
                raw_macro_dir=raw_macro_dir,
                calendar_dir=calendar_dir,
            )
        )

        summaries.append(
            summarize_release_coverage(
                series_name=series_name,
                aligned=aligned,
                problems=problems,
            )
        )

        if not problems.empty:
            problem_copy = problems[
                [
                    "reference_date",
                    "value",
                    "release_date",
                    "release_time",
                    "release_timezone",
                ]
            ].copy()

            problem_copy.insert(
                0,
                "series_name",
                series_name,
            )

            problem_frames.append(
                problem_copy
            )

    summary = pd.DataFrame(summaries)

    if problem_frames:
        all_problems = pd.concat(
            problem_frames,
            ignore_index=True,
        )
    else:
        all_problems = pd.DataFrame(
            columns=[
                "series_name",
                "reference_date",
                "value",
                "release_date",
                "release_time",
                "release_timezone",
            ]
        )

    expected_unmatched = int(
        summary["unmatched_observations"].sum()
    )

    if len(all_problems) != expected_unmatched:
        raise RuntimeError(
            "Release coverage validation is internally "
            "inconsistent: summary reports "
            f"{expected_unmatched} unmatched observations "
            f"but detail table contains "
            f"{len(all_problems)} rows."
        )

    return summary, all_problems


def main() -> None:
    """Run Employment Situation release-coverage validation."""

    summary, problems = (
        validate_employment_situation_coverage()
    )

    print("\nEmployment Situation release coverage")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
        )
    )

    print("\nUnmatched non-null observations")
    print("=" * 72)

    if problems.empty:
        print("None")
    else:
        columns = [
            column
            for column in [
                "series_name",
                "reference_date",
                "value",
            ]
            if column in problems.columns
        ]

        print(
            problems[
                columns
            ].to_string(
                index=False,
            )
        )


if __name__ == "__main__":
    main()