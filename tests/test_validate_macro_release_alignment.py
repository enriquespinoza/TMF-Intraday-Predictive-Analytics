"""Tests for macro release-alignment validation."""

from __future__ import annotations

import pandas as pd

from src.analysis.validate_macro_release_alignment import (
    summarize_release_coverage,
    validate_employment_situation_coverage,
    validate_series_release_coverage,
)


def write_raw_series(
    path,
    values,
):
    """Write a synthetic raw macro series."""

    pd.DataFrame(
        {
            "reference_date": [
                "2025-09-01",
                "2025-10-01",
                "2025-11-01",
            ],
            "value": values,
        }
    ).to_csv(
        path,
        index=False,
    )


def write_calendar(
    path,
):
    """Write a synthetic release calendar."""

    pd.DataFrame(
        {
            "series_name": [
                "unemployment_rate",
                "unemployment_rate",
            ],
            "reference_date": [
                "2025-09-01",
                "2025-11-01",
            ],
            "release_date": [
                "2025-11-20",
                "2025-12-16",
            ],
            "release_time": [
                "08:30",
                "08:30",
            ],
            "release_timezone": [
                "America/New_York",
                "America/New_York",
            ],
        }
    ).to_csv(
        path,
        index=False,
    )


def test_null_observation_without_release_is_allowed(
    tmp_path,
):
    """Missing observations do not require a release event."""

    raw_dir = tmp_path / "raw"
    calendar_dir = tmp_path / "calendar"

    raw_dir.mkdir()
    calendar_dir.mkdir()

    write_raw_series(
        raw_dir / "unemployment_rate.csv",
        [4.4, None, 4.6],
    )

    write_calendar(
        calendar_dir / "unemployment_rate.csv"
    )

    aligned, problems = (
        validate_series_release_coverage(
            series_name="unemployment_rate",
            raw_macro_dir=raw_dir,
            calendar_dir=calendar_dir,
        )
    )

    assert len(aligned) == 3
    assert problems.empty


def test_non_null_observation_without_release_is_flagged(
    tmp_path,
):
    """Observed values must have legitimate release events."""

    raw_dir = tmp_path / "raw"
    calendar_dir = tmp_path / "calendar"

    raw_dir.mkdir()
    calendar_dir.mkdir()

    write_raw_series(
        raw_dir / "unemployment_rate.csv",
        [4.4, 4.5, 4.6],
    )

    write_calendar(
        calendar_dir / "unemployment_rate.csv"
    )

    _, problems = (
        validate_series_release_coverage(
            series_name="unemployment_rate",
            raw_macro_dir=raw_dir,
            calendar_dir=calendar_dir,
        )
    )

    assert len(problems) == 1

    assert (
        problems.iloc[0]["reference_date"]
        == pd.Timestamp("2025-10-01")
    )


def test_release_coverage_summary():
    """Coverage statistics should count only non-null observations."""

    aligned = pd.DataFrame(
        {
            "value": [
                4.4,
                None,
                4.6,
            ],
            "release_date": [
                pd.Timestamp("2025-11-20"),
                pd.NaT,
                pd.Timestamp("2025-12-16"),
            ],
        }
    )

    problems = aligned.iloc[0:0].copy()

    result = summarize_release_coverage(
        series_name="unemployment_rate",
        aligned=aligned,
        problems=problems,
    )

    assert (
        result["non_null_observations"]
        == 2
    )

    assert (
        result["matched_release_events"]
        == 2
    )

    assert (
        result["unmatched_observations"]
        == 0
    )

    assert (
        result["coverage_rate"]
        == 1.0
    )
def test_existing_series_name_column_is_supported(
    tmp_path,
    monkeypatch,
):
    """Raw files may already contain a series_name column."""

    raw_dir = tmp_path / "raw"
    calendar_dir = tmp_path / "calendar"

    raw_dir.mkdir()
    calendar_dir.mkdir()

    import src.analysis.validate_macro_release_alignment as module

    monkeypatch.setattr(
        module,
        "EMPLOYMENT_SITUATION_SERIES",
        ["unemployment_rate"],
    )

    pd.DataFrame(
        {
            "reference_date": [
                "2025-09-01",
                "2025-10-01",
                "2025-11-01",
            ],
            "value": [
                4.4,
                4.5,
                4.6,
            ],
            "series_name": [
                "unemployment_rate",
                "unemployment_rate",
                "unemployment_rate",
            ],
        }
    ).to_csv(
        raw_dir / "unemployment_rate.csv",
        index=False,
    )

    write_calendar(
        calendar_dir / "unemployment_rate.csv"
    )

    summary, problems = (
        validate_employment_situation_coverage(
            raw_macro_dir=raw_dir,
            calendar_dir=calendar_dir,
        )
    )

    assert len(summary) == 1
    assert len(problems) == 1

    assert (
        problems.iloc[0]["series_name"]
        == "unemployment_rate"
    )

    assert (
        problems.iloc[0]["reference_date"]
        == pd.Timestamp("2025-10-01")
    )