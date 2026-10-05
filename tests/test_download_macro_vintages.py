"""Tests for downloading and validating FRED macro vintages."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.download_macro_vintages import (
    parse_vintage_observations,
    validate_vintage_data,
)


def make_payload() -> dict:
    """Create an output_type=3 FRED vintage response."""

    return {
        "output_type": 3,
        "observations": [
            {
                "date": "2021-09-01",
                "PAYEMS_20211008": "147553",
                "PAYEMS_20211105": "147788",
                "PAYEMS_20211203": "147855",
            },
            {
                "date": "2021-10-01",
                "PAYEMS_20211105": "148319",
                "PAYEMS_20211203": "148401",
                "PAYEMS_20220107": "148503",
            },
        ],
    }


def test_parse_vintage_observations():
    data = parse_vintage_observations(
        payload=make_payload(),
        series_name="nonfarm_payrolls",
        series_id="PAYEMS",
    )

    assert list(data.columns) == [
        "series_name",
        "series_id",
        "reference_date",
        "vintage_date",
        "value_as_known",
    ]

    assert len(data) == 6

    assert (
        data["series_name"] == "nonfarm_payrolls"
    ).all()

    assert (
        data["series_id"] == "PAYEMS"
    ).all()


def test_vintage_date_parsed_from_column_name():
    data = parse_vintage_observations(
        payload=make_payload(),
        series_name="nonfarm_payrolls",
        series_id="PAYEMS",
    )

    first = data.iloc[0]

    assert (
        first["reference_date"]
        == pd.Timestamp("2021-09-01")
    )

    assert (
        first["vintage_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert first["value_as_known"] == 147553


def test_revisions_preserved():
    data = parse_vintage_observations(
        payload=make_payload(),
        series_name="nonfarm_payrolls",
        series_id="PAYEMS",
    )

    september = data.loc[
        data["reference_date"]
        == pd.Timestamp("2021-09-01")
    ]

    assert september["value_as_known"].tolist() == [
        147553,
        147788,
        147855,
    ]

    assert september["vintage_date"].tolist() == [
        pd.Timestamp("2021-10-08"),
        pd.Timestamp("2021-11-05"),
        pd.Timestamp("2021-12-03"),
    ]


def test_missing_fred_value_becomes_nan():
    payload = {
        "output_type": 3,
        "observations": [
            {
                "date": "2021-09-01",
                "PAYEMS_20211008": ".",
            }
        ],
    }

    data = parse_vintage_observations(
        payload=payload,
        series_name="nonfarm_payrolls",
        series_id="PAYEMS",
    )

    assert pd.isna(
        data.iloc[0]["value_as_known"]
    )


def test_vintage_before_reference_rejected():
    data = pd.DataFrame(
        {
            "series_name": [
                "nonfarm_payrolls"
            ],
            "series_id": [
                "PAYEMS"
            ],
            "reference_date": [
                pd.Timestamp("2021-09-01")
            ],
            "vintage_date": [
                pd.Timestamp("2021-08-01")
            ],
            "value_as_known": [
                147553
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="vintage dates before",
    ):
        validate_vintage_data(
            data=data,
            series_name="nonfarm_payrolls",
        )


def test_duplicate_vintage_event_rejected():
    data = parse_vintage_observations(
        payload=make_payload(),
        series_name="nonfarm_payrolls",
        series_id="PAYEMS",
    )

    duplicate = data.iloc[[0]].copy()

    data = pd.concat(
        [data, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_vintage_data(
            data=data,
            series_name="nonfarm_payrolls",
        )


def test_no_vintage_columns_rejected():
    payload = {
        "output_type": 3,
        "observations": [
            {
                "date": "2021-09-01",
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="No vintage columns found",
    ):
        parse_vintage_observations(
            payload=payload,
            series_name="nonfarm_payrolls",
            series_id="PAYEMS",
        )