"""Integration tests using the downloaded PAYEMS vintage dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.macro_vintage import (
    latest_reference_period_as_of,
    reconstruct_as_of,
)


DATA_PATH = Path(
    "data/raw/macro_vintages/nonfarm_payrolls.csv"
)


@pytest.fixture(scope="module")
def payems_vintages() -> pd.DataFrame:
    """Load the real downloaded PAYEMS vintage history."""

    if not DATA_PATH.exists():
        pytest.skip(
            "PAYEMS vintage dataset has not been downloaded."
        )

    return pd.read_csv(
        DATA_PATH,
        parse_dates=[
            "reference_date",
            "vintage_date",
        ],
    )


def test_real_payems_file_has_expected_schema(
    payems_vintages: pd.DataFrame,
):
    expected = {
        "series_name",
        "series_id",
        "reference_date",
        "vintage_date",
        "value_as_known",
    }

    assert set(payems_vintages.columns) == expected
    assert not payems_vintages.empty


def test_real_payems_september_2021_revision_history(
    payems_vintages: pd.DataFrame,
):
    september = payems_vintages.loc[
        payems_vintages["reference_date"]
        == pd.Timestamp("2021-09-01")
    ].sort_values("vintage_date")

    assert len(september) >= 3

    first_three = september.iloc[:3]

    assert first_three["vintage_date"].tolist() == [
        pd.Timestamp("2021-10-08"),
        pd.Timestamp("2021-11-05"),
        pd.Timestamp("2021-12-03"),
    ]

    assert first_three["value_as_known"].tolist() == [
        147553,
        147788,
        147855,
    ]


def test_real_payems_future_revision_does_not_leak(
    payems_vintages: pd.DataFrame,
):
    """
    On 2021-10-08, the model must see the original September
    payroll value, not revisions published in November or December.
    """

    result = reconstruct_as_of(
        data=payems_vintages,
        as_of_date="2021-10-08",
    )

    september = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-09-01")
    ]

    assert len(september) == 1

    row = september.iloc[0]

    assert row["value_as_known"] == 147553

    assert (
        row["vintage_date"]
        == pd.Timestamp("2021-10-08")
    )

    assert (
        result["vintage_date"]
        <= pd.Timestamp("2021-10-08")
    ).all()


def test_real_payems_november_revision_becomes_available(
    payems_vintages: pd.DataFrame,
):
    """
    By 2021-11-05, the revised September value and initial
    October value should both be part of the information set.
    """

    result = reconstruct_as_of(
        data=payems_vintages,
        as_of_date="2021-11-05",
    )

    september = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-09-01")
    ].iloc[0]

    october = result.loc[
        result["reference_date"]
        == pd.Timestamp("2021-10-01")
    ].iloc[0]

    assert september["value_as_known"] == 147788
    assert october["value_as_known"] == 148319

    assert (
        september["vintage_date"]
        == pd.Timestamp("2021-11-05")
    )

    assert (
        october["vintage_date"]
        == pd.Timestamp("2021-11-05")
    )


def test_real_payems_december_information_set(
    payems_vintages: pd.DataFrame,
):
    """
    On 2021-12-03 the model should reconstruct the values
    actually known for September, October, and November.
    """

    result = reconstruct_as_of(
        data=payems_vintages,
        as_of_date="2021-12-03",
    )

    expected = {
        pd.Timestamp("2021-09-01"): 147855,
        pd.Timestamp("2021-10-01"): 148401,
        pd.Timestamp("2021-11-01"): 148611,
    }

    actual = (
        result.set_index("reference_date")[
            "value_as_known"
        ]
        .to_dict()
    )

    for reference_date, expected_value in expected.items():
        assert actual[reference_date] == expected_value


def test_real_payems_latest_period_as_of_december_release(
    payems_vintages: pd.DataFrame,
):
    result = latest_reference_period_as_of(
        data=payems_vintages,
        as_of_date="2021-12-03",
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert (
        row["reference_date"]
        == pd.Timestamp("2021-11-01")
    )

    assert row["value_as_known"] == 148611

    assert (
        row["vintage_date"]
        == pd.Timestamp("2021-12-03")
    )


def test_real_payems_never_uses_future_vintage(
    payems_vintages: pd.DataFrame,
):
    """
    General invariant across several historical as-of dates:
    reconstructed data may never contain a vintage later than
    the requested information date.
    """

    dates = [
        "2021-10-08",
        "2021-11-05",
        "2021-12-03",
        "2022-02-04",
        "2023-02-03",
        "2024-02-02",
        "2025-02-07",
    ]

    for date in dates:
        result = reconstruct_as_of(
            data=payems_vintages,
            as_of_date=date,
        )

        assert (
            result["vintage_date"]
            <= pd.Timestamp(date)
        ).all()