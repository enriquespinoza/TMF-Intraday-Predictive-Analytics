import pandas as pd
import pytest

from src.analysis.purged_split import (
    add_target_end_date,
    purged_date_range,
    purged_train_validation_split,
)


def make_market_data():
    """Create simple business-day market data."""

    dates = pd.bdate_range(
        "2025-12-15",
        "2026-01-15",
    )

    return pd.DataFrame(
        {
            "market_date": dates,
            "close": range(len(dates)),
        }
    )


def test_target_end_date_uses_future_observations():

    df = make_market_data()

    result = add_target_end_date(
        df,
        horizon=5,
    )

    assert (
        result.loc[0, "target_end_date"]
        == result.loc[5, "market_date"]
    )


def test_target_end_date_is_not_calendar_days():

    df = make_market_data()

    result = add_target_end_date(
        df,
        horizon=5,
    )

    first_date = result.loc[
        0,
        "market_date",
    ]

    target_date = result.loc[
        0,
        "target_end_date",
    ]

    assert (
        target_date
        != first_date + pd.Timedelta(days=5)
    )


def test_purge_removes_targets_crossing_fold_boundary():

    df = make_market_data()

    result = purged_date_range(
        df=df,
        start_date="2025-12-15",
        end_date="2025-12-31",
        horizon=5,
    )

    assert (
        result["target_end_date"]
        <= pd.Timestamp("2025-12-31")
    ).all()


def test_development_targets_do_not_enter_2026():

    df = make_market_data()

    result = purged_date_range(
        df=df,
        start_date="2025-12-15",
        end_date="2025-12-31",
        horizon=5,
    )

    assert (
        result["target_end_date"].dt.year
        <= 2025
    ).all()


def test_validation_fold_stays_inside_validation_period():

    df = make_market_data()

    train, validation = (
        purged_train_validation_split(
            df=df,
            train_start="2025-12-15",
            train_end="2025-12-31",
            validation_start="2026-01-01",
            validation_end="2026-01-15",
            horizon=5,
        )
    )

    assert (
        train["target_end_date"]
        <= pd.Timestamp("2025-12-31")
    ).all()

    assert (
        validation["target_end_date"]
        <= pd.Timestamp("2026-01-15")
    ).all()


def test_invalid_horizon_raises_error():

    df = make_market_data()

    with pytest.raises(ValueError):

        add_target_end_date(
            df,
            horizon=0,
        )


def test_overlapping_train_validation_dates_raise_error():

    df = make_market_data()

    with pytest.raises(ValueError):

        purged_train_validation_split(
            df=df,
            train_start="2025-12-15",
            train_end="2026-01-05",
            validation_start="2026-01-01",
            validation_end="2026-01-15",
            horizon=5,
        )
def test_expanding_walk_forward_splits():

    from src.analysis.purged_split import (
        expanding_walk_forward_splits,
    )

    dates = pd.bdate_range(
        "2021-10-04",
        "2025-12-31",
    )

    df = pd.DataFrame(
        {
            "market_date": dates,
            "close": range(len(dates)),
        }
    )

    folds = expanding_walk_forward_splits(
        df,
        horizon=5,
    )

    assert len(folds) == 3

    assert folds[0]["train"][
        "target_end_date"
    ].max() <= pd.Timestamp(
        "2022-12-31"
    )

    assert folds[0]["validation"][
        "target_end_date"
    ].max() <= pd.Timestamp(
        "2023-12-31"
    )

    assert folds[1]["train"][
        "target_end_date"
    ].max() <= pd.Timestamp(
        "2023-12-31"
    )

    assert folds[1]["validation"][
        "target_end_date"
    ].max() <= pd.Timestamp(
        "2024-12-31"
    )

    assert folds[2]["train"][
        "target_end_date"
    ].max() <= pd.Timestamp(
        "2024-12-31"
    )

    assert folds[2]["validation"][
        "target_end_date"
    ].max() <= pd.Timestamp(
        "2025-12-31"
    )