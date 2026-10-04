"""Purged chronological splitting utilities.

These functions prevent forward-looking targets from crossing
chronological train/validation/test boundaries.
"""

from __future__ import annotations

import pandas as pd


def add_target_end_date(
    df: pd.DataFrame,
    horizon: int,
    date_column: str = "market_date",
) -> pd.DataFrame:
    """Add the actual market date used by a forward-return target.

    For a horizon h, row t uses the closing price from row t+h.
    Therefore the target belongs to the period ending at the
    market date h observations in the future.

    Parameters
    ----------
    df:
        Chronologically ordered market dataset.

    horizon:
        Forward-return horizon in trading observations.

    date_column:
        Name of the chronological market-date column.

    Returns
    -------
    pd.DataFrame
        Copy of the input containing ``target_end_date``.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero"
        )

    if date_column not in df.columns:
        raise KeyError(
            f"Missing date column: {date_column}"
        )

    data = df.copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data = data.sort_values(
        date_column
    ).reset_index(drop=True)

    data["target_end_date"] = (
        data[date_column].shift(-horizon)
    )

    return data


def purged_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    horizon: int,
    date_column: str = "market_date",
) -> pd.DataFrame:
    """Return rows whose feature date and target end date stay in a fold.

    A row is retained only when:

        start_date <= feature_date <= end_date

    and:

        target_end_date <= end_date

    This prevents a forward target from leaking across the end
    boundary of the fold.
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError(
            "start_date must be <= end_date"
        )

    data = add_target_end_date(
        df=df,
        horizon=horizon,
        date_column=date_column,
    )

    mask = (
        (data[date_column] >= start)
        & (data[date_column] <= end)
        & (data["target_end_date"] <= end)
        & data["target_end_date"].notna()
    )

    return data.loc[mask].copy()


def purged_train_validation_split(
    df: pd.DataFrame,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    horizon: int,
    date_column: str = "market_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological train and validation folds with purging."""

    train_end_ts = pd.Timestamp(train_end)
    validation_start_ts = pd.Timestamp(
        validation_start
    )

    if train_end_ts >= validation_start_ts:
        raise ValueError(
            "Training period must end before "
            "validation period begins."
        )

    train = purged_date_range(
        df=df,
        start_date=train_start,
        end_date=train_end,
        horizon=horizon,
        date_column=date_column,
    )

    validation = purged_date_range(
        df=df,
        start_date=validation_start,
        end_date=validation_end,
        horizon=horizon,
        date_column=date_column,
    )
    
    return train, validation
def expanding_walk_forward_splits(
    df: pd.DataFrame,
    horizon: int,
    date_column: str = "market_date",
) -> list[dict]:
    """Create expanding chronological train/validation folds.

    Fold 1:
        Train through 2022
        Validate on 2023

    Fold 2:
        Train through 2023
        Validate on 2024

    Fold 3:
        Train through 2024
        Validate on 2025

    Forward-return targets are purged at every fold boundary.
    """

    fold_definitions = [
        {
            "name": "fold_1",
            "train_start": "2021-10-04",
            "train_end": "2022-12-31",
            "validation_start": "2023-01-01",
            "validation_end": "2023-12-31",
        },
        {
            "name": "fold_2",
            "train_start": "2021-10-04",
            "train_end": "2023-12-31",
            "validation_start": "2024-01-01",
            "validation_end": "2024-12-31",
        },
        {
            "name": "fold_3",
            "train_start": "2021-10-04",
            "train_end": "2024-12-31",
            "validation_start": "2025-01-01",
            "validation_end": "2025-12-31",
        },
    ]

    folds = []

    for definition in fold_definitions:

        train, validation = (
            purged_train_validation_split(
                df=df,
                train_start=definition[
                    "train_start"
                ],
                train_end=definition[
                    "train_end"
                ],
                validation_start=definition[
                    "validation_start"
                ],
                validation_end=definition[
                    "validation_end"
                ],
                horizon=horizon,
                date_column=date_column,
            )
        )

        folds.append(
            {
                "name": definition["name"],
                "train": train,
                "validation": validation,
            }
        )

    return folds