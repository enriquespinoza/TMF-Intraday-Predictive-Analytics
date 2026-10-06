from datetime import date
from pathlib import Path

import pandas as pd

from src.data.download_tmf import (
    normalize_yahoo_data,
)


def test_normalize_yahoo_data():

    index = pd.DatetimeIndex(
        [
            "2026-10-05",
        ],
        name="Date",
        tz="America/New_York",
    )

    source = pd.DataFrame(
        {
            "Open": [
                25.26,
            ],
            "High": [
                25.36,
            ],
            "Low": [
                24.79,
            ],
            "Close": [
                25.19,
            ],
            "Volume": [
                11826300,
            ],
        },
        index=index,
    )

    result = normalize_yahoo_data(
        source
    )

    assert list(
        result.columns
    ) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert len(
        result
    ) == 1

    assert (
        float(
            result.iloc[
                0
            ][
                "close"
            ]
        )
        == 25.19
    )