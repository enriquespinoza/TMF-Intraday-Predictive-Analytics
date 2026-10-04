import pandas as pd

from src.analysis.train_test_split import (
    chronological_split,
)


def test_chronological_split():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-30",
                "2025-12-31",
                "2026-01-02",
                "2026-01-05",
            ],
            "close": [
                30,
                31,
                32,
                33,
            ],
        }
    )

    development, test = chronological_split(df)

    assert len(development) == 2
    assert len(test) == 2


def test_split_has_no_overlap():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-30",
                "2025-12-31",
                "2026-01-02",
                "2026-01-05",
            ],
            "close": [
                30,
                31,
                32,
                33,
            ],
        }
    )

    development, test = chronological_split(df)

    assert (
        development["timestamp"].max()
        < test["timestamp"].min()
    )
    