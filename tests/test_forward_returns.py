import numpy as np
import pandas as pd

from src.analysis.forward_returns import (
    add_forward_returns,
    build_directional_score,
)


def make_sample_data(rows: int = 100) -> pd.DataFrame:
    close = np.linspace(25, 35, rows)

    return pd.DataFrame(
        {
            "close": close,
            "volume": np.full(rows, 1_000_000),
        }
    )


def test_directional_score_range():
    df = make_sample_data()

    result = build_directional_score(df)

    assert result["directional_score"].min() >= -5
    assert result["directional_score"].max() <= 5


def test_forward_return_columns_created():
    df = make_sample_data()

    result = add_forward_returns(df)

    expected = [
        "forward_return_1",
        "forward_return_3",
        "forward_return_5",
        "forward_return_10",
        "forward_return_20",
    ]

    for column in expected:
        assert column in result.columns


def test_forward_return_math():
    df = pd.DataFrame(
        {
            "close": [100.0, 105.0, 110.0],
            "volume": [1000, 1000, 1000],
        }
    )

    result = add_forward_returns(df)

    assert np.isclose(
        result.loc[0, "forward_return_1"],
        0.05,
    )

