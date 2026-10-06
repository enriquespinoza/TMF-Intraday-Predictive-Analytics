import pandas as pd
import pytest

from src.analysis.evaluate_market_confirmation_v1 import (
    add_forward_path_metrics,
)


def test_forward_path_metrics():
    data = pd.DataFrame(
        {
            "close": [
                100.0,
                97.0,
                104.0,
                92.0,
                108.0,
                115.0,
            ]
        }
    )

    result = add_forward_path_metrics(
        data,
        horizon=5,
    )

    # From the first observation:
    #
    # 97  = -3%
    # 104 = +4%
    # 92  = -8%
    # 108 = +8%
    # 115 = +15%

    assert result.loc[
        0,
        "forward_5d_mae",
    ] == pytest.approx(
        -0.08
    )

    assert result.loc[
        0,
        "forward_5d_mfe",
    ] == pytest.approx(
        0.15
    )


def test_forward_metrics_do_not_use_current_close_as_path_point():
    data = pd.DataFrame(
        {
            "close": [
                100.0,
                110.0,
                120.0,
            ]
        }
    )

    result = add_forward_path_metrics(
        data,
        horizon=2,
    )

    # Future path from 100 is +10%, +20%.
    # MAE should therefore be +10%, not zero.
    assert result.loc[
        0,
        "forward_5d_mae",
    ] == pytest.approx(
        0.10
    )

    assert result.loc[
        0,
        "forward_5d_mfe",
    ] == pytest.approx(
        0.20
    )