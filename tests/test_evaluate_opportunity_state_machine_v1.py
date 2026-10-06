import pandas as pd
import pytest

from src.analysis.evaluate_opportunity_state_machine_v1 import (
    add_forward_path_metrics,
    classify_research_period,
    parse_bool_value,
)


def test_parse_bool_values():

    assert parse_bool_value(
        True
    ) is True

    assert parse_bool_value(
        False
    ) is False

    assert parse_bool_value(
        "True"
    ) is True

    assert parse_bool_value(
        "False"
    ) is False

    assert parse_bool_value(
        1
    ) is True

    assert parse_bool_value(
        0
    ) is False


def test_invalid_bool_rejected():

    with pytest.raises(
        ValueError
    ):
        parse_bool_value(
            "maybe"
        )


def test_forward_path_metrics():

    data = pd.DataFrame(
        {
            "close": [
                100.0,
                95.0,
                105.0,
                90.0,
                110.0,
                120.0,
            ]
        }
    )

    result = add_forward_path_metrics(
        data,
        horizon=5,
    )

    assert result.loc[
        0,
        "forward_5d_mae",
    ] == pytest.approx(
        -0.10
    )

    assert result.loc[
        0,
        "forward_5d_mfe",
    ] == pytest.approx(
        0.20
    )


def test_research_period_classification():

    start = pd.Timestamp(
        "2022-11-14"
    )

    end = pd.Timestamp(
        "2026-08-31"
    )

    assert (
        classify_research_period(
            pd.Timestamp(
                "2022-06-01"
            ),
            start,
            end,
        )
        == "PRE_CALIBRATION"
    )

    assert (
        classify_research_period(
            pd.Timestamp(
                "2025-06-01"
            ),
            start,
            end,
        )
        == "CALIBRATION_WINDOW"
    )

    assert (
        classify_research_period(
            pd.Timestamp(
                "2026-09-01"
            ),
            start,
            end,
        )
        == "POST_CALIBRATION_DESCRIPTIVE"
    )