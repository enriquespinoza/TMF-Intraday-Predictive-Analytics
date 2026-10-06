import pandas as pd
import pytest

from src.execution.integrated_bridge import (
    build_risk_inputs_v1,
    parse_bool_value,
)

from src.execution.position_lifecycle import (
    PositionMemoryV1,
)


def base_row():
    return pd.Series(
        {
            "close":
                96.0,

            "tmf_return_1d_pct":
                -2.0,

            "change_30y_1d_bp":
                5.0,

            "change_30y_5d_bp":
                10.0,

            "real_yield_10y_change_5d_bp":
                4.0,

            "volatility_ratio_5d_20d":
                1.1,

            "below_ema21":
                True,

            "below_ema21_streak":
                2,

            "critical_data_stale":
                False,
        }
    )


def test_flat_position_has_no_drawdown():
    inputs, drawdown = (
        build_risk_inputs_v1(
            row=base_row(),
            position=PositionMemoryV1(),
        )
    )

    assert (
        inputs.position_open
        is False
    )

    assert (
        inputs.drawdown_from_entry_pct
        is None
    )

    assert drawdown is None


def test_open_position_drawdown_uses_original_anchor():
    position = PositionMemoryV1(
        current_exposure=0.5,
        entry_anchor_price=100.0,
    )

    inputs, drawdown = (
        build_risk_inputs_v1(
            row=base_row(),
            position=position,
        )
    )

    assert (
        inputs.position_open
        is True
    )

    assert drawdown == pytest.approx(
        -4.0
    )

    assert (
        inputs.drawdown_from_entry_pct
        == pytest.approx(
            -4.0
        )
    )


def test_csv_false_is_not_truthy():
    assert (
        parse_bool_value(
            "False"
        )
        is False
    )

    assert (
        parse_bool_value(
            "True"
        )
        is True
    )


def test_missing_required_numeric_rejected():
    row = base_row()

    row[
        "change_30y_1d_bp"
    ] = None

    with pytest.raises(
        ValueError,
        match="change_30y_1d_bp",
    ):
        build_risk_inputs_v1(
            row=row,
            position=PositionMemoryV1(),
        )