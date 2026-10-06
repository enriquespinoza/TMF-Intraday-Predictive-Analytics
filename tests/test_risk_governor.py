import pytest

from src.risk.risk_governor import (
    RiskInputs,
    RiskState,
    evaluate_risk,
)


def safe_inputs(
    **overrides,
) -> RiskInputs:
    values = {
        "tmf_return_1d_pct":
            0.25,

        "yield_30y_change_1d_bp":
            1.0,

        "yield_30y_change_5d_bp":
            2.0,

        "real_yield_10y_change_5d_bp":
            1.0,

        "volatility_ratio_5d_20d":
            1.0,

        "below_ema21":
            False,

        "critical_data_stale":
            False,

        "position_open":
            False,

        "drawdown_from_entry_pct":
            None,
    }

    values.update(
        overrides
    )

    return RiskInputs(
        **values
    )


def test_normal_market_is_green():
    decision = evaluate_risk(
        safe_inputs()
    )

    assert (
        decision.state
        == RiskState.GREEN
    )

    assert (
        decision.new_entries_allowed
        is True
    )

    assert (
        decision.max_position_multiplier
        == 1.0
    )

    assert (
        decision.exit_required
        is False
    )


def test_single_warning_is_yellow():
    decision = evaluate_risk(
        safe_inputs(
            below_ema21=True,
        )
    )

    assert (
        decision.state
        == RiskState.YELLOW
    )

    assert (
        decision.new_entries_allowed
        is False
    )

    assert (
        decision.max_position_multiplier
        == 0.5
    )

    assert (
        decision.exit_required
        is False
    )


def test_stale_data_blocks_new_entries():
    decision = evaluate_risk(
        safe_inputs(
            critical_data_stale=True,
        )
    )

    assert (
        decision.state
        == RiskState.YELLOW
    )

    assert (
        decision.new_entries_allowed
        is False
    )

    assert (
        decision.exit_required
        is False
    )


def test_direct_red_yield_move():
    decision = evaluate_risk(
        safe_inputs(
            yield_30y_change_1d_bp=16.0,
        )
    )

    assert (
        decision.state
        == RiskState.RED
    )

    assert (
        decision.exit_required
        is True
    )

    assert (
        decision.max_position_multiplier
        == 0.0
    )


def test_multiple_yellow_conditions_escalate_to_red():
    decision = evaluate_risk(
        safe_inputs(
            tmf_return_1d_pct=-3.5,
            yield_30y_change_1d_bp=10.0,
            below_ema21=True,
        )
    )

    assert (
        decision.state
        == RiskState.RED
    )

    assert (
        decision.yellow_warning_count
        >= 3
    )

    assert (
        decision.exit_required
        is True
    )


def test_emergency_position_drawdown():
    decision = evaluate_risk(
        safe_inputs(
            position_open=True,
            drawdown_from_entry_pct=-8.5,
        )
    )

    assert (
        decision.state
        == RiskState.EMERGENCY_EXIT
    )

    assert (
        decision.exit_required
        is True
    )

    assert (
        decision.cooldown_required
        is True
    )


def test_emergency_one_day_tmf_move():
    decision = evaluate_risk(
        safe_inputs(
            tmf_return_1d_pct=-8.5,
        )
    )

    assert (
        decision.state
        == RiskState.EMERGENCY_EXIT
    )


def test_emergency_30y_yield_move():
    decision = evaluate_risk(
        safe_inputs(
            yield_30y_change_1d_bp=26.0,
        )
    )

    assert (
        decision.state
        == RiskState.EMERGENCY_EXIT
    )


def test_open_position_requires_drawdown():
    with pytest.raises(
        ValueError,
        match=(
            "drawdown_from_entry_pct"
        ),
    ):
        evaluate_risk(
            safe_inputs(
                position_open=True,
                drawdown_from_entry_pct=None,
            )
        )


def test_invalid_volatility_ratio_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "volatility_ratio_5d_20d"
        ),
    ):
        evaluate_risk(
            safe_inputs(
                volatility_ratio_5d_20d=0.0,
            )
        )


def test_yellow_position_does_not_force_exit():
    decision = evaluate_risk(
        safe_inputs(
            position_open=True,
            drawdown_from_entry_pct=-3.5,
        )
    )

    assert (
        decision.state
        == RiskState.YELLOW
    )

    assert (
        decision.exit_required
        is False
    )

    assert (
        decision.max_position_multiplier
        == 0.5
    )


def test_red_position_drawdown_forces_exit():
    decision = evaluate_risk(
        safe_inputs(
            position_open=True,
            drawdown_from_entry_pct=-5.5,
        )
    )

    assert (
        decision.state
        == RiskState.RED
    )

    assert (
        decision.exit_required
        is True
    )