import pytest

from src.risk.intraday_interface import (
    IntradayAction,
    IntradayEmergencySignal,
)

from src.risk.risk_governor_v2 import (
    ExposureState,
    GovernorMemory,
    RiskInputsV2,
    evaluate_risk_v2,
)


def safe_inputs(
    **overrides,
) -> RiskInputsV2:
    values = {
        "tmf_return_1d_pct": 0.5,
        "yield_30y_change_1d_bp": 1.0,
        "yield_30y_change_5d_bp": 2.0,
        "real_yield_10y_change_5d_bp": 1.0,
        "volatility_ratio_5d_20d": 1.0,
        "below_ema21": False,
        "critical_data_stale": False,
        "position_open": False,
        "drawdown_from_entry_pct": None,
    }

    values.update(
        overrides
    )

    return RiskInputsV2(
        **values
    )


def test_safe_market_is_green_and_entry_allowed():
    decision, _ = evaluate_risk_v2(
        safe_inputs()
    )

    assert (
        decision.exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.entry_blocked
        is False
    )

    assert (
        decision.new_entries_allowed
        is True
    )

    assert (
        decision.max_position_multiplier
        == 1.0
    )


def test_below_ema_blocks_entry_without_reducing_exposure():
    decision, _ = evaluate_risk_v2(
        safe_inputs(
            below_ema21=True,
        )
    )

    assert (
        decision.exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.entry_blocked
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


def test_stale_data_blocks_entry_without_reducing_exposure():
    decision, _ = evaluate_risk_v2(
        safe_inputs(
            critical_data_stale=True,
        )
    )

    assert (
        decision.exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.entry_blocked
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


def test_ema_and_stale_data_do_not_create_exposure_warning():
    decision, _ = evaluate_risk_v2(
        safe_inputs(
            below_ema21=True,
            critical_data_stale=True,
        )
    )

    assert (
        decision.raw_exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.yellow_warning_count
        == 0
    )

    assert (
        decision.entry_blocked
        is True
    )


def test_market_warning_reduces_exposure():
    decision, _ = evaluate_risk_v2(
        safe_inputs(
            yield_30y_change_1d_bp=10.0,
        )
    )

    assert (
        decision.exposure_state
        == ExposureState.YELLOW
    )

    assert (
        decision.entry_blocked
        is True
    )

    assert (
        decision.max_position_multiplier
        == 0.5
    )


def test_multiple_exposure_warnings_escalate_to_red():
    decision, _ = evaluate_risk_v2(
        safe_inputs(
            tmf_return_1d_pct=-3.5,
            yield_30y_change_1d_bp=10.0,
            real_yield_10y_change_5d_bp=12.0,
        )
    )

    assert (
        decision.raw_exposure_state
        == ExposureState.RED
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    assert (
        decision.yellow_warning_count
        >= 3
    )

    assert (
        decision.exit_required
        is True
    )


def test_deterioration_is_immediate():
    memory = GovernorMemory(
        state=ExposureState.GREEN,
        recovery_streak=0,
    )

    decision, memory = evaluate_risk_v2(
        safe_inputs(
            yield_30y_change_5d_bp=30.0,
        ),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    assert (
        memory.state
        == ExposureState.RED
    )


def test_red_recovery_requires_two_sessions_per_step():
    memory = GovernorMemory()

    # Immediate GREEN -> RED.
    decision, memory = evaluate_risk_v2(
        safe_inputs(
            yield_30y_change_5d_bp=30.0,
        ),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    # First safer observation:
    # RED remains RED.
    decision, memory = evaluate_risk_v2(
        safe_inputs(),
        memory=memory,
    )

    assert (
        decision.raw_exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    assert (
        decision.recovery_streak
        == 1
    )

    # Second safer observation:
    # RED -> YELLOW.
    decision, memory = evaluate_risk_v2(
        safe_inputs(),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.YELLOW
    )

    assert (
        decision.recovery_streak
        == 0
    )

    # One more safe day is not enough.
    decision, memory = evaluate_risk_v2(
        safe_inputs(),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.YELLOW
    )

    assert (
        decision.recovery_streak
        == 1
    )

    # Second safe day:
    # YELLOW -> GREEN.
    decision, memory = evaluate_risk_v2(
        safe_inputs(),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.recovery_streak
        == 0
    )


def test_equal_risk_state_resets_recovery_streak():
    memory = GovernorMemory(
        state=ExposureState.RED,
        recovery_streak=1,
    )

    decision, memory = evaluate_risk_v2(
        safe_inputs(
            yield_30y_change_5d_bp=30.0,
        ),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    assert (
        decision.recovery_streak
        == 0
    )


def test_intraday_block_entry_does_not_reduce_exposure():
    signal = IntradayEmergencySignal(
        action=(
            IntradayAction.BLOCK_ENTRY
        ),
        reasons=(
            "Intraday monitor detected "
            "abnormal conditions.",
        ),
    )

    decision, _ = evaluate_risk_v2(
        safe_inputs(),
        intraday_signal=signal,
    )

    assert (
        decision.exposure_state
        == ExposureState.GREEN
    )

    assert (
        decision.entry_blocked
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


def test_intraday_force_exit_overrides_safe_market():
    signal = IntradayEmergencySignal(
        action=(
            IntradayAction.FORCE_EXIT
        ),
        reasons=(
            "Intraday emergency stop triggered.",
        ),
    )

    decision, _ = evaluate_risk_v2(
        safe_inputs(),
        intraday_signal=signal,
    )

    assert (
        decision.raw_exposure_state
        == ExposureState.EMERGENCY_EXIT
    )

    assert (
        decision.exposure_state
        == ExposureState.EMERGENCY_EXIT
    )

    assert (
        decision.exit_required
        is True
    )

    assert (
        decision.cooldown_required
        is True
    )

    assert (
        decision.max_position_multiplier
        == 0.0
    )


def test_direct_emergency_still_works():
    decision, _ = evaluate_risk_v2(
        safe_inputs(
            tmf_return_1d_pct=-8.5,
        )
    )

    assert (
        decision.exposure_state
        == ExposureState.EMERGENCY_EXIT
    )

    assert (
        decision.exit_required
        is True
    )


def test_open_position_requires_drawdown():
    with pytest.raises(
        ValueError,
        match="drawdown_from_entry_pct",
    ):
        evaluate_risk_v2(
            safe_inputs(
                position_open=True,
                drawdown_from_entry_pct=None,
            )
        )


def test_invalid_volatility_ratio_rejected():
    with pytest.raises(
        ValueError,
        match="volatility_ratio_5d_20d",
    ):
        evaluate_risk_v2(
            safe_inputs(
                volatility_ratio_5d_20d=0.0,
            )
        )