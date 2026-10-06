import pytest

from src.risk.intraday_interface import (
    IntradayAction,
    IntradayEmergencySignal,
)

from src.risk.risk_governor_v2 import (
    ExposureState,
    GovernorMemory,
)

from src.risk.risk_governor_v3 import (
    DEFAULT_V3_POLICY,
    RiskInputsV3,
    evaluate_risk_v3,
)


def safe_inputs(
    **overrides,
) -> RiskInputsV3:
    values = {
        "tmf_return_1d_pct": 0.5,
        "yield_30y_change_1d_bp": 1.0,
        "yield_30y_change_5d_bp": 2.0,
        "real_yield_10y_change_5d_bp": 1.0,
        "volatility_ratio_5d_20d": 1.0,
        "below_ema21": False,
        "below_ema21_streak": 0,
        "critical_data_stale": False,
        "position_open": False,
        "drawdown_from_entry_pct": None,
    }

    values.update(
        overrides
    )

    return RiskInputsV3(
        **values
    )


def test_v3_ema_threshold_is_four_sessions():
    assert (
        DEFAULT_V3_POLICY
        .ema_entry_block_streak
        == 4
    )


def test_above_ema_allows_entry_in_green_market():
    decision, _ = evaluate_risk_v3(
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


@pytest.mark.parametrize(
    "streak",
    [1, 2, 3],
)
def test_short_ema_break_does_not_block_entry(
    streak,
):
    decision, _ = evaluate_risk_v3(
        safe_inputs(
            below_ema21=True,
            below_ema21_streak=streak,
        )
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


def test_four_day_ema_break_blocks_entry():
    decision, _ = evaluate_risk_v3(
        safe_inputs(
            below_ema21=True,
            below_ema21_streak=4,
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


def test_long_ema_break_blocks_entry():
    decision, _ = evaluate_risk_v3(
        safe_inputs(
            below_ema21=True,
            below_ema21_streak=12,
        )
    )

    assert (
        decision.entry_blocked
        is True
    )


def test_ema_persistence_does_not_reduce_exposure():
    decision, _ = evaluate_risk_v3(
        safe_inputs(
            below_ema21=True,
            below_ema21_streak=20,
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
        decision.max_position_multiplier
        == 1.0
    )

    assert (
        decision.exit_required
        is False
    )


def test_stale_data_still_blocks_entry():
    decision, _ = evaluate_risk_v3(
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


def test_yellow_market_still_blocks_entry():
    decision, _ = evaluate_risk_v3(
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


def test_red_market_still_forces_exit():
    decision, _ = evaluate_risk_v3(
        safe_inputs(
            yield_30y_change_5d_bp=30.0,
        )
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    assert (
        decision.exit_required
        is True
    )

    assert (
        decision.max_position_multiplier
        == 0.0
    )


def test_v3_preserves_v2_hysteresis():
    memory = GovernorMemory()

    decision, memory = evaluate_risk_v3(
        safe_inputs(
            yield_30y_change_5d_bp=30.0,
        ),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    decision, memory = evaluate_risk_v3(
        safe_inputs(),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.RED
    )

    assert (
        decision.recovery_streak
        == 1
    )

    decision, memory = evaluate_risk_v3(
        safe_inputs(),
        memory=memory,
    )

    assert (
        decision.exposure_state
        == ExposureState.YELLOW
    )


def test_intraday_block_entry_still_works():
    signal = IntradayEmergencySignal(
        action=(
            IntradayAction.BLOCK_ENTRY
        ),
        reasons=(
            "Intraday risk warning.",
        ),
    )

    decision, _ = evaluate_risk_v3(
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
        decision.exit_required
        is False
    )


def test_intraday_force_exit_still_overrides_everything():
    signal = IntradayEmergencySignal(
        action=(
            IntradayAction.FORCE_EXIT
        ),
        reasons=(
            "Emergency stop.",
        ),
    )

    decision, _ = evaluate_risk_v3(
        safe_inputs(),
        intraday_signal=signal,
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


def test_above_ema_requires_zero_streak():
    with pytest.raises(
        ValueError,
        match="below_ema21_streak must be 0",
    ):
        evaluate_risk_v3(
            safe_inputs(
                below_ema21=False,
                below_ema21_streak=2,
            )
        )


def test_below_ema_requires_positive_streak():
    with pytest.raises(
        ValueError,
        match="below_ema21_streak must be at least 1",
    ):
        evaluate_risk_v3(
            safe_inputs(
                below_ema21=True,
                below_ema21_streak=0,
            )
        )


def test_negative_streak_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        evaluate_risk_v3(
            safe_inputs(
                below_ema21_streak=-1,
            )
        )