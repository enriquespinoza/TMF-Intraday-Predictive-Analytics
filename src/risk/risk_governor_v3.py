"""
V4 TMF Risk Governor V3.

V3 changes only the EMA21 entry-gate logic from Risk Governor V2.

Exposure-state logic, thresholds, hysteresis, exposure multipliers,
and intraday FORCE_EXIT behavior remain identical to V2.

V3 EMA rule:

    below_ema21_streak 0-3
        -> no automatic EMA entry block

    below_ema21_streak >= 4
        -> new entry blocked
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.risk.intraday_interface import (
    IntradayAction,
    IntradayEmergencySignal,
)

from src.risk.risk_governor_v2 import (
    DEFAULT_THRESHOLDS_V2,
    ExposureState,
    GovernorMemory,
    RiskInputsV2,
    RiskThresholdsV2,
    apply_recovery_hysteresis,
    determine_raw_exposure_state,
    max_exposure_for_state,
    validate_inputs,
)


@dataclass(frozen=True)
class RiskGovernorV3Policy:
    """
    Risk Governor V3 policy parameters.

    ema_entry_block_streak is research-derived from the historical
    EMA21 persistence diagnostic.

    It is not an independently validated threshold.
    """

    ema_entry_block_streak: int = 4


DEFAULT_V3_POLICY = (
    RiskGovernorV3Policy()
)


@dataclass(frozen=True)
class RiskInputsV3:
    """
    Inputs consumed by Risk Governor V3.

    below_ema21_streak:
        Consecutive market sessions with TMF closing below EMA21.

        0 means TMF is currently above EMA21.
    """

    tmf_return_1d_pct: float

    yield_30y_change_1d_bp: float
    yield_30y_change_5d_bp: float

    real_yield_10y_change_5d_bp: float

    volatility_ratio_5d_20d: float

    below_ema21: bool
    below_ema21_streak: int

    critical_data_stale: bool

    position_open: bool

    drawdown_from_entry_pct: Optional[
        float
    ] = None


@dataclass(frozen=True)
class RiskDecisionV3:
    """Complete Risk Governor V3 decision."""

    raw_exposure_state: ExposureState

    exposure_state: ExposureState

    entry_blocked: bool

    new_entries_allowed: bool

    max_position_multiplier: float

    exit_required: bool

    cooldown_required: bool

    yellow_warning_count: int

    recovery_streak: int

    exposure_reasons: tuple[str, ...]

    entry_block_reasons: tuple[str, ...]

    intraday_reasons: tuple[str, ...]


def validate_v3_inputs(
    inputs: RiskInputsV3,
) -> None:
    """Validate V3-specific input consistency."""

    if (
        inputs.below_ema21_streak
        < 0
    ):
        raise ValueError(
            "below_ema21_streak "
            "cannot be negative."
        )

    if (
        not inputs.below_ema21
        and inputs.below_ema21_streak
        != 0
    ):
        raise ValueError(
            "below_ema21_streak must be 0 "
            "when below_ema21 is False."
        )

    if (
        inputs.below_ema21
        and inputs.below_ema21_streak
        == 0
    ):
        raise ValueError(
            "below_ema21_streak must be at least 1 "
            "when below_ema21 is True."
        )

    # Reuse V2's exposure-input validation.
    validate_inputs(
        to_v2_inputs(
            inputs
        )
    )


def to_v2_inputs(
    inputs: RiskInputsV3,
) -> RiskInputsV2:
    """
    Convert V3 inputs to the unchanged V2 exposure input contract.

    EMA persistence is intentionally ignored by the exposure engine.
    """

    return RiskInputsV2(
        tmf_return_1d_pct=(
            inputs.tmf_return_1d_pct
        ),

        yield_30y_change_1d_bp=(
            inputs.yield_30y_change_1d_bp
        ),

        yield_30y_change_5d_bp=(
            inputs.yield_30y_change_5d_bp
        ),

        real_yield_10y_change_5d_bp=(
            inputs.real_yield_10y_change_5d_bp
        ),

        volatility_ratio_5d_20d=(
            inputs.volatility_ratio_5d_20d
        ),

        # V2 exposure calculations do not use this field.
        below_ema21=(
            inputs.below_ema21
        ),

        critical_data_stale=(
            inputs.critical_data_stale
        ),

        position_open=(
            inputs.position_open
        ),

        drawdown_from_entry_pct=(
            inputs.drawdown_from_entry_pct
        ),
    )


def entry_gate_reasons_v3(
    inputs: RiskInputsV3,
    exposure_state: ExposureState,
    intraday_signal: IntradayEmergencySignal,
    policy: RiskGovernorV3Policy,
) -> list[str]:
    """
    Determine V3 entry-block reasons.

    EMA21 blocks entry only after a persistent breakdown.
    """

    reasons: list[str] = []

    if (
        exposure_state
        != ExposureState.GREEN
    ):
        reasons.append(
            f"Exposure state "
            f"{exposure_state.value} "
            "blocks new entries."
        )

    if (
        inputs.below_ema21_streak
        >= policy.ema_entry_block_streak
    ):
        reasons.append(
            "TMF below-EMA21 streak reached "
            f"{policy.ema_entry_block_streak} "
            "sessions."
        )

    if inputs.critical_data_stale:
        reasons.append(
            "Critical market data is stale."
        )

    if (
        intraday_signal.action
        == IntradayAction.BLOCK_ENTRY
    ):
        reasons.append(
            "Intraday emergency layer "
            "blocked new entries."
        )

    if (
        intraday_signal.action
        == IntradayAction.FORCE_EXIT
    ):
        reasons.append(
            "Intraday emergency layer "
            "required immediate exit."
        )

    return reasons


def evaluate_risk_v3(
    inputs: RiskInputsV3,
    memory: GovernorMemory = GovernorMemory(),
    thresholds: RiskThresholdsV2 = (
        DEFAULT_THRESHOLDS_V2
    ),
    policy: RiskGovernorV3Policy = (
        DEFAULT_V3_POLICY
    ),
    intraday_signal: IntradayEmergencySignal = (
        IntradayEmergencySignal()
    ),
) -> tuple[
    RiskDecisionV3,
    GovernorMemory,
]:
    """
    Evaluate one Risk Governor V3 observation.

    Exposure behavior is intentionally inherited from V2.

    Only entry-gate EMA logic changes.
    """

    validate_v3_inputs(
        inputs
    )

    v2_inputs = to_v2_inputs(
        inputs
    )

    (
        raw_state,
        exposure_reasons,
        yellow_warning_count,
    ) = determine_raw_exposure_state(
        inputs=v2_inputs,
        thresholds=thresholds,
        intraday_signal=intraday_signal,
    )

    (
        effective_state,
        recovery_streak,
        hysteresis_note,
    ) = apply_recovery_hysteresis(
        raw_state=raw_state,
        memory=memory,
        thresholds=thresholds,
    )

    if hysteresis_note:
        exposure_reasons = [
            *exposure_reasons,
            hysteresis_note,
        ]

    block_reasons = (
        entry_gate_reasons_v3(
            inputs=inputs,
            exposure_state=effective_state,
            intraday_signal=intraday_signal,
            policy=policy,
        )
    )

    entry_blocked = bool(
        block_reasons
    )

    exit_required = (
        effective_state
        in {
            ExposureState.RED,
            ExposureState.EMERGENCY_EXIT,
        }
    )

    cooldown_required = (
        effective_state
        == ExposureState.EMERGENCY_EXIT
    )

    decision = RiskDecisionV3(
        raw_exposure_state=raw_state,

        exposure_state=effective_state,

        entry_blocked=entry_blocked,

        new_entries_allowed=(
            not entry_blocked
        ),

        max_position_multiplier=(
            max_exposure_for_state(
                effective_state
            )
        ),

        exit_required=exit_required,

        cooldown_required=(
            cooldown_required
        ),

        yellow_warning_count=(
            yellow_warning_count
        ),

        recovery_streak=(
            recovery_streak
        ),

        exposure_reasons=tuple(
            exposure_reasons
        ),

        entry_block_reasons=tuple(
            block_reasons
        ),

        intraday_reasons=tuple(
            intraday_signal.reasons
        ),
    )

    updated_memory = GovernorMemory(
        state=effective_state,
        recovery_streak=(
            recovery_streak
        ),
    )

    return (
        decision,
        updated_memory,
    )