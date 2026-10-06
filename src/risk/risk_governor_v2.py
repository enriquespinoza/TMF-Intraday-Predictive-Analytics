"""
V4 TMF Risk Governor V2.

V2 separates:

1. Entry permission
2. Existing-position exposure control

It also adds stateful recovery hysteresis and an explicit interface
for a future intraday emergency layer.

Market-risk thresholds are intentionally unchanged from V1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.risk.intraday_interface import (
    IntradayAction,
    IntradayEmergencySignal,
)


class ExposureState(str, Enum):
    """Risk states controlling existing exposure."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


STATE_SEVERITY = {
    ExposureState.GREEN: 0,
    ExposureState.YELLOW: 1,
    ExposureState.RED: 2,
    ExposureState.EMERGENCY_EXIT: 3,
}


SEVERITY_STATE = {
    value: key
    for key, value
    in STATE_SEVERITY.items()
}


@dataclass(frozen=True)
class RiskThresholdsV2:
    """
    Frozen Risk Governor V2 thresholds.

    These numerical market-risk thresholds are unchanged from V1.
    """

    # TMF one-day return.
    yellow_tmf_return_1d_pct: float = -3.0
    red_tmf_return_1d_pct: float = -5.0
    emergency_tmf_return_1d_pct: float = -8.0

    # Position drawdown.
    yellow_drawdown_from_entry_pct: float = -3.0
    red_drawdown_from_entry_pct: float = -5.0
    emergency_drawdown_from_entry_pct: float = -8.0

    # 30-year yield.
    yellow_30y_change_1d_bp: float = 8.0
    red_30y_change_1d_bp: float = 15.0
    emergency_30y_change_1d_bp: float = 25.0

    yellow_30y_change_5d_bp: float = 15.0
    red_30y_change_5d_bp: float = 25.0

    # 10-year real yield.
    yellow_real_10y_change_5d_bp: float = 10.0
    red_real_10y_change_5d_bp: float = 20.0

    # Volatility.
    yellow_volatility_ratio: float = 1.35
    red_volatility_ratio: float = 1.80

    # Multi-warning escalation.
    yellow_warnings_for_red: int = 3

    # Recovery hysteresis.
    recovery_sessions_per_step: int = 2


DEFAULT_THRESHOLDS_V2 = (
    RiskThresholdsV2()
)


@dataclass(frozen=True)
class RiskInputsV2:
    """
    Inputs consumed by Risk Governor V2.

    EMA21 and stale-data inputs affect entry permission only.

    They do NOT directly affect exposure state.
    """

    tmf_return_1d_pct: float

    yield_30y_change_1d_bp: float
    yield_30y_change_5d_bp: float

    real_yield_10y_change_5d_bp: float

    volatility_ratio_5d_20d: float

    below_ema21: bool
    critical_data_stale: bool

    position_open: bool

    drawdown_from_entry_pct: Optional[
        float
    ] = None


@dataclass(frozen=True)
class GovernorMemory:
    """
    Stateful information required for recovery hysteresis.

    state:
        Current effective exposure state.

    recovery_streak:
        Number of consecutive observations with a raw state less
        severe than the effective state.
    """

    state: ExposureState = (
        ExposureState.GREEN
    )

    recovery_streak: int = 0


@dataclass(frozen=True)
class RiskDecisionV2:
    """Complete Risk Governor V2 decision."""

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


def validate_inputs(
    inputs: RiskInputsV2,
) -> None:
    """Validate basic consistency."""

    if (
        inputs.volatility_ratio_5d_20d
        <= 0
    ):
        raise ValueError(
            "volatility_ratio_5d_20d "
            "must be greater than zero."
        )

    if (
        inputs.position_open
        and inputs.drawdown_from_entry_pct
        is None
    ):
        raise ValueError(
            "drawdown_from_entry_pct "
            "is required when a position is open."
        )


def emergency_conditions(
    inputs: RiskInputsV2,
    thresholds: RiskThresholdsV2,
) -> list[str]:
    """Return direct emergency exposure reasons."""

    reasons: list[str] = []

    if (
        inputs.tmf_return_1d_pct
        <= thresholds
        .emergency_tmf_return_1d_pct
    ):
        reasons.append(
            "TMF one-day return reached "
            "emergency threshold."
        )

    if (
        inputs.position_open
        and inputs.drawdown_from_entry_pct
        is not None
        and inputs.drawdown_from_entry_pct
        <= thresholds
        .emergency_drawdown_from_entry_pct
    ):
        reasons.append(
            "Position drawdown reached "
            "emergency threshold."
        )

    if (
        inputs.yield_30y_change_1d_bp
        >= thresholds
        .emergency_30y_change_1d_bp
    ):
        reasons.append(
            "30-year Treasury yield "
            "one-day increase reached "
            "emergency threshold."
        )

    return reasons


def red_conditions(
    inputs: RiskInputsV2,
    thresholds: RiskThresholdsV2,
) -> list[str]:
    """Return direct RED exposure reasons."""

    reasons: list[str] = []

    if (
        inputs.tmf_return_1d_pct
        <= thresholds
        .red_tmf_return_1d_pct
    ):
        reasons.append(
            "TMF one-day return reached "
            "RED threshold."
        )

    if (
        inputs.position_open
        and inputs.drawdown_from_entry_pct
        is not None
        and inputs.drawdown_from_entry_pct
        <= thresholds
        .red_drawdown_from_entry_pct
    ):
        reasons.append(
            "Position drawdown reached "
            "RED threshold."
        )

    if (
        inputs.yield_30y_change_1d_bp
        >= thresholds
        .red_30y_change_1d_bp
    ):
        reasons.append(
            "30-year Treasury yield "
            "one-day increase reached "
            "RED threshold."
        )

    if (
        inputs.yield_30y_change_5d_bp
        >= thresholds
        .red_30y_change_5d_bp
    ):
        reasons.append(
            "30-year Treasury yield "
            "five-day increase reached "
            "RED threshold."
        )

    if (
        inputs.real_yield_10y_change_5d_bp
        >= thresholds
        .red_real_10y_change_5d_bp
    ):
        reasons.append(
            "10-year real yield "
            "five-day increase reached "
            "RED threshold."
        )

    if (
        inputs.volatility_ratio_5d_20d
        >= thresholds
        .red_volatility_ratio
    ):
        reasons.append(
            "Realized volatility expansion "
            "reached RED threshold."
        )

    return reasons


def yellow_conditions(
    inputs: RiskInputsV2,
    thresholds: RiskThresholdsV2,
) -> list[str]:
    """
    Return moderate exposure-risk warnings.

    Important:
    EMA21 and stale data are NOT evaluated here.
    """

    reasons: list[str] = []

    if (
        inputs.tmf_return_1d_pct
        <= thresholds
        .yellow_tmf_return_1d_pct
    ):
        reasons.append(
            "TMF one-day return reached "
            "YELLOW threshold."
        )

    if (
        inputs.position_open
        and inputs.drawdown_from_entry_pct
        is not None
        and inputs.drawdown_from_entry_pct
        <= thresholds
        .yellow_drawdown_from_entry_pct
    ):
        reasons.append(
            "Position drawdown reached "
            "YELLOW threshold."
        )

    if (
        inputs.yield_30y_change_1d_bp
        >= thresholds
        .yellow_30y_change_1d_bp
    ):
        reasons.append(
            "30-year Treasury yield "
            "one-day increase reached "
            "YELLOW threshold."
        )

    if (
        inputs.yield_30y_change_5d_bp
        >= thresholds
        .yellow_30y_change_5d_bp
    ):
        reasons.append(
            "30-year Treasury yield "
            "five-day increase reached "
            "YELLOW threshold."
        )

    if (
        inputs.real_yield_10y_change_5d_bp
        >= thresholds
        .yellow_real_10y_change_5d_bp
    ):
        reasons.append(
            "10-year real yield "
            "five-day increase reached "
            "YELLOW threshold."
        )

    if (
        inputs.volatility_ratio_5d_20d
        >= thresholds
        .yellow_volatility_ratio
    ):
        reasons.append(
            "Realized volatility expansion "
            "reached YELLOW threshold."
        )

    return reasons


def determine_raw_exposure_state(
    inputs: RiskInputsV2,
    thresholds: RiskThresholdsV2,
    intraday_signal: IntradayEmergencySignal,
) -> tuple[
    ExposureState,
    list[str],
    int,
]:
    """
    Determine current raw exposure state before hysteresis.
    """

    if (
        intraday_signal.action
        == IntradayAction.FORCE_EXIT
    ):
        reasons = list(
            intraday_signal.reasons
        )

        if not reasons:
            reasons.append(
                "Intraday emergency layer "
                "required immediate exit."
            )

        return (
            ExposureState.EMERGENCY_EXIT,
            reasons,
            0,
        )

    emergency = emergency_conditions(
        inputs,
        thresholds,
    )

    if emergency:
        return (
            ExposureState.EMERGENCY_EXIT,
            emergency,
            0,
        )

    red = red_conditions(
        inputs,
        thresholds,
    )

    yellow = yellow_conditions(
        inputs,
        thresholds,
    )

    yellow_count = len(
        yellow
    )

    if red:
        return (
            ExposureState.RED,
            red,
            yellow_count,
        )

    if (
        yellow_count
        >= thresholds
        .yellow_warnings_for_red
    ):
        reasons = list(
            yellow
        )

        reasons.append(
            "Multiple YELLOW exposure warnings "
            "escalated raw exposure state to RED."
        )

        return (
            ExposureState.RED,
            reasons,
            yellow_count,
        )

    if yellow:
        return (
            ExposureState.YELLOW,
            yellow,
            yellow_count,
        )

    return (
        ExposureState.GREEN,
        [],
        0,
    )


def apply_recovery_hysteresis(
    raw_state: ExposureState,
    memory: GovernorMemory,
    thresholds: RiskThresholdsV2,
) -> tuple[
    ExposureState,
    int,
    Optional[str],
]:
    """
    Apply asymmetric state transitions.

    Deterioration:
        Immediate.

    Recovery:
        Requires consecutive observations with a raw state less
        severe than the current effective state.

        Once the recovery threshold is reached, the effective state
        improves by exactly one severity level.
    """

    current_state = (
        memory.state
    )

    raw_severity = (
        STATE_SEVERITY[
            raw_state
        ]
    )

    current_severity = (
        STATE_SEVERITY[
            current_state
        ]
    )

    # Immediate deterioration.
    if (
        raw_severity
        > current_severity
    ):
        return (
            raw_state,
            0,
            None,
        )

    # No improvement signal.
    if (
        raw_severity
        == current_severity
    ):
        return (
            current_state,
            0,
            None,
        )

    # Raw state is safer than effective state.
    recovery_streak = (
        memory.recovery_streak
        + 1
    )

    if (
        recovery_streak
        < thresholds
        .recovery_sessions_per_step
    ):
        note = (
            f"Recovery hysteresis retained "
            f"{current_state.value}; "
            f"{recovery_streak}/"
            f"{thresholds.recovery_sessions_per_step} "
            f"recovery sessions completed."
        )

        return (
            current_state,
            recovery_streak,
            note,
        )

    improved_severity = (
        current_severity
        - 1
    )

    improved_state = (
        SEVERITY_STATE[
            improved_severity
        ]
    )

    note = (
        f"Recovery hysteresis improved "
        f"{current_state.value} to "
        f"{improved_state.value} after "
        f"{thresholds.recovery_sessions_per_step} "
        f"consecutive less-severe observations."
    )

    return (
        improved_state,
        0,
        note,
    )


def entry_gate_reasons(
    inputs: RiskInputsV2,
    exposure_state: ExposureState,
    intraday_signal: IntradayEmergencySignal,
) -> list[str]:
    """
    Determine reasons a new entry is prohibited.

    These reasons are separate from exposure reduction.
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

    if inputs.below_ema21:
        reasons.append(
            "TMF is below its 21-day EMA."
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


def max_exposure_for_state(
    state: ExposureState,
) -> float:
    """Map effective exposure state to maximum position multiplier."""

    if state == ExposureState.GREEN:
        return 1.0

    if state == ExposureState.YELLOW:
        return 0.5

    if state in {
        ExposureState.RED,
        ExposureState.EMERGENCY_EXIT,
    }:
        return 0.0

    raise ValueError(
        f"Unsupported exposure state: {state}"
    )


def evaluate_risk_v2(
    inputs: RiskInputsV2,
    memory: GovernorMemory = GovernorMemory(),
    thresholds: RiskThresholdsV2 = (
        DEFAULT_THRESHOLDS_V2
    ),
    intraday_signal: IntradayEmergencySignal = (
        IntradayEmergencySignal()
    ),
) -> tuple[
    RiskDecisionV2,
    GovernorMemory,
]:
    """
    Evaluate one Risk Governor V2 observation.

    Returns:

        decision
        updated_memory

    The updated memory must be passed into the next chronological
    observation when evaluating a time series.
    """

    validate_inputs(
        inputs
    )

    (
        raw_state,
        exposure_reasons,
        yellow_warning_count,
    ) = determine_raw_exposure_state(
        inputs=inputs,
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
        entry_gate_reasons(
            inputs=inputs,
            exposure_state=effective_state,
            intraday_signal=intraday_signal,
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

    decision = RiskDecisionV2(
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