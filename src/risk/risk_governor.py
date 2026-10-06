"""
V4 TMF Risk Governor.

The risk governor is an independent fail-safe layer for the TMF
trading system.

It does not predict returns.

It determines whether current market conditions permit capital
exposure.

State priority:

    EMERGENCY_EXIT
        >
    RED
        >
    YELLOW
        >
    GREEN

The initial thresholds are prespecified V1 research defaults.
They are deliberately transparent and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskState(str, Enum):
    """Permitted V4 risk states."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


@dataclass(frozen=True)
class RiskThresholds:
    """
    Frozen Risk Governor V1 thresholds.

    Percent inputs use percentage points.

    Example:
        -3.0 means -3%.

    Yield changes use basis points.
    """

    # TMF one-day return.
    yellow_tmf_return_1d_pct: float = -3.0
    red_tmf_return_1d_pct: float = -5.0
    emergency_tmf_return_1d_pct: float = -8.0

    # Position drawdown.
    yellow_drawdown_from_entry_pct: float = -3.0
    red_drawdown_from_entry_pct: float = -5.0
    emergency_drawdown_from_entry_pct: float = -8.0

    # 30-year yield change.
    yellow_30y_change_1d_bp: float = 8.0
    red_30y_change_1d_bp: float = 15.0
    emergency_30y_change_1d_bp: float = 25.0

    yellow_30y_change_5d_bp: float = 15.0
    red_30y_change_5d_bp: float = 25.0

    # Real-yield deterioration.
    yellow_real_10y_change_5d_bp: float = 10.0
    red_real_10y_change_5d_bp: float = 20.0

    # Volatility expansion.
    yellow_volatility_ratio: float = 1.35
    red_volatility_ratio: float = 1.80

    # Multiple moderate warnings escalate to RED.
    yellow_warnings_for_red: int = 3


DEFAULT_THRESHOLDS = RiskThresholds()


@dataclass(frozen=True)
class RiskInputs:
    """
    Market and position information consumed by the governor.

    All inputs must represent information available at the decision
    timestamp.

    tmf_return_1d_pct:
        One-day TMF percentage return.

    drawdown_from_entry_pct:
        Current percentage change from entry price.
        Required only when position_open=True.

    yield_30y_change_1d_bp:
        One-day change in 30-year Treasury yield in basis points.

    yield_30y_change_5d_bp:
        Five-trading-day change in 30-year Treasury yield.

    real_yield_10y_change_5d_bp:
        Five-trading-day change in 10-year real yield.

    volatility_ratio_5d_20d:
        5-day realized volatility divided by 20-day realized
        volatility.

    below_ema21:
        Whether TMF is below its 21-day EMA.

    critical_data_stale:
        Whether required market data is stale or unavailable.

    position_open:
        Whether the strategy currently owns TMF.
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
class RiskDecision:
    """Final governor decision."""

    state: RiskState

    new_entries_allowed: bool

    max_position_multiplier: float

    exit_required: bool

    cooldown_required: bool

    yellow_warning_count: int

    reasons: tuple[str, ...]


def validate_inputs(
    inputs: RiskInputs,
) -> None:
    """Validate basic risk-input consistency."""

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


def evaluate_emergency_conditions(
    inputs: RiskInputs,
    thresholds: RiskThresholds,
) -> list[str]:
    """Return all emergency-condition reasons."""

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


def evaluate_red_conditions(
    inputs: RiskInputs,
    thresholds: RiskThresholds,
) -> list[str]:
    """Return all direct RED-condition reasons."""

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


def evaluate_yellow_conditions(
    inputs: RiskInputs,
    thresholds: RiskThresholds,
) -> list[str]:
    """Return moderate-risk warning reasons."""

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

    if inputs.below_ema21:
        reasons.append(
            "TMF is below its 21-day EMA."
        )

    if inputs.critical_data_stale:
        reasons.append(
            "Critical market data is stale."
        )

    return reasons


def decision_from_state(
    state: RiskState,
    reasons: list[str],
    yellow_warning_count: int,
) -> RiskDecision:
    """Map risk state to trading permissions."""

    if state == RiskState.GREEN:
        return RiskDecision(
            state=state,
            new_entries_allowed=True,
            max_position_multiplier=1.0,
            exit_required=False,
            cooldown_required=False,
            yellow_warning_count=(
                yellow_warning_count
            ),
            reasons=tuple(reasons),
        )

    if state == RiskState.YELLOW:
        return RiskDecision(
            state=state,
            new_entries_allowed=False,
            max_position_multiplier=0.5,
            exit_required=False,
            cooldown_required=False,
            yellow_warning_count=(
                yellow_warning_count
            ),
            reasons=tuple(reasons),
        )

    if state == RiskState.RED:
        return RiskDecision(
            state=state,
            new_entries_allowed=False,
            max_position_multiplier=0.0,
            exit_required=True,
            cooldown_required=False,
            yellow_warning_count=(
                yellow_warning_count
            ),
            reasons=tuple(reasons),
        )

    if (
        state
        == RiskState.EMERGENCY_EXIT
    ):
        return RiskDecision(
            state=state,
            new_entries_allowed=False,
            max_position_multiplier=0.0,
            exit_required=True,
            cooldown_required=True,
            yellow_warning_count=(
                yellow_warning_count
            ),
            reasons=tuple(reasons),
        )

    raise ValueError(
        f"Unsupported risk state: {state}"
    )


def evaluate_risk(
    inputs: RiskInputs,
    thresholds: RiskThresholds = (
        DEFAULT_THRESHOLDS
    ),
) -> RiskDecision:
    """
    Evaluate the current TMF risk environment.

    Evaluation priority:

        emergency
        direct red
        multiple yellow warnings
        yellow
        green
    """

    validate_inputs(
        inputs
    )

    emergency_reasons = (
        evaluate_emergency_conditions(
            inputs,
            thresholds,
        )
    )

    red_reasons = (
        evaluate_red_conditions(
            inputs,
            thresholds,
        )
    )

    yellow_reasons = (
        evaluate_yellow_conditions(
            inputs,
            thresholds,
        )
    )

    yellow_warning_count = len(
        yellow_reasons
    )

    if emergency_reasons:
        return decision_from_state(
            state=(
                RiskState
                .EMERGENCY_EXIT
            ),
            reasons=(
                emergency_reasons
            ),
            yellow_warning_count=(
                yellow_warning_count
            ),
        )

    if red_reasons:
        return decision_from_state(
            state=RiskState.RED,
            reasons=red_reasons,
            yellow_warning_count=(
                yellow_warning_count
            ),
        )

    if (
        yellow_warning_count
        >= thresholds
        .yellow_warnings_for_red
    ):
        reasons = list(
            yellow_reasons
        )

        reasons.append(
            "Multiple YELLOW warnings "
            "escalated risk state to RED."
        )

        return decision_from_state(
            state=RiskState.RED,
            reasons=reasons,
            yellow_warning_count=(
                yellow_warning_count
            ),
        )

    if yellow_reasons:
        return decision_from_state(
            state=RiskState.YELLOW,
            reasons=yellow_reasons,
            yellow_warning_count=(
                yellow_warning_count
            ),
        )

    return decision_from_state(
        state=RiskState.GREEN,
        reasons=[],
        yellow_warning_count=0,
    )