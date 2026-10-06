"""
Execution Architecture V1 position lifecycle.

Opportunity Engine V1 controls new-position initiation.

Risk Governor V3 controls exposure after a position exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Optional

from src.signals.opportunity_state_machine import (
    OpportunityState,
)


DEFAULT_POLICY_FILE = Path(
    "config/"
    "execution_v1_policy.json"
)


ALLOWED_EXPOSURES = {
    0.0,
    0.5,
    1.0,
}


class ExecutionAction(str, Enum):
    """Execution Architecture V1 actions."""

    HOLD_FLAT = "HOLD_FLAT"

    OPEN_POSITION = "OPEN_POSITION"

    HOLD_POSITION = "HOLD_POSITION"

    REDUCE_POSITION = "REDUCE_POSITION"

    RESTORE_POSITION = "RESTORE_POSITION"

    EXIT_POSITION = "EXIT_POSITION"


@dataclass(frozen=True)
class ExecutionPolicyV1:
    """Frozen execution-policy settings."""

    entry_target_exposure: float

    transaction_cost_bps_per_unit_turnover: float

    execution_timing: str

    drawdown_anchor: str


@dataclass(frozen=True)
class PositionMemoryV1:
    """
    Memory for one position cycle.

    entry_anchor_price is established only when moving
    from flat to positive exposure.
    """

    current_exposure: float = 0.0

    entry_anchor_price: Optional[
        float
    ] = None


@dataclass(frozen=True)
class ExecutionInputsV1:
    """Inputs needed for one execution decision."""

    opportunity_state: OpportunityState

    risk_entry_blocked: bool

    risk_max_exposure: float

    risk_force_exit: bool = False

    risk_cooldown_active: bool = False


@dataclass(frozen=True)
class ExecutionDecisionV1:
    """Target exposure decision."""

    action: ExecutionAction

    current_exposure: float

    target_exposure: float

    requires_order: bool

    reasons: tuple[str, ...]


def load_execution_policy_v1(
    path: Path = DEFAULT_POLICY_FILE,
) -> ExecutionPolicyV1:
    """Load frozen Execution Architecture V1."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        policy = json.load(
            file
        )

    return ExecutionPolicyV1(
        entry_target_exposure=float(
            policy[
                "entry_target_exposure"
            ]
        ),

        transaction_cost_bps_per_unit_turnover=float(
            policy[
                "transaction_cost_bps_per_unit_turnover"
            ]
        ),

        execution_timing=str(
            policy[
                "execution_timing"
            ]
        ),

        drawdown_anchor=str(
            policy[
                "drawdown_anchor"
            ]
        ),
    )


def validate_exposure(
    name: str,
    value: float,
) -> None:
    """Validate one V1 exposure level."""

    if float(
        value
    ) not in ALLOWED_EXPOSURES:
        raise ValueError(
            f"{name} must be one of "
            f"{sorted(ALLOWED_EXPOSURES)}."
        )


def validate_position_memory(
    position: PositionMemoryV1,
) -> None:
    """Validate position memory consistency."""

    validate_exposure(
        "current_exposure",
        position.current_exposure,
    )

    if (
        position.current_exposure == 0.0
        and position.entry_anchor_price is not None
    ):
        raise ValueError(
            "Flat position must not retain "
            "an entry anchor."
        )

    if (
        position.current_exposure > 0.0
        and position.entry_anchor_price is None
    ):
        raise ValueError(
            "Open position requires an "
            "entry anchor."
        )

    if (
        position.entry_anchor_price is not None
        and position.entry_anchor_price <= 0
    ):
        raise ValueError(
            "entry_anchor_price must be positive."
        )


def validate_inputs(
    inputs: ExecutionInputsV1,
) -> None:
    """Validate execution inputs."""

    validate_exposure(
        "risk_max_exposure",
        inputs.risk_max_exposure,
    )


def evaluate_execution_v1(
    inputs: ExecutionInputsV1,
    position: PositionMemoryV1,
    policy: ExecutionPolicyV1,
) -> ExecutionDecisionV1:
    """
    Determine the next target exposure.

    Opportunity Engine controls new entries only.

    Risk Governor controls an existing position.
    """

    validate_inputs(
        inputs
    )

    validate_position_memory(
        position
    )

    current = float(
        position.current_exposure
    )

    reasons: list[str] = []

    # ---------------------------------------------------------
    # Risk FORCE_EXIT always has highest execution authority.
    # ---------------------------------------------------------

    if inputs.risk_force_exit:

        if current == 0.0:

            reasons.append(
                "Risk Governor FORCE_EXIT active; "
                "remain flat."
            )

            return ExecutionDecisionV1(
                action=ExecutionAction.HOLD_FLAT,
                current_exposure=current,
                target_exposure=0.0,
                requires_order=False,
                reasons=tuple(
                    reasons
                ),
            )

        reasons.append(
            "Risk Governor FORCE_EXIT requires "
            "full liquidation."
        )

        return ExecutionDecisionV1(
            action=ExecutionAction.EXIT_POSITION,
            current_exposure=current,
            target_exposure=0.0,
            requires_order=True,
            reasons=tuple(
                reasons
            ),
        )

    # ---------------------------------------------------------
    # Existing position:
    # Risk Governor has sole sizing / exit authority.
    # Opportunity state is intentionally ignored.
    # ---------------------------------------------------------

    if current > 0.0:

        if inputs.risk_cooldown_active:

            reasons.append(
                "Risk cooldown requires position "
                "cycle to be flat."
            )

            return ExecutionDecisionV1(
                action=ExecutionAction.EXIT_POSITION,
                current_exposure=current,
                target_exposure=0.0,
                requires_order=True,
                reasons=tuple(
                    reasons
                ),
            )

        target = float(
            inputs.risk_max_exposure
        )

        if target == 0.0:

            reasons.append(
                "Risk Governor maximum exposure "
                "is zero."
            )

            return ExecutionDecisionV1(
                action=ExecutionAction.EXIT_POSITION,
                current_exposure=current,
                target_exposure=0.0,
                requires_order=True,
                reasons=tuple(
                    reasons
                ),
            )

        if target < current:

            reasons.append(
                "Risk Governor requires reduced "
                "exposure."
            )

            return ExecutionDecisionV1(
                action=ExecutionAction.REDUCE_POSITION,
                current_exposure=current,
                target_exposure=target,
                requires_order=True,
                reasons=tuple(
                    reasons
                ),
            )

        if target > current:

            reasons.append(
                "Risk Governor permits restoration "
                "of the existing position."
            )

            return ExecutionDecisionV1(
                action=ExecutionAction.RESTORE_POSITION,
                current_exposure=current,
                target_exposure=target,
                requires_order=True,
                reasons=tuple(
                    reasons
                ),
            )

        reasons.append(
            "Risk Governor exposure target is "
            "unchanged."
        )

        return ExecutionDecisionV1(
            action=ExecutionAction.HOLD_POSITION,
            current_exposure=current,
            target_exposure=current,
            requires_order=False,
            reasons=tuple(
                reasons
            ),
        )

    # ---------------------------------------------------------
    # Flat position:
    # a fresh Opportunity ENTRY_CANDIDATE is required.
    # ---------------------------------------------------------

    if inputs.risk_cooldown_active:

        reasons.append(
            "Risk Governor cooldown blocks "
            "new entry."
        )

        return ExecutionDecisionV1(
            action=ExecutionAction.HOLD_FLAT,
            current_exposure=0.0,
            target_exposure=0.0,
            requires_order=False,
            reasons=tuple(
                reasons
            ),
        )

    if inputs.risk_entry_blocked:

        reasons.append(
            "Risk Governor entry gate blocks "
            "new entry."
        )

        return ExecutionDecisionV1(
            action=ExecutionAction.HOLD_FLAT,
            current_exposure=0.0,
            target_exposure=0.0,
            requires_order=False,
            reasons=tuple(
                reasons
            ),
        )

    if inputs.risk_max_exposure == 0.0:

        reasons.append(
            "Risk Governor permits no exposure."
        )

        return ExecutionDecisionV1(
            action=ExecutionAction.HOLD_FLAT,
            current_exposure=0.0,
            target_exposure=0.0,
            requires_order=False,
            reasons=tuple(
                reasons
            ),
        )

    if (
        inputs.opportunity_state
        != OpportunityState.ENTRY_CANDIDATE
    ):

        reasons.append(
            "Fresh ENTRY_CANDIDATE is required "
            "to initiate a position."
        )

        return ExecutionDecisionV1(
            action=ExecutionAction.HOLD_FLAT,
            current_exposure=0.0,
            target_exposure=0.0,
            requires_order=False,
            reasons=tuple(
                reasons
            ),
        )

    target = min(
        policy.entry_target_exposure,
        inputs.risk_max_exposure,
    )

    if target <= 0.0:
        raise AssertionError(
            "Entry target unexpectedly resolved "
            "to zero."
        )

    reasons.append(
        "Fresh ENTRY_CANDIDATE received."
    )

    reasons.append(
        "Risk Governor permits entry."
    )

    return ExecutionDecisionV1(
        action=ExecutionAction.OPEN_POSITION,
        current_exposure=0.0,
        target_exposure=float(
            target
        ),
        requires_order=True,
        reasons=tuple(
            reasons
        ),
    )


def apply_execution_fill(
    position: PositionMemoryV1,
    decision: ExecutionDecisionV1,
    execution_price: float,
) -> PositionMemoryV1:
    """
    Apply an executed next-session-open fill.

    Reductions and restorations preserve the original
    position-cycle entry anchor.
    """

    validate_position_memory(
        position
    )

    if execution_price <= 0:
        raise ValueError(
            "execution_price must be positive."
        )

    target = float(
        decision.target_exposure
    )

    validate_exposure(
        "target_exposure",
        target,
    )

    current = float(
        position.current_exposure
    )

    # Full exit clears the cycle.
    if target == 0.0:

        return PositionMemoryV1(
            current_exposure=0.0,
            entry_anchor_price=None,
        )

    # New position establishes a new cycle anchor.
    if current == 0.0:

        return PositionMemoryV1(
            current_exposure=target,
            entry_anchor_price=float(
                execution_price
            ),
        )

    # Existing position retains original anchor.
    return PositionMemoryV1(
        current_exposure=target,
        entry_anchor_price=(
            position.entry_anchor_price
        ),
    )


def calculate_drawdown_from_entry_pct(
    position: PositionMemoryV1,
    current_price: float,
) -> Optional[float]:
    """
    Risk Governor position drawdown from original
    position-cycle entry price.

    Returns None while flat.
    """

    validate_position_memory(
        position
    )

    if current_price <= 0:
        raise ValueError(
            "current_price must be positive."
        )

    if position.current_exposure == 0.0:
        return None

    return (
        (
            current_price
            / position.entry_anchor_price
        )
        - 1.0
    ) * 100.0


def transaction_cost_fraction(
    current_exposure: float,
    target_exposure: float,
    policy: ExecutionPolicyV1,
) -> float:
    """
    Transaction cost as fraction of portfolio equity.

    Example:
    5 bps × 1.0 turnover = 0.0005.
    """

    validate_exposure(
        "current_exposure",
        current_exposure,
    )

    validate_exposure(
        "target_exposure",
        target_exposure,
    )

    turnover = abs(
        target_exposure
        - current_exposure
    )

    return (
        turnover
        * policy.transaction_cost_bps_per_unit_turnover
        / 10_000.0
    )