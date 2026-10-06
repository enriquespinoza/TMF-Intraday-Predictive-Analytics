"""
Integrated V4 decision bridge.

Connects:

Risk Governor V3
    ->
Opportunity Engine V1
    ->
Execution Architecture V1

The bridge evaluates information available at the CLOSE of session t.

The resulting execution decision is intended for execution at the
OPEN of session t+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.execution.position_lifecycle import (
    ExecutionDecisionV1,
    ExecutionInputsV1,
    ExecutionPolicyV1,
    PositionMemoryV1,
    calculate_drawdown_from_entry_pct,
    evaluate_execution_v1,
)

from src.risk.risk_governor_v3 import (
    GovernorMemory,
    IntradayAction,
    IntradayEmergencySignal,
    RiskDecisionV3,
    RiskInputsV3,
    evaluate_risk_v3,
)

from src.signals.opportunity_state_machine import (
    OpportunityDecisionV1,
    OpportunityInputsV1,
    OpportunityPolicyV1,
    evaluate_opportunity_v1,
)


@dataclass(frozen=True)
class IntegratedCloseDecisionV1:
    """Complete end-of-day integrated decision."""

    drawdown_from_entry_pct: Optional[float]

    risk_decision: RiskDecisionV3

    opportunity_decision: OpportunityDecisionV1

    execution_decision: ExecutionDecisionV1

    updated_governor_memory: GovernorMemory


def parse_bool_value(
    value,
) -> bool:
    """
    Parse a bool safely from DataFrame/CSV values.

    Never use bool("False"), because that evaluates to True.
    """

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if pd.isna(
        value
    ):
        raise ValueError(
            "Required boolean value is missing."
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        if value == 0:
            return False

        if value == 1:
            return True

    text = (
        str(
            value
        )
        .strip()
        .lower()
    )

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    raise ValueError(
        f"Unable to parse boolean value: {value!r}"
    )


def required_float(
    row: pd.Series,
    column: str,
) -> float:
    """Read a required numeric field."""

    if column not in row.index:
        raise ValueError(
            f"Required column missing: {column}"
        )

    value = pd.to_numeric(
        pd.Series(
            [
                row[
                    column
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(
        value
    ):
        raise ValueError(
            f"Required numeric value missing: {column}"
        )

    return float(
        value
    )


def optional_float(
    row: pd.Series,
    column: str,
) -> Optional[float]:
    """Read an optional numeric field."""

    if column not in row.index:
        return None

    value = pd.to_numeric(
        pd.Series(
            [
                row[
                    column
                ]
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(
        value
    ):
        return None

    return float(
        value
    )


def build_risk_inputs_v1(
    row: pd.Series,
    position: PositionMemoryV1,
) -> tuple[
    RiskInputsV3,
    Optional[float],
]:
    """
    Build position-aware Risk Governor V3 inputs.

    Portfolio-dependent historical V3 columns are intentionally
    ignored.
    """

    position_open = (
        position.current_exposure > 0.0
    )

    if position_open:

        close_price = required_float(
            row,
            "close",
        )

        drawdown = (
            calculate_drawdown_from_entry_pct(
                position=position,
                current_price=close_price,
            )
        )

    else:
        drawdown = None

    inputs = RiskInputsV3(
        tmf_return_1d_pct=required_float(
            row,
            "tmf_return_1d_pct",
        ),

        yield_30y_change_1d_bp=required_float(
            row,
            "change_30y_1d_bp",
        ),

        yield_30y_change_5d_bp=required_float(
            row,
            "change_30y_5d_bp",
        ),

        real_yield_10y_change_5d_bp=required_float(
            row,
            "real_yield_10y_change_5d_bp",
        ),

        volatility_ratio_5d_20d=required_float(
            row,
            "volatility_ratio_5d_20d",
        ),

        below_ema21=parse_bool_value(
            row[
                "below_ema21"
            ]
        ),

        below_ema21_streak=int(
            required_float(
                row,
                "below_ema21_streak",
            )
        ),

        critical_data_stale=parse_bool_value(
            row[
                "critical_data_stale"
            ]
        ),

        position_open=position_open,

        drawdown_from_entry_pct=drawdown,
    )

    return (
        inputs,
        drawdown,
    )


def evaluate_integrated_close_v1(
    row: pd.Series,
    position: PositionMemoryV1,
    governor_memory: GovernorMemory,
    opportunity_policy: OpportunityPolicyV1,
    execution_policy: ExecutionPolicyV1,
    intraday_signal: Optional[
        IntradayEmergencySignal
    ] = None,
) -> IntegratedCloseDecisionV1:
    """
    Evaluate the full frozen V1 architecture at session close.

    No execution occurs here.

    The returned ExecutionDecisionV1 must be queued for the next
    market-session open.
    """

    if intraday_signal is None:
        intraday_signal = (
            IntradayEmergencySignal()
        )

    (
        risk_inputs,
        drawdown,
    ) = build_risk_inputs_v1(
        row=row,
        position=position,
    )

    (
        risk_decision,
        updated_governor_memory,
    ) = evaluate_risk_v3(
        inputs=risk_inputs,
        memory=governor_memory,
        intraday_signal=intraday_signal,
    )

    opportunity_inputs = OpportunityInputsV1(
        risk_entry_blocked=(
            risk_decision.entry_blocked
        ),

        technical_opportunity_score=(
            optional_float(
                row,
                "technical_opportunity_score",
            )
        ),

        treasury_confirmation_score=(
            optional_float(
                row,
                "treasury_confirmation_score",
            )
        ),

        macro_regime_score=(
            optional_float(
                row,
                "macro_regime_score",
            )
        ),

        market_confirmation_score=(
            optional_float(
                row,
                "market_confirmation_score",
            )
        ),
    )

    opportunity_decision = (
        evaluate_opportunity_v1(
            inputs=opportunity_inputs,
            policy=opportunity_policy,
        )
    )

    force_exit = (
        intraday_signal.action
        == IntradayAction.FORCE_EXIT
    )

    execution_inputs = ExecutionInputsV1(
        opportunity_state=(
            opportunity_decision.state
        ),

        risk_entry_blocked=(
            risk_decision.entry_blocked
        ),

        risk_max_exposure=float(
            risk_decision.max_position_multiplier
        ),

        risk_force_exit=force_exit,

        risk_cooldown_active=(
            risk_decision.cooldown_required
        ),
    )

    execution_decision = (
        evaluate_execution_v1(
            inputs=execution_inputs,
            position=position,
            policy=execution_policy,
        )
    )

    return IntegratedCloseDecisionV1(
        drawdown_from_entry_pct=drawdown,

        risk_decision=risk_decision,

        opportunity_decision=(
            opportunity_decision
        ),

        execution_decision=(
            execution_decision
        ),

        updated_governor_memory=(
            updated_governor_memory
        ),
    )