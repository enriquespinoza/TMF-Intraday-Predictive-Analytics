"""
Opportunity Engine V1 state machine.

This module converts the frozen Opportunity Engine V1 component
scores and Risk Governor permission into an actionable research state.

Decision authority:

Risk Governor V3:
    veto authority

Technical Opportunity V1:
    setup qualification

Treasury Confirmation V1:
    extension / do-not-chase check

Macro Regime V1:
    informational only

Market Confirmation V1:
    informational only

This is a candidate architecture for forward testing.
It does not establish a profitable trading rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Optional


DEFAULT_POLICY_FILE = Path(
    "config/"
    "opportunity_engine_v1_policy.json"
)


class OpportunityState(str, Enum):
    """Opportunity Engine V1 states."""

    NO_TRADE = "NO_TRADE"

    WATCH = "WATCH"

    ENTRY_CANDIDATE = "ENTRY_CANDIDATE"

    ARMED_DO_NOT_CHASE = (
        "ARMED_DO_NOT_CHASE"
    )


@dataclass(frozen=True)
class OpportunityPolicyV1:
    """Frozen Opportunity Engine V1 thresholds."""

    technical_strong_setup_threshold: float

    treasury_extreme_confirmation_threshold: float


@dataclass(frozen=True)
class OpportunityInputsV1:
    """
    Inputs consumed by Opportunity Engine V1.

    Macro and Market Confirmation are intentionally informational.
    """

    risk_entry_blocked: bool

    technical_opportunity_score: Optional[
        float
    ]

    treasury_confirmation_score: Optional[
        float
    ]

    macro_regime_score: Optional[
        float
    ] = None

    market_confirmation_score: Optional[
        float
    ] = None


@dataclass(frozen=True)
class OpportunityDecisionV1:
    """Complete Opportunity Engine V1 decision."""

    state: OpportunityState

    entry_candidate: bool

    risk_entry_blocked: bool

    technical_strong_setup: bool

    treasury_extreme_confirmation: bool

    technical_opportunity_score: Optional[
        float
    ]

    treasury_confirmation_score: Optional[
        float
    ]

    macro_regime_score: Optional[
        float
    ]

    market_confirmation_score: Optional[
        float
    ]

    reasons: tuple[str, ...]


def load_policy_v1(
    path: Path = DEFAULT_POLICY_FILE,
) -> OpportunityPolicyV1:
    """Load frozen Opportunity Engine V1 policy."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        policy = json.load(
            file
        )

    return OpportunityPolicyV1(
        technical_strong_setup_threshold=float(
            policy[
                "technical"
            ][
                "strong_setup_threshold"
            ]
        ),

        treasury_extreme_confirmation_threshold=float(
            policy[
                "treasury"
            ][
                "extreme_confirmation_threshold"
            ]
        ),
    )


def validate_score(
    name: str,
    value: Optional[float],
) -> None:
    """Validate optional 0-to-100 component score."""

    if value is None:
        return

    if not (
        0.0
        <= value
        <= 100.0
    ):
        raise ValueError(
            f"{name} must be between "
            "0 and 100."
        )


def validate_inputs(
    inputs: OpportunityInputsV1,
) -> None:
    """Validate Opportunity Engine inputs."""

    validate_score(
        "technical_opportunity_score",
        inputs.technical_opportunity_score,
    )

    validate_score(
        "treasury_confirmation_score",
        inputs.treasury_confirmation_score,
    )

    validate_score(
        "macro_regime_score",
        inputs.macro_regime_score,
    )

    validate_score(
        "market_confirmation_score",
        inputs.market_confirmation_score,
    )


def evaluate_opportunity_v1(
    inputs: OpportunityInputsV1,
    policy: OpportunityPolicyV1,
) -> OpportunityDecisionV1:
    """
    Evaluate Opportunity Engine V1.

    Decision precedence:

    1. Missing required score -> NO_TRADE
    2. Risk Governor blocked -> NO_TRADE
    3. Technical setup not strong -> WATCH
    4. Strong technical + extreme Treasury -> ARMED_DO_NOT_CHASE
    5. Strong technical + non-extreme Treasury -> ENTRY_CANDIDATE

    Macro and Market Confirmation never alter state.
    """

    validate_inputs(
        inputs
    )

    reasons: list[str] = []

    technical_score = (
        inputs.technical_opportunity_score
    )

    treasury_score = (
        inputs.treasury_confirmation_score
    )

    # Required opportunity inputs must exist.
    if (
        technical_score is None
        or treasury_score is None
    ):
        if technical_score is None:
            reasons.append(
                "Technical Opportunity score "
                "unavailable."
            )

        if treasury_score is None:
            reasons.append(
                "Treasury Confirmation score "
                "unavailable."
            )

        return OpportunityDecisionV1(
            state=(
                OpportunityState.NO_TRADE
            ),

            entry_candidate=False,

            risk_entry_blocked=(
                inputs.risk_entry_blocked
            ),

            technical_strong_setup=False,

            treasury_extreme_confirmation=False,

            technical_opportunity_score=(
                technical_score
            ),

            treasury_confirmation_score=(
                treasury_score
            ),

            macro_regime_score=(
                inputs.macro_regime_score
            ),

            market_confirmation_score=(
                inputs.market_confirmation_score
            ),

            reasons=tuple(
                reasons
            ),
        )

    technical_strong = (
        technical_score
        >= policy.technical_strong_setup_threshold
    )

    treasury_extreme = (
        treasury_score
        >= policy.treasury_extreme_confirmation_threshold
    )

    # Risk Governor has highest authority.
    if inputs.risk_entry_blocked:

        reasons.append(
            "Risk Governor V3 blocks entry."
        )

        return OpportunityDecisionV1(
            state=(
                OpportunityState.NO_TRADE
            ),

            entry_candidate=False,

            risk_entry_blocked=True,

            technical_strong_setup=(
                technical_strong
            ),

            treasury_extreme_confirmation=(
                treasury_extreme
            ),

            technical_opportunity_score=(
                technical_score
            ),

            treasury_confirmation_score=(
                treasury_score
            ),

            macro_regime_score=(
                inputs.macro_regime_score
            ),

            market_confirmation_score=(
                inputs.market_confirmation_score
            ),

            reasons=tuple(
                reasons
            ),
        )

    # Risk is open, but setup is not strong enough.
    if not technical_strong:

        reasons.append(
            "Technical Opportunity score is "
            "below the frozen strong-setup "
            "threshold."
        )

        return OpportunityDecisionV1(
            state=(
                OpportunityState.WATCH
            ),

            entry_candidate=False,

            risk_entry_blocked=False,

            technical_strong_setup=False,

            treasury_extreme_confirmation=(
                treasury_extreme
            ),

            technical_opportunity_score=(
                technical_score
            ),

            treasury_confirmation_score=(
                treasury_score
            ),

            macro_regime_score=(
                inputs.macro_regime_score
            ),

            market_confirmation_score=(
                inputs.market_confirmation_score
            ),

            reasons=tuple(
                reasons
            ),
        )

    # Strong setup but Treasury confirmation is extreme.
    if treasury_extreme:

        reasons.append(
            "Technical setup is strong."
        )

        reasons.append(
            "Treasury Confirmation is above "
            "the frozen extreme-confirmation "
            "threshold."
        )

        reasons.append(
            "Remain armed but do not chase "
            "the extended move."
        )

        return OpportunityDecisionV1(
            state=(
                OpportunityState.ARMED_DO_NOT_CHASE
            ),

            entry_candidate=False,

            risk_entry_blocked=False,

            technical_strong_setup=True,

            treasury_extreme_confirmation=True,

            technical_opportunity_score=(
                technical_score
            ),

            treasury_confirmation_score=(
                treasury_score
            ),

            macro_regime_score=(
                inputs.macro_regime_score
            ),

            market_confirmation_score=(
                inputs.market_confirmation_score
            ),

            reasons=tuple(
                reasons
            ),
        )

    # Risk open + strong technical setup + non-extreme Treasury.
    reasons.append(
        "Risk Governor V3 permits entry."
    )

    reasons.append(
        "Technical Opportunity exceeds the "
        "frozen strong-setup threshold."
    )

    reasons.append(
        "Treasury Confirmation is below the "
        "frozen extreme-confirmation threshold."
    )

    return OpportunityDecisionV1(
        state=(
            OpportunityState.ENTRY_CANDIDATE
        ),

        entry_candidate=True,

        risk_entry_blocked=False,

        technical_strong_setup=True,

        treasury_extreme_confirmation=False,

        technical_opportunity_score=(
            technical_score
        ),

        treasury_confirmation_score=(
            treasury_score
        ),

        macro_regime_score=(
            inputs.macro_regime_score
        ),

        market_confirmation_score=(
            inputs.market_confirmation_score
        ),

        reasons=tuple(
            reasons
        ),
    )