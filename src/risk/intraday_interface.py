"""
Interface between the V4 end-of-day risk governor and the future
intraday emergency-monitoring layer.

The intraday system is not implemented here.

This module only defines the contract that an intraday monitor must
return to Risk Governor V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntradayAction(str, Enum):
    """Actions available to the future intraday risk layer."""

    NONE = "NONE"

    BLOCK_ENTRY = "BLOCK_ENTRY"

    FORCE_EXIT = "FORCE_EXIT"


@dataclass(frozen=True)
class IntradayEmergencySignal:
    """
    Output contract for the future intraday emergency layer.

    action:
        NONE
            No intraday override.

        BLOCK_ENTRY
            Prevent a new position but do not automatically reduce
            an existing position.

        FORCE_EXIT
            Immediately force Risk Governor V2 into
            EMERGENCY_EXIT.

    reasons:
        Human-readable explanations for the signal.
    """

    action: IntradayAction = (
        IntradayAction.NONE
    )

    reasons: tuple[str, ...] = ()