# V4 Execution Architecture V1

## Purpose

Execution Architecture V1 defines how frozen Opportunity Engine V1
signals become positions and how Risk Governor V3 manages those
positions afterward.

This layer does not create new signals.

It does not modify:

- Risk Governor V3
- Technical Opportunity V1
- Treasury Confirmation V1
- Macro Regime V1
- Market Confirmation V1
- Opportunity Engine V1 thresholds

---

# Decision and Execution Timing

Opportunity Engine and Risk Governor decisions are calculated after
the close of market session t.

Orders generated from those decisions become effective at the next
available market session open, t+1.

Historical simulations must use the actual next-session open price.

The simulation must not substitute the decision-day closing price.

If an executable next-session open is unavailable, the trade must not
be simulated using another price without defining a new execution
version.

---

# Opportunity Engine Authority

Opportunity Engine V1 controls initiation of a new position only.

When flat:

NO_TRADE
    remain flat

WATCH
    remain flat

ARMED_DO_NOT_CHASE
    remain flat

ENTRY_CANDIDATE
    may initiate a position if Risk Governor permits entry

Once a position is open, Opportunity Engine states do not directly:

- liquidate the position
- reduce the position
- increase the position
- reset the entry price
- alter the Risk Governor

Position management belongs to Risk Governor V3.

---

# Risk Governor Authority

Risk Governor V3 has veto authority.

For a currently open position:

Risk maximum exposure = 1.00
    target exposure = 1.00

Risk maximum exposure = 0.50
    target exposure = 0.50

Risk maximum exposure = 0.00
    target exposure = 0.00

FORCE_EXIT
    target exposure = 0.00

Risk deterioration is applied immediately at the next executable
session under the daily architecture.

Risk recovery may restore a reduced existing position without requiring
a new Opportunity Engine ENTRY_CANDIDATE.

This is a restoration of an existing position cycle, not a new entry.

---

# Entry Rules

A new position requires all of the following:

1. Current position is flat.
2. Opportunity Engine state is ENTRY_CANDIDATE.
3. Risk Governor entry gate is open.
4. Risk Governor is not in cooldown.
5. Risk Governor permits positive exposure.

Target entry exposure is the lesser of:

- 1.00
- Risk Governor maximum permitted exposure

In the current architecture, Risk Governor entry blocking is expected
to prevent inappropriate new entries during elevated-risk conditions.

---

# Existing Position Rules

Once exposure is greater than zero:

WATCH does not cause an exit.

ARMED_DO_NOT_CHASE does not cause an exit.

NO_TRADE from the Opportunity Engine does not independently cause an
exit.

Only Risk Governor instructions control reductions, restorations, and
forced exits.

This prevents entry/setup indicators from becoming unresearched exit
signals.

---

# Full Exit and Re-Entry

A full exit ends the current position cycle.

After exposure reaches zero:

- the previous entry signal expires
- the previous entry-price anchor is cleared
- automatic re-entry is prohibited
- a fresh ENTRY_CANDIDATE is required

---

# Position Drawdown

Risk Governor V3 uses drawdown from the original position-cycle entry
price.

For an open position:

drawdown_from_entry_pct =
    (current_price / entry_anchor_price - 1) * 100

The entry anchor is established on the first flat-to-long execution.

Reducing exposure from 1.00 to 0.50 does not reset the anchor.

Restoring exposure from 0.50 to 1.00 does not reset the anchor.

A full exit clears the anchor.

A later new position establishes a new anchor.

---

# Transaction Costs

Execution Architecture V1 uses:

5 basis points per unit of absolute exposure turnover.

Examples:

0.00 -> 1.00
    turnover = 1.00
    cost = 5 bps

1.00 -> 0.50
    turnover = 0.50
    cost = 2.5 bps

0.50 -> 1.00
    turnover = 0.50
    cost = 2.5 bps

1.00 -> 0.00
    turnover = 1.00
    cost = 5 bps

A complete 0 -> 1 -> 0 round trip therefore costs 10 bps before any
additional market-impact assumptions.

---

# Intraday Emergency Handling

The existing daily architecture cannot prevent an adverse move that
occurs during the same session before the end-of-day risk calculation.

The separate intraday FORCE_EXIT interface remains reserved for future
implementation.

Execution Architecture V1 must not fabricate historical intraday exits
from daily data.

---

# Allowed Exposure Levels

Execution Architecture V1 supports:

- 0.00
- 0.50
- 1.00

No leverage multiplier above 1.00 is added by the execution engine.

TMF itself already provides leveraged ETF exposure.

---

# Research Governance

This execution policy must be frozen before integrated historical P&L
evaluation.

Historical P&L results must not be used to modify these execution rules
without creating Execution Architecture V2.
