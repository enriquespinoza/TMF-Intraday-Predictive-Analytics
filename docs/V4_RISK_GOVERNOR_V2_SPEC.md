# V4 Risk Governor V2 Specification

## Purpose

Risk Governor V2 preserves the market-risk thresholds introduced in
Risk Governor V1 while changing how risk signals control entries and
existing exposure.

V1 demonstrated that dynamic exposure reduction materially reduced
TMF volatility and drawdown relative to an always-long position, but
it also blocked entries on approximately 70% of historical trading
days and generated substantial exposure turnover.

A major contributor was the treatment of TMF trading below its
21-day EMA as an exposure-reduction signal.

V2 separates entry permission from exposure control.

---

# Design Changes From V1

Risk Governor V2 makes four architectural changes:

1. Entry blocking is separated from exposure state.
2. EMA21 and stale-data conditions no longer reduce existing exposure.
3. Recovery hysteresis prevents rapid defensive-state reversal.
4. An explicit interface is defined for a future intraday emergency
   layer.

The underlying market-risk thresholds are unchanged from V1.

---

# Entry Gate

The entry gate answers:

> May a new TMF position be opened?

A new entry is blocked when any of the following are true:

- Effective exposure state is not GREEN
- TMF is below its 21-day EMA
- Critical market data is stale
- The intraday emergency layer explicitly blocks new entries

These conditions do not necessarily require reduction of an existing
position.

Example:

TMF below EMA21
+
No Treasury shock
+
No volatility shock
+
No major TMF price deterioration

Result:

- Exposure state: GREEN
- Existing position: may remain at full permitted exposure
- New entry: blocked

---

# Exposure Governor

The exposure governor answers:

> How much existing TMF exposure is currently permitted?

Exposure-state inputs are limited to:

- TMF one-day return
- Drawdown from entry while a position is open
- 30-year Treasury one-day yield change
- 30-year Treasury five-day yield change
- 10-year real-yield five-day change
- Realized-volatility expansion
- Future intraday emergency signal

EMA21 position and data staleness are deliberately excluded from
automatic position reduction.

---

# Exposure States

## GREEN

Maximum permitted exposure:

1.00

Existing position:

May remain open.

## YELLOW

Maximum permitted exposure:

0.50

Existing position:

May remain open at reduced maximum exposure.

New entry:

Blocked.

## RED

Maximum permitted exposure:

0.00

Existing position:

Exit required.

New entry:

Blocked.

## EMERGENCY_EXIT

Maximum permitted exposure:

0.00

Existing position:

Immediate exit required.

New entry:

Blocked.

Cooldown:

Required.

---

# Exposure Thresholds

V2 retains the V1 thresholds unchanged.

## TMF One-Day Return

YELLOW:
-3.0% or worse

RED:
-5.0% or worse

EMERGENCY_EXIT:
-8.0% or worse

## Position Drawdown

YELLOW:
-3.0% or worse

RED:
-5.0% or worse

EMERGENCY_EXIT:
-8.0% or worse

Only evaluated while a position is open.

## 30-Year Treasury Yield — One Day

YELLOW:
+8 basis points

RED:
+15 basis points

EMERGENCY_EXIT:
+25 basis points

## 30-Year Treasury Yield — Five Days

YELLOW:
+15 basis points

RED:
+25 basis points

## 10-Year Real Yield — Five Days

YELLOW:
+10 basis points

RED:
+20 basis points

## Volatility Expansion

YELLOW:
5-day / 20-day realized volatility >= 1.35

RED:
5-day / 20-day realized volatility >= 1.80

## Multiple Warnings

Three simultaneous YELLOW exposure warnings escalate the raw exposure
state to RED.

EMA21 and stale-data entry blockers do not count toward this
three-warning exposure escalation.

---

# Recovery Hysteresis

Deterioration is immediate.

Recovery is deliberately slower.

A worsening raw state immediately becomes the effective state.

Example:

GREEN -> RED

occurs immediately when a RED condition appears.

Recovery requires two consecutive observations indicating a less
severe raw state before the effective state improves by one level.

Recovery therefore occurs stepwise:

EMERGENCY_EXIT -> RED -> YELLOW -> GREEN

and:

RED -> YELLOW -> GREEN

Each improvement requires two consecutive less-severe observations.

Example:

Day 1:
Effective RED
Raw GREEN
Recovery streak = 1
Effective remains RED

Day 2:
Effective RED
Raw GREEN
Recovery streak = 2
Effective improves to YELLOW

Day 3:
Effective YELLOW
Raw GREEN
Recovery streak = 1
Effective remains YELLOW

Day 4:
Effective YELLOW
Raw GREEN
Recovery streak = 2
Effective improves to GREEN

Any equal or worse raw state resets the recovery streak.

The value of two sessions is a V2 policy assumption rather than an
optimized historical parameter.

---

# Intraday Emergency Interface

Risk Governor V2 defines a contract for a future intraday monitoring
system.

The future intraday layer may return:

NONE

No intraday override.

BLOCK_ENTRY

Do not permit new positions, but do not automatically reduce existing
exposure.

FORCE_EXIT

Immediately produce EMERGENCY_EXIT regardless of the end-of-day
exposure signals.

Possible future intraday inputs include:

- TMF intraday percentage loss
- Intraday 30-year yield shock
- Intraday real-yield shock
- Portfolio loss limit
- Abnormal volatility
- Market-data integrity failure

The exact intraday thresholds are not part of Risk Governor V2 and
will be developed separately.

---

# No Threshold Optimization

V2 does not alter V1 market-risk thresholds based on the V1 historical
portfolio backtest.

This allows the V2 test to isolate the value of the architectural
changes:

- Separate entry gate
- Cleaner exposure signals
- Hysteresis
- Intraday override interface

If numerical thresholds are later changed, that policy must be
identified as a separate governor version.

---

# Evaluation

V2 will first be evaluated independently of the opportunity engine.

Primary comparisons against V1:

- State distribution
- Percentage of days new entries are blocked
- Mean portfolio exposure
- Exposure changes
- Total turnover
- Maximum drawdown
- Annualized volatility
- Worst day
- Return after transaction costs

Only after V2's risk behavior is evaluated will the opportunity engine
be connected to it.