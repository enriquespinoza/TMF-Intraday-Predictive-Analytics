# V4 TMF Trading Decision Engine Protocol

## Purpose

V4 changes the objective of the TMF research project.

V1-V3 primarily investigated whether technical and macroeconomic
features could predict the direction of TMF over a fixed five-day
horizon.

V4 is designed as a trading decision system.

Prediction remains useful, but prediction is only one component of
the system.

The V4 architecture separates:

1. Opportunity identification
2. Market confirmation
3. Position entry
4. Real-time risk monitoring
5. Position reduction and exit
6. Performance measurement

The risk governor has veto authority over all entry signals.

---

# Core Principle

The opportunity engine answers:

> Is this an attractive environment in which to take TMF risk?

The risk governor answers:

> Is it currently safe to have capital exposed to TMF?

If those systems disagree, the risk governor wins.

A strong opportunity score can never override a RED or
EMERGENCY_EXIT risk state.

---

# V4 Development Sequence

V4 will be developed in the following order:

Risk Governor
    ↓
Risk Governor Backtest
    ↓
Opportunity Engine
    ↓
Confirmation Engine
    ↓
Trading State Machine
    ↓
Position Sizing
    ↓
Integrated Backtest
    ↓
Paper Forward Test

The risk governor is intentionally developed before the entry model.

---

# Risk Governor

The risk governor is independent from the predictive model.

Its purpose is capital preservation rather than return prediction.

The initial governor is deterministic and rule-based.

No machine-learning model is used in Risk Governor V1.

This makes every state transition explainable and auditable.

---

# Risk States

## GREEN

Market risk conditions are normal.

Actions:

- New entries permitted
- Existing position may remain open
- Maximum permitted risk allocation: 100% of the allocation
  authorized by the future position-sizing engine

GREEN does not mean TMF should be purchased.

It only means the risk governor is not blocking an otherwise valid
entry.

---

## YELLOW

Meaningful risk deterioration is present but emergency conditions
have not been reached.

Actions:

- New entries prohibited
- Existing position may remain open
- Position-sizing engine may reduce exposure
- Maximum permitted risk allocation: 50%

YELLOW acts as a defensive state.

---

## RED

Major market deterioration is present.

Actions:

- New entries prohibited
- Existing long position must exit
- Maximum permitted allocation: 0%

RED has veto authority over the entry engine.

---

## EMERGENCY_EXIT

Extreme market or position deterioration is present.

Actions:

- New entries prohibited
- Existing position must exit
- Maximum permitted allocation: 0%
- Trading state machine enters cooldown

This state is reserved for extreme adverse conditions.

---

# Risk Inputs

Risk Governor V1 uses fast-moving market information rather than
slow macroeconomic releases.

## TMF Price Risk

- One-day TMF return
- Drawdown from trade entry
- TMF position relative to the 21-day EMA

## Treasury Rate Risk

- One-day change in the 30-year Treasury yield
- Five-day change in the 30-year Treasury yield
- Five-day change in the 10-year real Treasury yield

Because TMF is a leveraged long-duration Treasury ETF, rapidly rising
long-term and real yields are treated as direct threats to the trade.

## Volatility Risk

- Five-day realized volatility relative to twenty-day realized
  volatility

The governor is concerned primarily with volatility expansion rather
than high volatility in isolation.

## Data Integrity Risk

- Critical real-time data stale or unavailable

The system must not initiate a new position when the inputs required
for risk evaluation are stale.

Data staleness alone does not automatically force an existing
position to liquidate.

---

# Initial Research Thresholds

The first threshold set is deliberately simple and conservative.

These values are research defaults.

They are not claimed to be optimal trading thresholds.

They must be evaluated as a complete policy rather than individually
adjusted to improve isolated historical trades.

## TMF One-Day Return

YELLOW:
-3.0% or worse

RED:
-5.0% or worse

EMERGENCY_EXIT:
-8.0% or worse

## Drawdown From Entry

YELLOW:
-3.0% or worse

RED:
-5.0% or worse

EMERGENCY_EXIT:
-8.0% or worse

This condition is evaluated only while a position is open.

## 30-Year Treasury Yield — One Day

YELLOW:
+8 basis points or more

RED:
+15 basis points or more

EMERGENCY_EXIT:
+25 basis points or more

## 30-Year Treasury Yield — Five Days

YELLOW:
+15 basis points or more

RED:
+25 basis points or more

## 10-Year Real Yield — Five Days

YELLOW:
+10 basis points or more

RED:
+20 basis points or more

## Volatility Expansion

YELLOW:
5-day / 20-day realized-volatility ratio >= 1.35

RED:
5-day / 20-day realized-volatility ratio >= 1.80

## Technical Breakdown

TMF closing below its 21-day EMA contributes one YELLOW warning.

It does not independently trigger RED.

## Data Integrity

Critical data staleness contributes one YELLOW warning and prevents
new entries.

---

# Multi-Signal Escalation

Individual warning conditions are not treated independently.

If three or more YELLOW conditions occur simultaneously, the governor
escalates the state to RED.

This allows several moderate deteriorations to produce the same
response as one major deterioration.

Any RED condition produces RED immediately.

Any EMERGENCY condition produces EMERGENCY_EXIT immediately.

Priority:

EMERGENCY_EXIT
    >
RED
    >
YELLOW
    >
GREEN

---

# Entry Permission

The risk governor controls whether the future opportunity engine may
open a position.

GREEN:
entry permitted

YELLOW:
entry prohibited

RED:
entry prohibited

EMERGENCY_EXIT:
entry prohibited

An opportunity score can never override this permission.

---

# Position Exposure

Initial governor permissions:

GREEN:
1.00 maximum position multiplier

YELLOW:
0.50 maximum position multiplier

RED:
0.00

EMERGENCY_EXIT:
0.00

These are maximum permissions.

The eventual position-sizing engine may always choose a smaller
allocation.

---

# Separation From Opportunity Engine

The following variables may eventually contribute to the opportunity
engine but do not currently control Risk Governor V1:

- PAYEMS macro state
- CPI
- Retail sales
- Industrial production
- Five-day return prediction
- V3 probability
- Future opportunity score

A bullish macro thesis cannot suppress a risk warning.

This separation is intentional.

---

# V3 Research

V3 remains a completed experiment.

V3 results must not be rewritten or reclassified based on V4
performance.

Data and infrastructure created during V3 may be reused by V4, but
V4 represents a separate research protocol.

---

# V4 Evaluation

The risk governor will eventually be evaluated using trading metrics
rather than classification metrics.

Important metrics include:

- Maximum drawdown
- Maximum adverse excursion
- Average adverse excursion
- Return
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Profit factor
- Average winner
- Average loser
- Win rate
- Expected value per trade
- Time in market
- Turnover
- Number of forced exits
- Number of false defensive exits
- Transaction costs
- Slippage

The primary purpose of the governor is not to maximize total return.

Its purpose is to reduce severe adverse outcomes while preserving
enough participation for the opportunity engine to generate positive
expectancy.

---

# Threshold Governance

Risk Governor V1 thresholds are frozen before integrated trading
performance is evaluated.

Changes to thresholds must create a new governor version.

Examples:

Risk Governor V1
Risk Governor V2

A threshold must not be changed simply because it would have avoided
one losing historical trade.

All modifications must have an economic or risk-management rationale
and must be evaluated as a complete new policy.

---

# Forward Testing

After the integrated V4 policy is frozen, future recommendations will
be logged before their outcomes are known.

Each decision record should eventually include:

- Timestamp
- TMF price
- Risk state
- Risk reasons
- Opportunity score
- Trading state
- Recommended action
- Position size
- Treasury yields
- Volatility state
- Entry price
- Exit price
- Trade P&L
- Maximum adverse excursion
- Maximum favorable excursion

The objective is to create an auditable record of what the trading
system would actually have done in real time.