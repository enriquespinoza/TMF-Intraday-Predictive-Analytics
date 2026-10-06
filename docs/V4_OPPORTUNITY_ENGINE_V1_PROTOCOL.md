# V4 Opportunity Engine V1 Protocol

## Purpose

The Opportunity Engine identifies attractive TMF entry environments.

It does not control existing-position risk.

Risk Governor V3 remains the independent capital-preservation layer
and has veto authority over all opportunity signals.

The Opportunity Engine is evaluated only when the Risk Governor entry
gate is open.

---

# Core Architecture

Risk Governor V3
    |
    |-- Entry blocked
    |       -> NO ENTRY
    |
    `-- Entry open
            |
            v
    Opportunity Engine V1
            |
            |-- Technical Opportunity
            |-- Treasury Confirmation
            |-- Macro Regime
            `-- Market Confirmation

The Opportunity Engine will eventually generate an aggregate
opportunity assessment.

V1 components are developed and tested independently before being
combined.

---

# Development Sequence

1. Technical Opportunity
2. Treasury Confirmation
3. Macro Regime
4. Market Confirmation
5. Composite Opportunity Score
6. Trading-state integration
7. Position-sizing integration
8. Historical backtest
9. Paper forward test

---

# Design Principles

## No Lookahead

Every signal must use only information available at the decision
timestamp.

Future returns may be used only for later evaluation.

## Relative Rather Than Fixed Technical Thresholds

TMF is a leveraged duration ETF whose volatility changes materially
across market regimes.

Technical Opportunity V1 therefore uses trailing historical
percentiles instead of fixed values such as:

RSI < 30

or

5-day return < -5%

The current observation is compared only with observations available
before the current market date.

## Risk and Opportunity Remain Separate

A high opportunity score cannot override:

- RED risk
- EMERGENCY_EXIT
- entry_blocked=True
- future intraday FORCE_EXIT

The Risk Governor always has veto authority.

---

# Technical Opportunity V1

Technical Opportunity V1 measures whether TMF is trading in an
unusually depressed or pullback-like condition relative to its own
recent history.

Inputs:

- return_5d
- rsi_14
- distance_from_20d_high
- return_1d
- below_ema21_streak

The first three variables measure pullback depth.

return_1d measures whether very short-term price action has begun to
stabilize or reverse.

below_ema21_streak provides pullback context.

---

# Point-in-Time Percentiles

Percentiles are calculated using prior observations only.

Default lookback:

252 trading sessions

Minimum history:

63 trading sessions

The current observation is never included in its own historical
reference window.

No future data enter the percentile calculation.

---

# Technical Subscores

Each subscore ranges from 0 to 1.

## Five-Day Pullback

Lower historical percentile of return_5d produces a higher
opportunity score.

A highly negative 5-day return relative to prior history is treated
as a stronger pullback.

## RSI

Lower historical percentile of RSI14 produces a higher opportunity
score.

## Distance From 20-Day High

A more negative distance from the 20-day high relative to prior
history produces a higher opportunity score.

## One-Day Stabilization

Higher historical percentile of return_1d produces a higher score.

This does not attempt to predict returns.

It measures whether current-day price action is stronger than recent
history after the pullback setup.

## EMA21 Pullback Context

Research Governor V3 restored entry eligibility for 1-to-3-session
EMA21 breaks.

Technical Opportunity V1 assigns:

Above EMA21:
0.50

1-to-3 sessions below EMA21:
1.00

4 or more sessions below EMA21:
0.00

The 4-session boundary is research-derived and therefore not
independently validated.

Risk Governor V3 normally blocks entry when the streak reaches four
sessions, so this component should not be interpreted as permission
to override the governor.

---

# Technical Opportunity Score

Five equally weighted technical subscores are averaged.

Technical Opportunity Score:

0 to 100

No BUY threshold is defined at this stage.

The initial objective is to study:

- score distribution
- stability through time
- behavior conditional on Risk Governor V3 being open
- subsequent return distributions
- maximum adverse excursion
- maximum favorable excursion

Thresholds for WATCH, ARMED, or ENTRY CANDIDATE will not be defined
until the remaining Opportunity Engine components exist.

---

# Research Governance

Technical Opportunity V1 is frozen before target-based historical
evaluation.

Changes after historical evaluation require a new version.

The purpose of the first evaluation is descriptive research, not
threshold optimization.