# V4 Market Confirmation V1 Specification

## Purpose

Market Confirmation V1 measures whether TMF price action is beginning
to respond in a direction consistent with a potential long-duration
Treasury entry.

It answers:

> Is the market itself confirming the opportunity?

This component is separate from:

- Technical Opportunity V1
- Treasury Confirmation V1
- Macro Regime V1
- Risk Governor V3

Risk Governor V3 retains veto authority over all entry decisions.

---

# Role in the Trading Architecture

Technical Opportunity:
Identifies whether TMF is in an attractive pullback/setup condition.

Treasury Confirmation:
Measures whether Treasury yield behavior is supportive.

Macro Regime:
Measures whether the economic backdrop is supportive of duration risk.

Market Confirmation:
Measures whether TMF price action is actually beginning to respond.

---

# Inputs

Market Confirmation V1 uses:

- return_3d
- close_vs_ema9
- ema_spread_pct
- relative_volume
- return_1d

return_1d is used only to determine the direction of current-day
volume participation.

It is not scored independently because Technical Opportunity V1
already contains a one-day stabilization component.

---

# Historical Reference Window

Default lookback:

252 trading sessions

Minimum history:

63 trading sessions

All percentiles use prior observations only.

The current observation is excluded from its own historical reference
window.

---

# Three-Day Price Response

return_3d measures recent short-term price response.

Higher historical percentile:

stronger confirmation

Score:

historical percentile of return_3d

---

# EMA9 Price Confirmation

close_vs_ema9 measures TMF price relative to its 9-day exponential
moving average.

Higher historical percentile:

stronger price confirmation

Score:

historical percentile of close_vs_ema9

---

# EMA Trend Structure

ema_spread_pct measures the relationship between the 9-day and 21-day
exponential moving averages.

Higher historical percentile:

stronger short-term trend structure

Score:

historical percentile of ema_spread_pct

---

# Signed Volume Participation

Relative volume by itself does not provide directional information.

Market Confirmation V1 therefore defines:

signed_relative_volume =
    sign(return_1d) * relative_volume

Interpretation:

High-volume positive session:
strong positive confirmation

High-volume negative session:
strong negative confirmation

Low-volume session:
weaker directional confirmation

The signed relative-volume observation is compared with its prior
historical distribution.

Higher historical percentile:

stronger confirmation

---

# Market Confirmation Score

Four equally weighted subscores are averaged:

1. Three-day price-response score
2. EMA9 price-confirmation score
3. EMA trend-structure score
4. Signed-volume participation score

Final score:

0 to 100

No BUY, WATCH, ARMED, or ENTRY threshold is defined in V1.

---

# No Lookahead

All signal values must be available at the current end-of-day
decision timestamp.

Future returns are used only during later evaluation.

Market Confirmation V1 is currently an end-of-day component.

A future intraday confirmation layer will be treated separately.

---

# Research Governance

Market Confirmation V1 must be frozen before target-based historical
evaluation.

Historical evaluation may describe:

- score distribution
- forward returns
- positive-return frequency
- MAE
- MFE
- quintile behavior

Historical results must not be used to modify weights or thresholds
without defining a new Market Confirmation version.