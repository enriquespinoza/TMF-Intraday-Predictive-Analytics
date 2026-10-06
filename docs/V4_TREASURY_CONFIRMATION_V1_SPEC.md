# V4 Treasury Confirmation V1 Specification

## Purpose

Treasury Confirmation V1 measures whether the Treasury market is
currently moving in a direction consistent with a favorable TMF
entry.

TMF is a leveraged long-duration Treasury ETF.

Falling long-duration Treasury yields generally support TMF prices,
while rising long-duration yields generally create headwinds.

This component does not control risk.

Risk Governor V3 retains veto authority.

---

# Inputs

Treasury Confirmation V1 uses:

- 30-year Treasury one-day yield change
- 30-year Treasury five-day yield change
- 10-year real-yield five-day change

All yield changes are measured in basis points.

---

# Point-in-Time Percentiles

Each current yield-change observation is compared only with prior
observations.

Default lookback:

252 trading sessions

Minimum history:

63 trading sessions

The current observation is excluded from its own reference sample.

---

# Scoring Direction

For all three inputs:

Lower yield change
=
stronger TMF confirmation

Therefore:

Treasury subscore
=
1 - historical percentile

Examples:

Unusually large yield decline:
score near 1.0

Typical yield movement:
score near 0.5

Unusually large yield increase:
score near 0.0

---

# Treasury Confirmation Score

Three equally weighted subscores are averaged:

1. 30Y one-day confirmation
2. 30Y five-day confirmation
3. 10Y real-yield five-day confirmation

Final score:

0 to 100

No entry threshold is defined in Treasury Confirmation V1.

---

# Separation From Risk Governor

The same underlying Treasury market may appear in both systems, but
the systems answer different questions.

Risk Governor:

Is the yield move dangerous enough to reduce or exit exposure?

Treasury Confirmation:

Is the direction of the yield move supportive of a potential entry?

A high Treasury Confirmation score can never override a defensive
Risk Governor state.

---

# Research Governance

Treasury Confirmation V1 is frozen before forward-return evaluation.

Historical evaluation is descriptive.

No threshold or weighting changes may be made solely to improve
historical quintile performance without creating a new version.