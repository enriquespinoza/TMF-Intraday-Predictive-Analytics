# V4 Macro Regime V1 Specification

## Purpose

Macro Regime V1 measures whether the labor-market environment is
becoming supportive of lower interest rates and therefore potentially
supportive of long-duration Treasury exposure.

The component reuses the point-in-time PAYEMS infrastructure developed
during V3.

It does not attempt to predict whether TMF will be higher exactly five
trading days later.

Instead, it answers:

> Is the current labor-market regime relatively favorable to a
> long-duration Treasury thesis?

Risk Governor V3 retains veto authority over all opportunity signals.

---

# Inputs

Macro Regime V1 uses the frozen PAYEMS-A variables:

- payroll_change_1m
- payroll_revision_total
- payroll_broad_revision_event

These variables are already reconstructed point-in-time.

No revised future labor-market information may be substituted for the
values known at the historical decision timestamp.

---

# Macro-State Sampling

PAYEMS variables are slow-moving.

The same PAYEMS state can persist across many daily TMF observations.

Macro Regime V1 therefore does not calculate historical percentiles
using every trading day.

Instead:

1. Sort observations chronologically.
2. Detect genuine changes in the PAYEMS feature state.
3. Treat each changed PAYEMS state as one macro observation.
4. Calculate percentiles using prior macro states only.
5. Carry the resulting macro score forward until the next PAYEMS state
   change.

This prevents a monthly macro state from being counted repeatedly as
if each trading day were an independent economic observation.

---

# Historical Reference Window

Default macro-state lookback:

36 PAYEMS states

Minimum history:

12 PAYEMS states

These are design assumptions for Macro Regime V1.

They are not optimized against future TMF returns.

---

# Payroll Growth Score

Lower payroll growth is treated as more supportive of a long-duration
Treasury thesis.

The current payroll_change_1m value is compared with prior PAYEMS
states.

Score:

1 - historical percentile

Unusually weak payroll growth:

score near 1.0

Unusually strong payroll growth:

score near 0.0

---

# Payroll Revision Score

More-negative payroll revisions are treated as more supportive of a
long-duration Treasury thesis.

The current payroll_revision_total is compared with prior PAYEMS
states.

Score:

1 - historical percentile

Large downward revisions:

score near 1.0

Large upward revisions:

score near 0.0

---

# Broad Revision Score

payroll_broad_revision_event identifies a broad revision event but is
not directional by itself.

Direction is therefore determined using payroll_revision_total.

Broad revision event with negative total revision:

1.00

No broad revision event:

0.50

Broad revision event with positive total revision:

0.00

A zero revision receives a neutral score of 0.50.

---

# Macro Regime Score

Three equally weighted subscores are averaged:

1. Payroll growth score
2. Payroll revision score
3. Broad revision score

Final Macro Regime score:

0 to 100

No BUY, WATCH, ARMED, or ENTRY threshold is defined at this stage.

---

# Relationship to Prior V3 Research

PAYEMS-A survived historical development validation but failed to
improve the final locked V3 directional classifier in 2026.

Macro Regime V1 does not reinterpret that failed V3 result.

The objective is different.

V3 asked whether PAYEMS improved a fixed-horizon directional
classifier.

Macro Regime V1 asks whether labor-market information is useful as
slow-moving economic context inside a broader trading system.

---

# Evaluation

Macro Regime V1 will be evaluated independently before integration
with the other Opportunity Engine components.

Evaluation will include:

- Score coverage
- Score distribution
- V3-entry-open observations
- Five-day return distribution
- Five-day positive rate
- Maximum adverse excursion
- Maximum favorable excursion
- Daily score quintiles
- Macro-state-level score quintiles

The state-level evaluation is important because multiple trading days
can share the same PAYEMS state.

---

# Research Governance

Macro Regime V1 must be frozen before target-based evaluation.

Historical results must not be used to alter weights or thresholds
without defining a new macro-regime version.