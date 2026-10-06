# V4 Opportunity Engine V1 — Temporal Robustness Protocol

## Purpose

This diagnostic tests whether relationships identified in the
Opportunity Engine V1 interaction analysis persist through time.

It does not:

- fit a predictive model
- optimize thresholds
- modify component definitions
- create a composite score
- define entry rules

The relationships being tested were discovered in prior historical
analysis.

Therefore, this is a stability diagnostic, not independent
out-of-sample validation.

---

# Common Sample

Only observations satisfying all of the following are included:

- Risk Governor V3 entry gate open
- Technical Opportunity V1 score available
- Treasury Confirmation V1 score available
- Macro Regime V1 score available
- Market Confirmation V1 score available
- valid five-day forward return

Existing quartile classifications from the frozen interaction
diagnostic are reused.

Quartiles are NOT recalculated separately inside each time period.

This preserves the meaning of Q1/Q2/Q3/Q4 across time.

---

# Temporal Partitions

Two time-partition schemes are used.

## Calendar Periods

- 2022-2023
- 2024
- 2025
- 2026 YTD

2022 is combined with 2023 because the common sample begins late in
2022.

## Equal-Count Chronological Blocks

The complete common sample is divided chronologically into four
approximately equal-sized blocks:

- BLOCK_1
- BLOCK_2
- BLOCK_3
- BLOCK_4

These provide an additional check that conclusions are not dominated
by one calendar-year sample.

---

# Period Baseline

Each time period receives its own baseline.

Interaction cells are compared with the baseline performance from the
same period.

This is necessary because the underlying TMF return environment can
differ substantially across time.

Metrics:

- mean 1-day forward return
- mean 5-day forward return
- median 5-day forward return
- five-day positive rate
- mean five-day MAE
- median five-day MAE
- mean five-day MFE
- median five-day MFE

---

# Pre-Specified Relationships

## Hypothesis 1 — Technical Q4 Setup Quality

Previous analysis suggested Technical Opportunity Q4 had better
five-day return and MAE than the common-sample baseline.

Temporal test:

Technical Q4 should continue to show:

- positive return lift versus its period baseline
- positive MAE lift versus its period baseline

Positive MAE lift means less-negative adverse excursion.

---

## Hypothesis 2 — Treasury Q2 Versus Treasury Q4

Previous analysis suggested moderate Treasury Confirmation performed
better than extreme Treasury Confirmation.

Temporal test:

Treasury Q2 should outperform Treasury Q4 on:

- five-day return
- MAE

This tests the possibility that extreme Treasury confirmation is an
already-extended condition.

---

## Hypothesis 3 — Technical × Market Relationship

Previous analysis suggested:

Technical Q4 + Market Q1

performed better than:

Technical Q1 + Market Q4

Temporal test:

Compare these two cells on:

- return
- positive rate
- MAE
- MFE

This tests whether attractive technical setups combined with relatively
low market confirmation are consistently preferable to weak technical
setups combined with extreme confirmation.

---

## Hypothesis 4 — Technical × Macro Crossover

Previous analysis suggested a possible crossover.

Among Technical Q4 observations:

Macro Q1 appeared better than Macro Q4.

Among Technical Q1 observations:

Macro Q4 appeared better than Macro Q1.

Both legs will be tested separately.

The combined crossover should not be considered supported unless both
relationships are reasonably stable.

---

## Hypothesis 5 — Macro × Market Conditional Behavior

Previous analysis suggested strong Market Confirmation behaved very
differently depending on Macro Regime.

Compare:

Macro Q4 + Market Q4

against:

Macro Q1 + Market Q4

This tests whether strong market confirmation is less dangerous when
the macro backdrop is supportive.

---

# Sample-Size Rules

Every cell remains visible.

Cells are never removed solely because they are small.

Flags:

n < 10:
small sample

n < 5:
critical sample

For period-level robustness scoring, pairwise comparisons require at
least 5 observations in BOTH cells.

Technical Q4 versus baseline requires at least 10 observations.

These are diagnostic reliability rules, not trading thresholds.

---

# Descriptive Support Rule

A relationship is considered directionally supported within one
period only when both:

1. five-day return comparison is favorable
2. MAE comparison is favorable

No statistical significance claim is made.

The number and share of eligible periods supporting each relationship
will be reported.

---

# Research Governance

These hypotheses were identified from the same historical sample.

Therefore:

- consistency across periods strengthens confidence
- inconsistency weakens confidence
- temporal consistency does NOT create true out-of-sample validation

No Opportunity Engine trading rule will be selected solely from this
diagnostic.