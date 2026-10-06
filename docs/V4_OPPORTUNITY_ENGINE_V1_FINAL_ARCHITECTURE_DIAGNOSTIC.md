# V4 Opportunity Engine V1 — Final Architecture Diagnostic

## Purpose

This is the final Opportunity Engine V1 diagnostic before defining the
trading state machine.

No predictive model is trained.

No component weights are optimized.

No new technical, Treasury, macro, or market thresholds are selected.

The test addresses one architecture question:

> When Technical Opportunity is already strong, is a non-extreme
> Treasury Confirmation environment preferable to an extreme Treasury
> Confirmation environment?

---

# Motivation

Prior Opportunity Engine V1 research found:

- Technical Opportunity Q4 behaved better than the common baseline in
  several periods but was not independently robust enough to serve as
  an automatic entry trigger.

- Treasury Confirmation Q2 generally performed better than Treasury
  Confirmation Q4.

- Extreme Treasury Confirmation Q4 frequently behaved like an
  already-extended market condition.

- Macro Regime and Market Confirmation interactions were not stable
  enough to receive trading authority.

Therefore, the final V1 architecture diagnostic does not attempt to
optimize all four components.

It tests only the interaction between:

1. Strong Technical Opportunity
2. Extreme versus non-extreme Treasury Confirmation

---

# Frozen Inputs

The diagnostic reuses the quartile classifications already created by
the frozen Opportunity Engine V1 interaction dataset.

Quartiles are NOT recalculated.

Required fields:

- technical_quartile
- treasury_quartile
- market_date
- forward_return_1
- forward_return_5
- forward_5d_mae
- forward_5d_mfe

---

# Primary Setup

Only observations where:

technical_quartile == Q4_HIGH

are included in the primary architecture comparison.

This represents the candidate high-quality technical setup.

---

# Treasury Classification

## Non-Extreme Treasury

Treasury quartiles:

- Q1_LOW
- Q2
- Q3

are pooled into:

NON_EXTREME

## Extreme Treasury

Treasury quartile:

- Q4_HIGH

is classified as:

EXTREME

The purpose of pooling Q1-Q3 is to avoid selecting Treasury Q2 as an
optimized historical sweet spot.

---

# Primary Architecture Contrast

Compare:

TECH_Q4 + TREASURY_NON_EXTREME

against:

TECH_Q4 + TREASURY_EXTREME

Metrics:

- observation count
- mean 1-day forward return
- mean 5-day forward return
- median 5-day forward return
- five-day positive rate
- mean five-day MAE
- median five-day MAE
- mean five-day MFE
- median five-day MFE

---

# Favorable Direction

NON_EXTREME is considered directionally favorable to EXTREME within a
period only when BOTH:

1. mean five-day return is higher
2. mean five-day MAE is higher

Because MAE is negative, a higher value means less adverse movement.

Example:

-2% MAE is better than -5% MAE.

Positive-rate and MFE differences are reported but are not required
for the primary support rule.

---

# Temporal Robustness

The primary contrast is evaluated on:

## Full Common Sample

All eligible Technical Q4 observations.

## Calendar Periods

- 2022-2023
- 2024
- 2025
- 2026 YTD

## Equal-Count Chronological Blocks

The existing four chronological blocks from the prior temporal
diagnostic are recreated deterministically from the complete common
sample:

- BLOCK_1
- BLOCK_2
- BLOCK_3
- BLOCK_4

---

# Sample Reliability

For period-level directional support:

Both NON_EXTREME and EXTREME cells must contain at least:

5 observations

Cells below 10 observations are flagged as small samples.

The full-sample comparison is always displayed.

---

# Secondary Descriptive Breakdown

Within Technical Q4, Treasury Q1, Q2, Q3, and Q4 are also reported
separately.

This breakdown is descriptive only.

It must not be used to choose one individual Treasury quartile as an
entry threshold.

---

# Macro and Market Confirmation

Macro Regime V1 and Market Confirmation V1 receive no trading authority
in this diagnostic.

They are intentionally excluded from the primary decision rule.

They may continue to be displayed as contextual information in the
future trading interface.

---

# Candidate Architecture

If the NON_EXTREME condition is reasonably stable across time, the
candidate state-machine architecture becomes:

Risk Governor V3 blocked
    -> FLAT / NO ENTRY

Risk Governor V3 open
and Technical Opportunity not strong
    -> WATCH / FLAT

Risk Governor V3 open
and Technical Opportunity strong
and Treasury Confirmation non-extreme
    -> ENTRY CANDIDATE

Risk Governor V3 open
and Technical Opportunity strong
and Treasury Confirmation extreme
    -> ARMED / DO NOT CHASE

This diagnostic does not itself implement those states.

---

# Research Governance

The architecture hypothesis was developed from the same historical
sample.

Therefore this is not independent out-of-sample validation.

The result can justify a candidate state-machine design for forward
testing, but not a claim of proven profitability.