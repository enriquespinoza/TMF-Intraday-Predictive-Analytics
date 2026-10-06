# V4 Risk Governor V3 Specification

## Purpose

Risk Governor V3 tests one narrow architectural change to Risk
Governor V2.

V2 separated entry permission from existing-position exposure control.

However, V2 still blocked every new entry whenever TMF was below its
21-day EMA.

Historical diagnostic research showed that short-duration EMA21
breaks behaved differently from persistent below-EMA conditions.

Risk Governor V3 therefore replaces the binary EMA21 entry veto with
a persistence-based EMA21 entry veto.

---

# V3 Research Hypothesis

A short break below the 21-day EMA may represent a pullback rather
than a persistent trend failure.

Therefore:

- 0 days below EMA21:
  no EMA entry block

- 1 to 3 consecutive sessions below EMA21:
  no EMA entry block

- 4 or more consecutive sessions below EMA21:
  new entries blocked

The 4-session threshold is research-derived from historical V4
diagnostic analysis.

It is not claimed to be independently validated.

---

# What Does Not Change

Risk Governor V3 does not change:

- TMF one-day return thresholds
- Position-drawdown thresholds
- 30-year Treasury yield thresholds
- 10-year real-yield thresholds
- Volatility thresholds
- Three-warning RED escalation
- Exposure-state definitions
- Exposure multipliers
- Recovery hysteresis
- Intraday emergency interface

Existing-position exposure control remains identical to V2.

---

# Exposure Governor

The exposure governor continues to use:

- TMF one-day return
- Position drawdown
- 30-year Treasury one-day change
- 30-year Treasury five-day change
- 10-year real-yield five-day change
- Volatility expansion
- Intraday FORCE_EXIT

EMA21 persistence does not directly reduce existing exposure.

---

# Entry Gate

A new entry is blocked when:

1. Effective exposure state is not GREEN
2. Critical market data is stale
3. EMA21 below-streak is at least 4 sessions
4. Intraday layer returns BLOCK_ENTRY
5. Intraday layer returns FORCE_EXIT

A 1-to-3-session EMA21 break does not automatically block entry.

---

# EMA21 Persistence

The input:

below_ema21_streak

represents the number of consecutive trading sessions on which TMF
has closed below its 21-day EMA.

Examples:

Above EMA21:
0

First close below EMA21:
1

Second consecutive close below:
2

Third consecutive close below:
3

Fourth consecutive close below:
4

A close back above EMA21 resets the streak to zero.

---

# Research Governance

Risk Governor V3 is a new candidate policy.

V2 remains unchanged.

The 4-session threshold was selected after examining historical EMA21
persistence behavior and therefore must be treated as in-sample
research.

V3 should be evaluated primarily as an architectural candidate.

It should not be described as independently validated until it is
tested on future unseen data.

---

# Initial V3 Evaluation

The first V3 evaluation will compare V2 and V3 on:

- Entry permission frequency
- Number of GREEN days still blocked
- Number of short-pullback days restored
- Number of persistent-breakdown days blocked

Because exposure-state logic is unchanged, V2 and V3 should produce
identical:

- Exposure states
- Maximum permitted exposure
- Turnover
- Portfolio overlay returns
- Maximum drawdown
- Volatility

Any difference in those quantities indicates an implementation error.

Only after confirming this clean separation will the V4 opportunity
engine be connected to Risk Governor V3.