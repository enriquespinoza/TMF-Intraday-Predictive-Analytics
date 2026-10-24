# V4 opportunity: independent prospective-state ledger

Research infrastructure only. V4 is a **state machine**, not a price forecaster or trading authorization system. The ledger neither reads nor writes V3_FINAL/TimesFM 2.5 artifacts.

## Upstream sources and limitations

The frozen decision function is in src/signals/opportunity_state_machine.py and its policy in config/opportunity_engine_v1_policy.json. Risk Governor V3 has veto authority. The ledger **does not** compute the Risk Governor or component scores; upstream V4 research pipelines must produce each score and a point-in-time risk permission before V4 issuance is possible.

Historical reports/V4_opportunity_state_machine_v1_daily.csv contains retrospective research and forward-return fields. **Do not backfill October 21, 22 or 23 as if they were prospectively issued V4 states.** No such V4 live states were archived through this new ledger. The previously observed October 2 NO_TRADE is also historical unless separately timestamp-attested.

Before recording a new live observation, create three genuine independently timestamped JSON files directly under data/raw/v4_live/, one per role:

- risk_governor_v3: new_entries_allowed (JSON boolean), exposure_state (GREEN/YELLOW/RED/EMERGENCY_EXIT)
- technical_opportunity_v1: technical_opportunity_score (numeric 0–100)
- treasury_confirmation_v1: treasury_confirmation_score (numeric 0–100)

Each JSON must include schema_version=v4_live_component_v1, role, market_date, source_snapshot, source_sha256, available_at_utc. This timestamp must reflect genuine knowledge as of that time, after close+30 minutes and before issuance. Risk state must be GREEN if new_entries_allowed is true. Component evidence containing forward/future outcome fields is rejected. Provenance files must be preserved. **Local JSON timestamps are claims, not external independently verified attestations.** Validate all upstream feature timestamps, yield/macro vintages, and Risk Governor carry-over memory before claiming fully point-in-time operation.

## Prospectively record a state

Only after the actual session is finalized, and before the next market opening, run from the repository root with the tmf-analytics environment:

```bash
python -m scripts.v4_opportunity_ledger issue \
  --origin YYYY-MM-DD \
  --daily data/raw/schwab/TMF_1d_<capture-id>.csv \
  --risk data/raw/v4_live/<risk-file>.json \
  --technical data/raw/v4_live/<technical-file>.json \
  --treasury data/raw/v4_live/<treasury-file>.json \
  --check
```

Inspect the prospective decision and independently generated evidence. If legitimate and source-audited, change --check to --run to save write-once output under data/forward_test/v4_opportunity/ and a SHA-256 manifest under runs/. Reissues and tampering fail closed. --check does not save a state. An expired origin must remain unissued. No broker or trading order is touched.

## Five-session outcome evaluation

After the fifth future NYSE session has closed, provide a separately captured, complete, validated target-date Schwab daily+5-minute pair:

```bash
python -m scripts.v4_opportunity_ledger score \
  --origin YYYY-MM-DD \
  --daily data/raw/schwab/TMF_1d_<target-capture-id>.csv \
  --check
python -m scripts.v4_opportunity_ledger score \
  --origin YYYY-MM-DD \
  --daily data/raw/schwab/TMF_1d_<target-capture-id>.csv \
  --run
python -m scripts.v4_opportunity_ledger report
```

Outcomes include close-to-close five-session realized return, positive-return indicator and minimum/maximum **closing-path** excursions. These are neither intraday MAE/MFE nor strategy P&L, because actual next-open execution, costs, sizing, governor exits and slippage are not simulated. Read-only state-conditioned reporting never rewrites the original issued state.

## Hypotheses and safeguards

Compare outcomes conditional on ENTRY_CANDIDATE, WATCH, ARMED_DO_NOT_CHASE and NO_TRADE, recording observations and baseline by market/volatility regime. Track risk-governor-veto sessions separately. Avoid overlapping-horizon independence claims, ex-post threshold changes and selection based on realized results. Future V4 vs V3 vs TimesFM reports must join on audited session and actual point-in-time provenance, remain research-only, and never feed V3 or TimesFM results into V4 policy.

## Local checks

```bash
python -m pytest -q tests/test_v4_opportunity_ledger.py
python -m pytest -q
python -m pip check
python -m scripts.v4_opportunity_ledger report
```

The initial ledger is not an upstream V4 data generator, automatic scheduler or broker connector. True forward testing starts only with complete, contemporaneously recorded upstream V4 components.
