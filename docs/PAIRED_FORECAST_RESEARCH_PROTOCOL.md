# Paired TMF five-session prospective testing protocol

Status: research-only. V3_FINAL and TimesFM 2.5 remain unchanged, independently
versioned forecasters. This protocol never submits orders or infers trading edge
from one or two observations.

## Experimental unit

One **origin** is an NYSE trading session with a finalized Schwab 1-day
and 5-minute snapshot (post-close plus 30 minutes, before the next NYSE
opening). Both models MUST use the same *immutable snapshot SHA-256*.
Predictions mature five NYSE sessions after the origin. Do not rewrite,
backdate or fill missing sessions with forecasts generated after the fact.

Each day's matched experiment includes:
- V3_FINAL probability of a positive five-session return and frozen base rate.
- TimesFM 2.5 five-session median closing-price path plus q10/q90.
- Both forecast JSONs and independently auditable integrity manifests.
- The input snapshot filename, hash, and exchange session.
- A directional relationship field, **not** an ensemble trade trigger.
- A pending outcome state until market data become observable.

## Day 0: issuance (Windows owner until Proxmox cutover)

After Schwab authentication and the 30-minute data finalization delay,
download **once**, validate the 1d/5m pair, and generate exactly one
V3 and one TimesFM forecast against that file. Each writer must be
separately audited; a failed validation or expired OAuth token blocks the
pair. Do not reconstruct a replacement daily bar manually.

For a prior origin with both forecasts already issued, read-only paired
inspection is:

```bash
python -m scripts.report_paired_forecast_signals --origin 2026-10-22
```

Confirm both audit results, matching source snapshot SHA-256, equal origin
close, and identical five-session target date. The reported
`directional_relation` is `AGREE_UP`, `AGREE_DOWN`, `DISAGREE`,
or `NEUTRAL`.

## Day 1–4: partial-outcome diagnostics

After each succeeding NYSE session has fully closed and a paired Schwab
snapshot is validated, run a **read-only** interim diagnostic against
earlier origin(s):

```bash
python -m scripts.report_paired_forecast_signals \
  --origin 2026-10-21 \
  --daily data/raw/schwab/TMF_1d_<validated-snapshot-id>.csv
```

All observed dates are matched to the original TimesFM forecast steps.
Collect actual close, original median, q10/q90, absolute error, no-change
absolute error, and empirical q10–q90 inclusion. This monitor NEVER
computes V3's five-session Brier score early. Do not treat partial-window
hit rates as independent successes.

Example: October 21 TimesFM forecast for October 22 had median $28.086
and q90 $28.865; observed October 22 close was $29.78. The one-step
absolute median error was about $1.694 and the close exceeded q90.

## Day 5: official, immutable evaluation

Only when the target session has ended and a validated target-day Schwab
pair exists, use the existing V3 evaluation and TimesFM evaluation
scripts to create their separate write-once scoring records. Then run:

```bash
python -m scripts.report_matched_forecasts
```

Only matching matured evaluations with verified hashes and consistent
realized prices enter the matched aggregate. Forecast 2026-10-21
matures 2026-10-28; forecast 2026-10-22 matures 2026-10-29.

## Benchmarks and caution

- V3: five-session directional accuracy and Brier score versus its frozen
  training base-rate probability.
- TimesFM: median five-session path MAE/RMSE and final-step absolute
  error versus persistence, plus empirical q10–q90 coverage/width.
- Cross-model: directional agreement, disagreement, origin close and
  final realized direction. Do not compare Brier and MAE as if they are
  the same scoring unit.
- Overlapping five-session windows are serially dependent. Track how
  many observations are independent enough to support stronger claims;
  descriptive averages alone do not prove forecasting skill.
- Evaluation research must not affect the issuance of V3/TimesFM
  forecasts or change model checkpoints until a separately designated
  versioned experiment is authorized.

## Testing and operational gates

```bash
python -m pytest -q tests/test_report_paired_forecast_signals.py
python -m pytest -q
python -m pip check
python -m scripts.report_paired_forecast_signals --origin 2026-10-21
python -m scripts.report_paired_forecast_signals --origin 2026-10-22
```

Verify tests locally before deployment. The new monitor does not start
daily jobs, refresh OAuth, or write reports to the forecast ledger.

## Historical source-alignment exception

The October 21, 2026 pilot forecasts were independently audited but used
**different** Schwab snapshot captures. The read-only paired monitor labels
such a case `CROSS_SNAPSHOT_HISTORICAL_COMPARISON` and exposes separate
filenames and hashes for each model. This is **not** treated as a strict
same-snapshot controlled experiment; neither forecast or source is modified.
If the same filename is recorded with conflicting hashes, the monitor fails.

The October 22, 2026 experiment used the exact same source snapshot and is
labeled `PAIRED_FORECASTS` / `SAME_VALIDATED_SNAPSHOT`.

When evaluating results, stratify strict same-source experiments separately
from this legacy pilot. Matched outcome scoring uses matching forecast
origin, target and independently verified realized values; it must not imply
equal input captures when they differed. Continue to require exact shared
snapshot SHA-256 for new forecast issuance.
