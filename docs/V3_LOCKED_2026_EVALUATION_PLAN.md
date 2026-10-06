# Final V3 Locked 2026 Evaluation Plan

## Status

Final V3 feature research is complete.

The final specification was frozen before exposure to 2026 outcomes.

No result from the locked 2026 evaluation may be used to add,
remove, transform, or tune V3 features or hyperparameters.

## Final V3

Technical V1:

- return_5d
- rsi_14
- volatility_20d
- relative_volume
- distance_from_20d_high

PAYEMS-A:

- payroll_change_1m
- payroll_revision_total
- payroll_broad_revision_event

Model:

- Logistic regression
- Existing frozen preprocessing pipeline
- 0.50 classification threshold
- No hyperparameter tuning

## Training Sample

Training uses all eligible observations before January 1, 2026.

A training observation is eligible only when its complete five-day
forward target also ends before January 1, 2026.

No 2026 return information may enter model training.

## Locked OOS Sample

The locked evaluation uses eligible market observations during 2026
with complete five-day forward targets.

At the time of the initial unlock, the source market dataset extends
through October 2, 2026.

The initial evaluation is therefore a 2026 YTD locked OOS evaluation,
not a completed full-calendar-year 2026 evaluation.

## Prespecified Benchmarks

1. Training-sample base-rate forecast
2. Frozen Technical V1
3. Frozen Final V3

## Prespecified Metrics

- ROC AUC
- Brier score
- Log loss
- Accuracy at 0.50
- Balanced accuracy at 0.50

For comparisons against Technical V1:

- Delta AUC
- Delta Brier
- Delta log loss

Positive Brier and log-loss deltas indicate improvement.

## Interpretation Rule

The primary question is whether Final V3 generalizes to unseen 2026
data after completion of the V3 research process.

The locked result will not be used to modify V3.

Any later modeling changes belong to a separately labeled V4 research
cycle and cannot retroactively alter the V3 result.