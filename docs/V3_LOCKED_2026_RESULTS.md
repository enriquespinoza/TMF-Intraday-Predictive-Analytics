# Final V3 Locked 2026 Out-of-Sample Results

## Status

Final V3 failed to demonstrate out-of-sample predictive value
on the locked 2026 YTD sample.

The V3 specification will not be modified based on these results.

Any subsequent modeling work will be conducted as a separately
labeled V4 research cycle.

## Frozen V3 Specification

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
- Frozen preprocessing pipeline
- 0.50 classification threshold

## Split Audit

Training:

- Observations: 1,037
- Market range: 2021-11-05 to 2025-12-23
- Latest target end: 2025-12-31
- Up rate: 42.82%

Locked 2026 OOS:

- Observations: 184
- Market range: 2026-01-02 to 2026-09-25
- Latest target end: 2026-10-02
- Up rate: 41.30%

No 2026 target information entered model training.

## Locked OOS Results

| Model | AUC | Brier | Log Loss | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Training base rate | 0.500000 | 0.242667 | 0.678415 | 0.586957 | 0.500000 |
| Technical V1 | 0.479654 | 0.243807 | 0.680694 | 0.586957 | 0.500000 |
| Final V3 | 0.481360 | 0.251757 | 0.696915 | 0.505435 | 0.465643 |

## V3 Versus Technical V1

- Delta AUC: +0.001706
- Delta Brier: -0.007950
- Delta Log Loss: -0.016221

Positive Brier and log-loss deltas would indicate improvement.

V3 therefore produced essentially no ranking improvement and
materially worse probability forecasts.

## Calibration-Level Observation

Observed 2026 up rate:

- 41.30%

Mean Technical V1 probability:

- 41.37%

Mean V3 probability:

- 45.60%

The PAYEMS-enhanced V3 specification produced a materially more
bullish average probability than the realized 2026 outcome rate.

## Final Decision

Final V3 is not validated for deployment or trading decisions.

The locked 2026 result is final and will not be used to modify V3.

Future research must begin under a separately defined V4 protocol.