# Growth/Activity V3 Frozen Validation Results

## Status

GROWTH-A: REJECTED

GROWTH-B: REJECTED

Growth/activity family: REJECTED FOR V3

2026 remains locked.

No growth candidate definitions were modified after validation results
were observed.

## V3 Base

Technical V1 + PAYEMS-A.

## GROWTH-A

Features:

- retail_sales_mom_pct
- retail_sales_3m_annualized_pct

Validation deltas versus V3 base:

| Year | Delta AUC | Delta Brier | Delta Log Loss |
|---|---:|---:|---:|
| 2023 | -0.005236 | -0.002802 | -0.008276 |
| 2024 | +0.013284 | +0.004108 | +0.009422 |
| 2025 | -0.017962 | -0.008728 | -0.018794 |

Across-fold mean:

- Delta AUC: -0.003305
- Delta Brier: -0.002474
- Delta Log Loss: -0.005883

GROWTH-A improved performance in 2024 but degraded performance
overall and is rejected.

## GROWTH-B

Features:

- industrial_production_mom_pct
- industrial_production_3m_annualized_pct

Validation deltas versus V3 base:

| Year | Delta AUC | Delta Brier | Delta Log Loss |
|---|---:|---:|---:|
| 2023 | +0.002551 | +0.001713 | +0.003742 |
| 2024 | -0.016355 | -0.002703 | -0.005678 |
| 2025 | -0.018697 | -0.000685 | -0.001616 |

Across-fold mean:

- Delta AUC: -0.010834
- Delta Brier: -0.000558
- Delta Log Loss: -0.001184

GROWTH-B produced a small improvement in 2023 but degraded
performance overall and is rejected.

## Final Growth Decision

Neither frozen growth candidate advances to robustness testing.

No additional growth/activity variants will be tested for V3.

This completes new macro-family feature research for V3.

2026 remains locked.