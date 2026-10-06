# Inflation V3 Frozen Validation Results

## Status

INFLATION-A: REJECTED

INFLATION-B: ADVANCES TO ROBUSTNESS ANALYSIS

2026 remains locked.

No inflation candidate definitions were modified after validation
results were observed.

## V3 Base

Technical V1 + PAYEMS-A.

## INFLATION-A

Features:

- breakeven_10y
- breakeven_10y_change_5d_bp

Mean validation deltas versus V3 base:

- AUC: -0.012564
- Brier: -0.002306
- Log loss: -0.004919

INFLATION-A degraded all three primary metrics overall and is rejected.

No additional breakeven variants will be created for V3.

## INFLATION-B

Features:

- core_cpi_yoy_pct
- core_cpi_3m_annualized_pct

### Validation results

| Year | Delta AUC | Delta Brier | Delta Log Loss |
|---|---:|---:|---:|
| 2023 | -0.000806 | +0.007245 | +0.014209 |
| 2024 | -0.003939 | +0.004668 | +0.011071 |
| 2025 | -0.002604 | +0.000250 | +0.000532 |

Mean:

- Delta AUC: -0.002449
- Delta Brier: +0.004054
- Delta Log Loss: +0.008604

INFLATION-B improved Brier score and log loss in all three
validation years, but slightly reduced AUC in every year.

The 2025 probability-score improvement was very small.

## Decision

INFLATION-A is rejected.

INFLATION-B advances to a prespecified robustness analysis focused on:

- coefficient stability
- probability-shift magnitude
- calibration
- performance by time since CPI state update
- paired moving-block bootstrap

No additional inflation feature selection or tuning is authorized.

2026 remains locked.