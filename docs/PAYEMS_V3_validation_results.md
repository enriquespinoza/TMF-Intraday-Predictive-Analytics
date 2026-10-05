# PAYEMS V3 Frozen Validation Results

## Status

PAYEMS-A advances to robustness analysis.

PAYEMS-B is rejected.

The PAYEMS candidate definitions were frozen in commit:

bc3f0b2 Freeze PAYEMS candidates for V3 evaluation

No PAYEMS feature definitions were changed after validation results
were observed.

2026 remains locked.

---

## Models

### Technical V1 matched control

- return_5d
- rsi_14
- volatility_20d
- relative_volume
- distance_from_20d_high

### PAYEMS-A

Technical V1 plus:

- payroll_change_1m
- payroll_revision_total
- payroll_broad_revision_event

### PAYEMS-B

Technical V1 plus:

- payroll_change_1m
- payroll_revision_2m
- payroll_broad_revision_event

---

## Frozen Validation Folds

1. Train through 2022, validate 2023
2. Train through 2023, validate 2024
3. Train through 2024, validate 2025

Forward-return targets were purged at fold boundaries.

All models used the same eligible observations within each fold.

---

## PAYEMS-A Results

| Year | V1 AUC | PAYEMS-A AUC | Delta AUC | Delta Brier | Delta Log Loss |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.600832 | 0.620569 | +0.019737 | +0.004047 | +0.007403 |
| 2024 | 0.562216 | 0.586849 | +0.024633 | +0.007757 | +0.016509 |
| 2025 | 0.533320 | 0.513956 | -0.019364 | +0.006799 | +0.015023 |

Mean AUC improvement: +0.008335

Mean Brier improvement: +0.006201

Mean log-loss improvement: +0.012978

PAYEMS-A improved Brier score and log loss in all three validation
years, although AUC deteriorated in 2025.

---

## PAYEMS-B Results

Mean AUC improvement: +0.005699

Mean Brier improvement: +0.001057

Mean log-loss improvement: -0.014010

PAYEMS-B showed substantial probability-score deterioration in 2023
and is rejected.

---

## Decision

PAYEMS-A advances to robustness and diagnostic analysis.

PAYEMS-B does not advance.

This decision does not authorize any additional PAYEMS feature
selection or hyperparameter tuning.

The 2026 out-of-sample period remains locked.