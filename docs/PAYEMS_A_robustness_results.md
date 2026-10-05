# PAYEMS-A Robustness Results

## Decision

PAYEMS-A advances as a provisional V3 macro component.

PAYEMS-A was frozen before validation and was not modified after
2023-2025 validation results were observed.

2026 remains locked.

## Coefficient Stability

Standardized PAYEMS-A coefficients:

| Validation Year | payroll_change_1m | payroll_revision_total | payroll_broad_revision_event |
|---|---:|---:|---:|
| 2023 | -0.057732 | -0.024331 | 0.200978 |
| 2024 | -0.238050 | -0.042933 | 0.240224 |
| 2025 | -0.306627 | -0.020348 | 0.198456 |

All three PAYEMS features retained the same coefficient sign across
the three expanding training folds.

## Block Bootstrap

Mean across-fold improvements:

### 5-day blocks

- Delta AUC: +0.008335
- Delta Brier: +0.006201
- Delta Log Loss: +0.012978
- Brier bootstrap share positive: 95.15%
- Log-loss bootstrap share positive: 94.95%

### 10-day blocks

- Delta AUC: +0.008335
- Delta Brier: +0.006201
- Delta Log Loss: +0.012978
- Brier bootstrap share positive: 93.70%
- Log-loss bootstrap share positive: 93.35%

The 95% intervals include zero, so these results are supportive but
not statistically conclusive.

## Release-Age Diagnostic

PAYEMS-A improvement was not concentrated around payroll release day.

The strongest Brier and log-loss improvements generally occurred
later in the payroll-release cycle, particularly 16 or more days
after the latest Employment Situation release.

PAYEMS-A should therefore be interpreted primarily as a persistent
macroeconomic-state feature rather than as a short-duration
announcement-reaction signal.

No event-only PAYEMS variant will be created for V3 because doing so
after inspecting validation results would constitute post-validation
feature engineering.

## Status

PAYEMS-A: ADVANCE

PAYEMS-B: REJECTED

2026: LOCKED