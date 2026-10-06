# INFLATION-B Robustness Results

## Decision

INFLATION-B is rejected for V3.

INFLATION-A was previously rejected.

Therefore no inflation candidate advances into the final V3 model.

2026 remains locked.

## Candidate

INFLATION-B added the following features to Technical V1 + PAYEMS-A:

- core_cpi_yoy_pct
- core_cpi_3m_annualized_pct

The candidate specification was frozen before validation.

## 1. Coefficient Stability

### Core CPI YoY

Standardized coefficients:

- 2023: -0.081416
- 2024: -0.244002
- 2025: -0.328866

The sign remained negative across all folds.

### Core CPI 3-Month Annualized

Standardized coefficients:

- 2023: -0.044405
- 2024: +0.072984
- 2025: +0.104237

The coefficient changed sign after the first fold.

Coefficient stability was therefore mixed.

## 2. Distribution of Score Improvement

Share of observations with improved Brier score:

- 2023: 49.39%
- 2024: 43.32%
- 2025: 45.31%

Median Brier improvement was negative in all three folds.

Share of observations with improved log loss:

- 2023: 49.39%
- 2024: 43.32%
- 2025: 45.31%

Median log-loss improvement was negative in all three folds.

The positive mean Brier and log-loss results were therefore not
broadly distributed across observations.

## 3. Calibration

INFLATION-B improved calibration in 2023 and 2025 but worsened
calibration in 2024.

Calibration improvement was not stable across validation folds.

## 4. CPI Update-Age Analysis

No stable relationship was observed between time since the latest
Core CPI state update and incremental model performance.

Days 1-5 after an update produced negative Brier and log-loss deltas
in all three validation years.

Later update-age buckets showed inconsistent results across years.

## 5. Moving-Block Bootstrap

Mean across-fold results:

### 5-day blocks

- Delta Brier: +0.004054
- Positive bootstrap share: 84.35%
- 95% interval: -0.003615 to +0.013006

- Delta Log Loss: +0.008604
- Positive bootstrap share: 83.75%
- 95% interval: -0.007700 to +0.027572

### 10-day blocks

- Delta Brier: +0.004054
- Positive bootstrap share: 82.80%
- 95% interval: -0.005136 to +0.014005

- Delta Log Loss: +0.008604
- Positive bootstrap share: 82.65%
- 95% interval: -0.011114 to +0.029873

All confidence intervals included zero.

Bootstrap support was weaker than the previously accepted PAYEMS-A
candidate.

## Final Status

INFLATION-A: REJECTED

INFLATION-B: REJECTED

Inflation family: REJECTED FOR V3

No additional inflation variants will be tested for V3.

2026 remains locked.