# TMF Predictive Analytics — V2A Research Plan

## Objective

Determine whether structural U.S. Treasury yield-curve information improves
out-of-sample prediction of TMF 5-trading-day direction relative to the
frozen Technical Baseline V1.

V2A is an incremental experiment. Technical Baseline V1 remains unchanged
and serves as the benchmark.

---

## Target

Primary target:

forward_return_5

Classification target:

target_up_5d = 1 if forward_return_5 > 0
target_up_5d = 0 otherwise

The final five observations without known 5-day forward returns are not
eligible for training or validation.

---

## Development and Holdout Periods

Development period:

2021-10-04 through 2025-12-31

Final holdout:

2026-01-01 onward

The 2026 period remains locked during V2A feature evaluation and model
selection.

No V2A model decision will use 2026 predictive performance.

---

## Validation Method

Use the same purged expanding walk-forward framework as V1.

Fold 1:
Train through 2022
Validate on 2023

Fold 2:
Train through 2023
Validate on 2024

Fold 3:
Train through 2024
Validate on 2025

For the 5-day forward target, training and validation observations whose
target window extends beyond the applicable fold boundary are removed.

No random train/test split is permitted.

---

## Model

Use the same Logistic Regression architecture as Technical Baseline V1:

- Median imputation
- StandardScaler
- LogisticRegression
- C = 1.0
- max_iter = 1000
- random_state = 42
- classification threshold = 0.50

No hyperparameter optimization will be performed during V2A.

---

## Benchmark — Technical Baseline V1

Features:

- return_5d
- rsi_14
- volatility_20d
- relative_volume
- distance_from_20d_high

This feature set remains frozen.

---

## V2A Structural Curve Features

The V2A Treasury curve model will use exactly:

- curve_level
- curve_slope_30y_3m_bp
- curve_curvature_5y_bp
- curve_level_change_5d_bp
- curve_slope_change_5d_bp
- curve_curvature_change_5d_bp
- curve_level_volatility_20d
- curve_slope_volatility_20d
- curve_curvature_volatility_20d

These features were selected before evaluating V2A predictive performance.

---

## Models to Compare

### Model 1 — Base Rate

Constant probability benchmark based on the training-period target rate.

### Model 2 — Technical Baseline V1

Frozen five-feature technical model.

### Model 3 — V2A Curve

Nine structural Treasury curve features.

### Model 4 — Technical + V2A Curve

The five frozen technical features plus the nine V2A structural curve
features.

---

## Primary Metrics

Evaluate each validation fold and the combined walk-forward predictions
using:

- ROC AUC
- Brier score
- Log loss
- Accuracy
- Balanced accuracy

AUC measures ranking ability.

Brier score and log loss measure probability quality.

Accuracy is secondary because the target classes are imbalanced.

---

## Secondary Analysis

For model probabilities, evaluate:

- probability quintiles
- realized forward returns by quintile
- realized positive-return rate by quintile
- Q5 minus Q1 return spread
- stability of ranking across 2023, 2024, and 2025

The primary question is whether higher predicted probabilities consistently
correspond to better subsequent TMF outcomes.

---

## V2A Hypotheses

### H1 — Curve Level

Higher Treasury yield levels should generally represent greater long-duration
rate pressure on TMF.

Expected relationship:

higher curve level -> weaker TMF forward returns

### H2 — Curve Slope

Changes in the short-to-long Treasury relationship may identify different
rate regimes that are not represented by technical indicators alone.

The predictive sign is not fixed in advance because steepening can occur
through materially different economic mechanisms.

### H3 — Curve Curvature

Changes in the belly of the Treasury curve relative to short and long
maturities may contain information about changes in rate expectations and
term structure dynamics.

### H4 — Curve Changes

Changes in level, slope, and curvature may contain more short-horizon
predictive information than their absolute levels.

### H5 — Curve Volatility

Higher volatility in Treasury curve factors may alter the behavior and risk
of a daily leveraged long-duration Treasury ETF.

### H6 — Incremental Information

If V2A contains useful information not already represented by TMF technical
features, the Technical + V2A model should outperform Technical Baseline V1
on walk-forward validation.

---

## Decision Rules

V2A will not be judged from a single metric or a single year.

Evidence in favor of V2A requires:

1. Improvement in average walk-forward ROC AUC relative to Technical
   Baseline V1.

2. No material deterioration in Brier score and log loss.

3. Predictive ranking that is reasonably stable across the 2023, 2024,
   and 2025 validation folds.

4. Evidence that the combined Technical + V2A model adds information beyond
   the V2A-only and Technical-only models.

If V2A fails these criteria, its features will not be optimized against the
same validation folds simply to improve reported performance.

Instead, the result will be documented and the project will proceed to the
next pre-specified research stage.

---

## V2B

V2B will evaluate fitted yield-curve representations such as Nelson-Siegel.

V2B is separate from V2A so that the project can determine whether a fitted
term-structure representation provides incremental predictive value beyond
simple empirical level, slope, and curvature factors.

---

## Final Holdout Policy

The 2026 holdout will remain locked throughout V2A and subsequent feature
development.

It will be evaluated only after the model architecture and feature-selection
process have been finalized.

---

## V2A Experimental Results

V2A was evaluated using the pre-specified purged expanding
walk-forward validation framework covering the 2023, 2024, and
2025 validation periods.

The 2026 holdout was not used during model evaluation or feature
selection.

### Average Walk-Forward Performance

| Model | ROC AUC | Brier Score | Log Loss | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Base Rate | 0.5000 | 0.2503 | 0.6940 | 0.5440 | 0.5000 |
| Technical Baseline V1 | 0.5554 | 0.2579 | 0.7114 | 0.5508 | 0.5088 |
| Treasury V1 | 0.5566 | 0.2656 | 0.7346 | 0.5090 | 0.5287 |
| Combined V1 | 0.5614 | 0.2729 | 0.7497 | 0.4873 | 0.5127 |
| V2A Curve | 0.5402 | 0.2825 | 0.7710 | 0.5034 | 0.5305 |
| Technical + V2A Curve | 0.5159 | 0.2897 | 0.8094 | 0.4940 | 0.5200 |

### V2A Curve ROC AUC by Validation Fold

| Validation Fold | V2A Curve AUC |
|---|---:|
| 2023 | 0.5635 |
| 2024 | 0.5325 |
| 2025 | 0.5245 |

The V2A curve-only model produced ROC AUC above 0.50 in all three
validation folds, indicating modest ranking information. However,
performance weakened across the validation periods and did not
outperform the frozen Technical Baseline V1 on average.

### Incremental V2A Test

The primary V2A test was whether structural yield-curve factors added
predictive information to Technical Baseline V1.

| Validation Fold | Technical V1 AUC | Technical + V2A AUC | Difference |
|---|---:|---:|---:|
| 2023 | 0.5143 | 0.5019 | -0.0124 |
| 2024 | 0.5816 | 0.5158 | -0.0658 |
| 2025 | 0.5702 | 0.5299 | -0.0403 |
| Average | 0.5554 | 0.5159 | -0.0395 |

Adding the V2A structural curve factors reduced ROC AUC in every
validation fold.

Probability-quality metrics also deteriorated relative to Technical
Baseline V1:

- Brier score increased from 0.2579 to 0.2897.
- Log loss increased from 0.7114 to 0.8094.

### Decision

**V2A result:** Simple empirical level/slope/curvature representation
did not add incremental predictive value to Technical Baseline V1.

Curve-only ranking remained modestly above random across all three
folds, but combining V2A with technical features degraded ROC AUC,
Brier score, and log loss.

**V2A is rejected as the primary Treasury representation.**

The V2A feature set will not be modified or optimized against the same
2023-2025 validation folds in response to these results.

The next research stage is V2B, which will evaluate a fitted
Nelson-Siegel term-structure representation while preserving the same
walk-forward validation methodology and keeping the 2026 holdout
locked.