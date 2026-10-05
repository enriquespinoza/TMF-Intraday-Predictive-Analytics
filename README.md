# TMF Intraday Predictive Analytics

A quantitative research and forward-testing system investigating whether Treasury-market, macroeconomic, cross-asset, and TMF-specific information contains statistically significant and out-of-sample predictive information about short-term TMF returns.

The project combines:

- Python-based quantitative research
- Intraday market-data engineering
- Multiple regression and regularized models
- Time-series and walk-forward validation
- thinkorswim market analysis
- thinkScript indicators
- paperMoney prospective testing
- Model-vs-realized performance monitoring

The primary prediction horizons are:

**15, 30, 60, and 120 minutes**

with the **60-minute forecast** serving as the initial primary research target.

> **Current Status:** Research Design & Data Source Selection

---

## Research Question

**Do changes in Treasury yields, Treasury bond prices, macroeconomic information, cross-asset markets, and TMF market behavior contain incremental predictive information about TMF returns over the next 15, 30, 60, and 120 minutes?**

The purpose of the project is not to assume these relationships are predictive, but to measure whether statistically significant and out-of-sample predictive relationships actually exist.

---

## Hypotheses

### Primary Null Hypothesis

**H₀:** After controlling for TMF's own recent market behavior, Treasury-rate, bond-market, macroeconomic, and cross-asset variables provide no incremental predictive information about future TMF returns.

### Alternative Hypothesis

**Hₐ:** At least one explanatory variable contains statistically significant incremental information about future TMF returns.

### Initial Economic Hypotheses

The project will initially test whether:

- Increases in long-duration Treasury yields have a negative relationship with subsequent TMF returns.
- Positive long-duration Treasury bond returns have a positive relationship with subsequent TMF returns.
- Changes in the shape of the Treasury yield curve provide information beyond individual yield changes.
- Macroeconomic surprises influence the magnitude and persistence of intraday TMF movements.
- Treasury-market volatility affects the strength of TMF's response to changes in yields.
- Predictive relationships vary across forecast horizons.
- Predictive relationships vary across different interest-rate and volatility regimes.

These are hypotheses to be tested rather than assumptions embedded into the final model.

---

## Prediction Targets

Forward TMF returns will initially be modeled across four horizons:

| Target | Forecast Horizon |
|---|---:|
| `TMF_Fwd_15m` | 15 minutes |
| `TMF_Fwd_30m` | 30 minutes |
| `TMF_Fwd_60m` | 60 minutes |
| `TMF_Fwd_120m` | 120 minutes |

The **60-minute horizon** will serve as the initial primary research target while predictive capability across all four horizons is compared.

Forward returns will be calculated approximately as:

\[
R_{t,h} = \frac{P_{t+h}}{P_t} - 1
\]

where:

- `t` = prediction timestamp
- `h` = forecast horizon
- `P_t` = TMF price at the prediction timestamp
- `P_(t+h)` = TMF price at the end of the forecast horizon

---

## Candidate Predictors

### Treasury Rates

Potential rate variables include:

- 10-Year Treasury yield
- 20-Year Treasury yield
- 30-Year Treasury yield
- 5-minute yield changes
- 15-minute yield changes
- 30-minute yield changes
- 60-minute yield changes
- 10Y–30Y yield spread
- Changes in Treasury curve slope

### Treasury & Bond Markets

Potential bond-market variables include:

- TLT returns
- TLT volume
- TLT volatility
- Treasury futures
- Long-duration Treasury futures
- Treasury-market volatility measures

### Cross-Asset Markets

Potential cross-asset predictors include:

- S&P 500 / SPY
- Nasdaq / QQQ
- U.S. Dollar Index or suitable dollar proxy
- Crude oil
- Gold
- Equity-market volatility
- Interest-rate volatility

### TMF Market Variables

TMF-specific variables may include:

- OHLCV
- 5-minute returns
- 15-minute returns
- Lagged returns
- VWAP
- Distance from VWAP
- Moving-average relationships
- Rolling realized volatility
- Volume
- Relative volume
- Opening-range position
- Intraday high/low position
- Time of day

### Macroeconomic Information

Event variables may include:

- CPI
- PCE inflation
- Nonfarm payrolls
- Unemployment
- Jobless claims
- Retail sales
- ISM
- JOLTS
- GDP
- Federal Reserve decisions
- Federal Reserve communications
- Treasury auctions

Where historical consensus forecasts are available, macroeconomic surprises may be represented as:

\[
Surprise = Actual - Consensus
\]

and potentially standardized relative to historical surprise distributions.

---

## Methodology

The project will progress through a structured quantitative research pipeline.

### Phase 1 — Research Design

Define:

- Research question
- Null and alternative hypotheses
- Forecast horizons
- Candidate explanatory variables
- Statistical evaluation criteria

### Phase 2 — Data Acquisition

Collect and preserve unmodified market and economic data.

Raw source data will remain unchanged so every transformation can be reproduced.

### Phase 3 — Data Cleaning & Synchronization

The pipeline will:

- Standardize timestamps
- Convert data to a common timezone
- Identify trading sessions
- Handle market holidays and early closes
- Identify missing observations
- Remove invalid observations
- Synchronize instruments to common intervals
- Prevent improper forward-filling
- Validate timestamp integrity

### Phase 4 — Feature Engineering

Potential engineered variables include:

- Log returns
- Yield changes
- Yield-curve changes
- Rolling volatility
- Lagged returns
- VWAP distance
- Relative volume
- Momentum variables
- Macroeconomic surprise variables
- Event indicators
- Time-of-day variables

Forward TMF returns will then be generated for each forecast horizon.

### Phase 5 — Exploratory Data Analysis

Initial analysis will examine:

- Distributions
- Missing values
- Correlations
- Lead-lag relationships
- Stationarity
- Outliers
- Volatility clustering
- Market-regime differences
- Event vs. non-event periods

### Phase 6 — Predictive Capability Analysis

Individual predictors will first be tested against future TMF returns.

The objective is to determine:

> Which variables contain information about future TMF returns, and at which forecast horizon is that information strongest?

### Phase 7 — Multivariate Regression

An initial multiple regression may take the general form:

\[
R_{TMF,t+60} =
\beta_0 +
\beta_1\Delta Y_{30Y} +
\beta_2\Delta Y_{10Y} +
\beta_3R_{TLT} +
\beta_4R_{DXY} +
\beta_5R_{SPY} +
\beta_6R_{Oil} +
\beta_7VWAP_{TMF} +
\beta_8Volume +
\epsilon
\]

The exact specification will depend on the results of the exploratory analysis.



## System Architecture

```text
             MARKET DATA
                  │
                  ▼
         ┌─────────────────┐
         │ Python ETL      │
         │ Cleaning        │
         │ Synchronization │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Feature Engine  │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Statistical     │
         │ Models          │
         │                 │
         │ OLS             │
         │ Ridge / Lasso   │
         │ Classification  │
         │ ML Models       │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Walk-Forward    │
         │ Validation      │
         └────────┬────────┘
                  │
             validated?
                  │
                  ▼
       ┌──────────────────────┐
       │ Signal Generation    │
       │                      │
       │ Expected Return      │
       │ Direction Probability│
       │ Signal Strength      │
       └──────────┬───────────┘
                  │
            ┌─────┴─────┐
            ▼           ▼
      thinkorswim     Reporting
      thinkScript     / API
            │
            ▼
       paperMoney
            │
            ▼
    Prospective Testing
            │
            ▼
 Backtest vs Forward Test
```
---

## Model Progression

The planned model progression is:

```text
OLS Regression
      ↓
Ridge / Lasso Regression
      ↓
Logistic Regression
      ↓
Tree-Based Models
      ↓
Gradient Boosting
```

More complex models will only be introduced after establishing interpretable statistical baselines.

The objective is not simply to maximize model complexity.

Each additional model must demonstrate improved **out-of-sample predictive capability**.

---

## Multicollinearity

Treasury yields, Treasury ETFs, and Treasury futures are naturally highly correlated.

Multicollinearity will therefore be evaluated using:

- Correlation matrices
- Variance Inflation Factor (VIF)
- Coefficient stability
- Ridge regression
- Lasso regression

Highly correlated variables will not automatically be removed if they contain economically meaningful information.

---

## Time-Series Validation

Random train/test splitting will **not** be the primary validation method.

Financial observations are time-dependent, and random splitting can introduce information leakage between market regimes.

The project will instead use chronological and eventually **walk-forward validation**.

Example:

```text
TRAIN                     TEST
2022 ───────────── 2024 | Jan 2025

TRAIN                         TEST
2022 ───────────────── Jan | Feb 2025

TRAIN                             TEST
2022 ───────────────────── Feb | Mar 2025
```

Results from each out-of-sample period can then be aggregated.

---

## Evaluation Metrics

Models will be evaluated using multiple measures.

### Regression Performance

- Out-of-sample R²
- RMSE
- MAE
- Predicted vs. realized return correlation

### Directional Performance

- Directional accuracy
- Precision
- Recall
- Probability calibration

### Conditional Predictive Performance

Particular attention will be given to model performance when predicted signals are unusually strong.

For example:

\[
P(R_{TMF,t+60}>0 \mid Forecast > +1\sigma)
\]

A model may have limited predictive capability across all observations while providing stronger information during specific market conditions.

### Stability

Performance will also be evaluated across:

- Interest-rate regimes
- High/low volatility environments
- Macroeconomic announcement days
- Non-announcement days
- Different times of day

---

## Data Architecture

```text
data/
│
├── raw/
│   ├── tmf/
│   ├── treasury/
│   ├── futures/
│   ├── cross_asset/
│   └── macro/
│
├── processed/
│   └── aligned_5min.parquet
│
└── features/
    └── tmf_features.parquet
```

Raw datasets will remain unchanged.

Processed datasets contain cleaned and synchronized observations.

Feature datasets contain variables specifically engineered for statistical modeling.

Large market datasets will not be committed directly to GitHub.

---

## Planned Repository Structure

```text
TMF-Intraday-Predictive-Analytics/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── forward_test/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── signals/
│   └── utils/
│
├── notebooks/
│
├── docs/
│   ├── research_design.md
│   ├── hypotheses.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   └── validation_framework.md
│
├── thinkorswim/
│   ├── studies/
│   ├── strategies/
│   ├── scans/
│   └── README.md
│
├── papermoney/
│   ├── exports/
│   └── analysis/
│
├── backtests/
│   └── results/
│
├── models/
│
├── reports/
│
└── tests/
```

---

## Data Leakage Controls

Preventing look-ahead bias is a core requirement of the project.

A predictor may only be used when the information represented by that predictor would have actually been available at the prediction timestamp.

Examples include:

- Future prices cannot influence current features.
- Macroeconomic releases cannot appear before their historical release timestamp.
- Consensus estimates must represent forecasts available before the announcement.
- Rolling statistics must use historical observations only.
- Scaling parameters must be calculated from training data rather than the complete dataset.
- Missing observations cannot be forward-filled using information unavailable at that timestamp.

---

## Normalization

Where appropriate, variables may be standardized using:

\[
Z = \frac{x-\mu}{\sigma}
\]

Normalization parameters used for predictive modeling will be estimated from the training period only.

This prevents future observations from influencing historical transformations.

---

## Project Roadmap

- [x] Define project objective
- [x] Define initial research question
- [x] Define hypothesis framework
- [x] Define prediction horizons
- [ ] Identify data sources
- [ ] Build raw-data acquisition pipeline
- [ ] Create data dictionary
- [ ] Collect initial TMF and TLT data
- [ ] Collect Treasury-market data
- [ ] Collect cross-asset data
- [ ] Integrate macroeconomic event data
- [ ] Clean and synchronize timestamps
- [ ] Perform data-quality validation
- [ ] Engineer initial features
- [ ] Generate forward-return targets
- [ ] Perform exploratory analysis
- [ ] Perform lead-lag analysis
- [ ] Test individual predictor capability
- [ ] Build OLS baseline
- [ ] Test multicollinearity
- [ ] Implement walk-forward validation
- [ ] Compare regularized models
- [ ] Evaluate nonlinear models
- [ ] Test model stability across regimes
- [ ] Develop reporting/dashboard layer if predictive capability is supported

---

## Technology Stack

Initial development will primarily use:

- Python
- pandas
- NumPy
- statsmodels
- scikit-learn
- Matplotlib
- Jupyter
- Git / GitHub

Additional technologies may be introduced as the project develops.

Potential later components include:

- FastAPI
- PostgreSQL
- Power BI
- Automated ETL pipelines
- Scheduled market-data ingestion

---

## Research Principles

This project follows several core principles:

**Reproducibility**  
Every transformation from raw data to model output should be reproducible.

**No Look-Ahead Bias**  
Models must only use information that existed at the prediction timestamp.

**Out-of-Sample Validation**  
Predictive relationships must survive data that was not used to estimate the model.

**Interpretability Before Complexity**  
Simple statistical models will establish baselines before more complex machine-learning approaches are introduced.

**Economic and Statistical Significance**  
Statistical significance alone does not establish practical predictive value.

**Evidence Over Assumption**  
Candidate relationships will be tested rather than assumed.

---

## Disclaimer

This repository is a quantitative research and educational project.

Statistical relationships, forecasts, model outputs, and backtests do not guarantee future market performance and should not be interpreted as investment advice.

## Research Progress

### Treasury Yield-Curve Research

- **V1:** Selected Treasury yield and regime features established the initial macro benchmark.
- **V2A:** Empirical level, slope, and curvature factors were evaluated using purged walk-forward validation. The V2A curve model achieved a 0.540 average ROC AUC, while adding V2A factors to Technical Baseline V1 reduced AUC from 0.555 to 0.516. V2A was therefore rejected as the primary Treasury representation.
- **V2B:** Next stage will evaluate Nelson-Siegel yield-curve factors.