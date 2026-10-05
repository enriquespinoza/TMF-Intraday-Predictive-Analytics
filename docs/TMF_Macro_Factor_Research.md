# TMF Macro Factor Research

## Research Objective

This document defines the macroeconomic research framework for the
TMF Intraday Predictive Analytics project.

The objective is to identify economic variables that may provide
incremental information about future movements in long-duration
U.S. Treasury yields and, consequently, TMF.

The purpose of this document is hypothesis formation and feature
pre-specification.

It is NOT intended to optimize features against the existing
2023-2025 validation results.

The 2026 dataset remains locked as the final out-of-sample evaluation
period.

---

# 1. Economic Transmission Framework

TMF seeks approximately 300% of the DAILY performance of the
ICE U.S. Treasury 20+ Year Bond Index.

Therefore, the primary economic transmission mechanism is:

Economic Information
        ↓
Expectations for Inflation / Growth / Monetary Policy
        ↓
Expected Future Short-Term Interest Rates
        +
Term Premium
        ↓
Long-Term Treasury Yields
        ↓
Long-Duration Treasury Prices
        ↓
TMF

This means macroeconomic indicators should not be interpreted simply
as "good economic data" or "bad economic data."

Instead, the important question is:

> How does new economic information change expectations for future
> interest rates, inflation, growth, monetary policy, and the term
> premium?

---

# 2. Bond-Yield Decomposition

Research on long-term government bond yields suggests the following
conceptual decomposition:

Long-Term Treasury Yield
≈
Expected Future Short-Term Rates
+
Term Premium

The expected short-rate component is influenced by:

- Current monetary policy
- Expected Federal Reserve policy
- Inflation
- Inflation expectations
- Employment conditions
- Economic growth expectations

The term-premium component may be influenced by:

- Inflation uncertainty
- Interest-rate uncertainty
- Risk aversion
- Financial instability
- Demand for safe assets
- Treasury supply
- Quantitative easing
- Quantitative tightening

This decomposition will guide macroeconomic feature construction.

---

# 3. Macro Factor Architecture

The macroeconomic information layer will initially be divided into
five economically distinct groups.

## Factor 1 — Inflation

Candidate indicators:

- Consumer Price Index (CPI)
- Core CPI
- Personal Consumption Expenditures Price Index (PCE)
- Core PCE
- Producer Price Index (PPI)
- Market-based inflation expectations
- Survey-based inflation expectations

Potential derived features:

- Year-over-year inflation
- Month-over-month inflation
- Three-month annualized inflation
- Six-month annualized inflation
- Inflation momentum
- Inflation acceleration
- Actual minus consensus expectation
- Inflation surprise z-score
- Change in long-term inflation expectations

### Hypothesis

Positive inflation surprises should generally place upward pressure
on Treasury yields because they can increase expected future policy
rates and/or inflation compensation.

For TMF:

Higher inflation surprise
→ potentially higher long-term yields
→ potentially lower long-duration Treasury prices
→ potentially lower TMF

The effect may depend on the inflation regime and which part of the
yield curve responds.

---

# 4. Labor Market Factor

Candidate indicators:

- Nonfarm payrolls
- Unemployment rate
- Initial unemployment claims
- Continuing claims
- Average hourly earnings
- Average weekly hours
- Labor-force participation
- Job openings
- Hiring / quits data

Potential derived features:

- Payroll surprise
- Payroll three-month moving average
- Payroll momentum
- Change in unemployment
- Claims four-week moving average
- Claims acceleration
- Wage-growth momentum
- Hours-worked trend
- Composite labor-strength score

### Hypothesis

A stronger-than-expected labor market can increase expectations for
restrictive monetary policy.

Strong labor data
→ higher expected policy path
→ potentially higher Treasury yields
→ potentially lower TMF

Weakening labor conditions may produce the opposite effect.

However, severe labor deterioration may also increase risk aversion
and safe-haven demand for Treasury securities.

Therefore, labor-market relationships may be nonlinear.

---

# 5. Growth and Economic Activity Factor

Candidate indicators:

- Retail sales
- Industrial production
- Real GDP
- ISM Manufacturing PMI
- ISM Services PMI
- Durable goods orders
- Housing indicators
- Consumer spending
- Business investment

The existing U.S. Retail Sales Analytics project will serve as one
source for this factor.

Retail sales should NOT initially be treated as an independent TMF
forecasting model.

Instead, it should represent one component of broader economic
activity.

Potential derived features:

- Retail-sales YoY growth
- Retail-sales momentum
- Retail-sales breadth
- Retail-sales forecast
- Retail-sales forecast error
- Industrial-production momentum
- PMI level
- PMI change
- GDP growth
- Growth surprise
- Composite growth factor

### Hypothesis

Stronger economic growth can increase:

1. Expected future policy rates
2. Expected real interest rates
3. Inflation expectations

These channels may place upward pressure on Treasury yields.

Growth deterioration may reduce expected future rates and therefore
support long-duration Treasury prices.

---

# 6. Monetary Policy Factor

Candidate variables:

- Effective Federal Funds Rate
- Federal Reserve target range
- FOMC decisions
- Federal Reserve balance sheet
- Quantitative easing
- Quantitative tightening
- Market-implied future policy rates
- FOMC communication

Future extensions may include:

- Fed speech NLP
- FOMC statement NLP
- FOMC minutes NLP
- Hawkish/dovish language scores

### Hypothesis

The important variable is not necessarily the announced policy rate.

Financial markets attempt to price policy before the announcement.

Therefore:

Policy Surprise
=
Actual Policy Outcome
-
Expected Policy Outcome

may contain more predictive information than the policy level alone.

The same principle applies to forward guidance.

---

# 7. Yield Curve Factor

The project currently contains Treasury yields for:

- 3 month
- 6 month
- 1 year
- 2 year
- 3 year
- 5 year
- 7 year
- 10 year
- 20 year
- 30 year

Existing research has produced:

- Yield levels
- Yield changes
- Curve spreads
- Curve changes
- Yield volatility
- Empirical level/slope/curvature factors
- Nelson-Siegel factors

Existing experiments:

- V1 Treasury features
- V2A empirical structural curve factors
- V2B Nelson-Siegel factors

The existing results indicate that richer representations of the
yield curve have not automatically improved five-day TMF prediction.

This is an important negative result.

---

# 8. 10-Year Minus 3-Month Term Spread

Yield-curve research identifies the spread between the 10-year
Treasury yield and the 3-month Treasury yield as a particularly
useful indicator of future U.S. economic activity.

Define:

10Y3M Spread
=
10-Year Treasury Yield
-
3-Month Treasury Yield

Research indicates that the LEVEL of the spread may be more
informative for recession forecasting than short-term changes in
the spread.

Potential features:

- 10Y minus 3M spread
- Inversion indicator
- Depth of inversion
- Days/months inverted
- Time since inversion
- Time since uninversion
- Spread percentile
- Persistent inversion indicator

### Important Horizon Limitation

Yield-curve recession forecasting generally operates over horizons
measured in quarters.

The TMF prediction target currently operates over five trading days.

Therefore, recession-prediction information should not automatically
be treated as a five-day trading signal.

The term spread may be more useful as a macroeconomic regime variable.

---

# 9. Term Premium and Interest-Rate Risk

Long-term Treasury yields contain more than expectations for future
Federal Reserve policy.

The term premium represents compensation investors require for
holding long-duration bonds.

Candidate indicators:

- Treasury term-premium estimates
- Interest-rate volatility
- MOVE Index or similar rate-volatility measures
- Inflation uncertainty
- Treasury market liquidity
- Safe-haven demand
- Credit spreads
- Financial stress measures

### Hypothesis

Two periods with identical expected Federal Reserve policy paths can
still have different long-term Treasury yields because the term
premium differs.

Therefore:

Expected Fed Policy
≠
Complete Explanation of 30-Year Treasury Yield

This may be particularly important for TMF because its underlying
exposure is concentrated in long-duration Treasury securities.

---

# 10. Financial Conditions / Risk Factor

Candidate indicators:

- Corporate credit spreads
- High-yield spreads
- Investment-grade spreads
- Financial Conditions Indexes
- Equity volatility
- Interest-rate volatility
- Banking stress indicators
- Liquidity measures

### Hypothesis

Financial stress can affect Treasury yields through several channels.

Financial stress
→ weaker growth expectations
→ lower expected policy rates

and/or:

Financial stress
→ safe-haven Treasury demand
→ lower Treasury yields

Therefore, financial conditions may provide information not fully
captured by traditional economic releases.

---

# 11. Fiscal and Treasury Supply Factor

Candidate indicators:

- Federal budget deficit
- Treasury borrowing requirements
- Treasury auction sizes
- Treasury auction results
- Dealer inventories
- Foreign Treasury holdings
- Treasury issuance by maturity

Potential auction features:

- Bid-to-cover ratio
- Auction tail
- Indirect bidder participation
- Direct bidder participation
- Dealer take-down

### Hypothesis

Large Treasury supply may increase the yield required by investors,
particularly at longer maturities.

Therefore:

Higher duration supply
→ potentially higher term premium
→ potentially higher long-term yields
→ potentially lower TMF

This channel may operate independently of the economic-growth and
Federal Reserve channels.

---

# 12. Economic Expectations vs. Economic Levels

One of the most important distinctions for this project is between:

Economic Level

and

Economic Surprise

Markets continuously form expectations before official economic
releases.

Therefore, the same reported number can produce different market
reactions depending on expectations.

Example:

Payrolls = +200,000

may be:

Bullish economic surprise if consensus = +100,000

or

Bearish economic surprise if consensus = +300,000.

For short-horizon Treasury forecasting, the surprise component may
be more useful than the raw economic level.

General feature:

Economic Surprise
=
Actual Release
-
Consensus Forecast

Possible standardized version:

Surprise Z-Score
=
(Actual - Consensus)
/
Historical Standard Deviation of Surprise

This should eventually be tested for major scheduled releases.

---

# 13. Point-in-Time Data Requirement

All macroeconomic features must satisfy strict point-in-time rules.

For every observation we should ideally know:

- Reference period
- Initial release timestamp
- Value available at release
- Consensus expectation
- Previous value known at that time
- Revised value
- Revision timestamp

A model must never receive information that was unavailable at the
prediction timestamp.

This is particularly important because many macroeconomic series are
revised after initial publication.

The final historical value cannot automatically be treated as the
value investors knew in real time.

---

# 14. Release Timing

Economic data should not simply be joined to TMF observations by
calendar date.

For each release:

available_timestamp <= prediction_timestamp

must hold.

Example:

If CPI is released at 8:30 AM ET on a particular day, a model making
a prediction before 8:30 AM cannot use that CPI release.

A model making a prediction after the release may use it.

Daily models may require an explicit rule defining whether same-day
releases are considered available.

Intraday models will require exact timestamps.

---

# 15. Slow Factors vs. Fast Factors

The research suggests separating variables according to their
forecasting horizon.

## Slow Macro Regime

Examples:

- Yield-curve level
- Inflation trend
- Labor trend
- Growth trend
- Monetary-policy regime
- Long-term inflation expectations
- Term premium

These variables may describe the economic environment over weeks,
months, or quarters.

## Fast Market Information

Examples:

- Economic release surprises
- Yield changes
- Rate volatility
- Technical momentum
- Relative volume
- Market reactions following releases

These variables may contain information relevant to short-horizon
TMF movements.

The architecture should therefore distinguish:

Macro Regime
        ↓
Rates Regime
        ↓
Fast Market Signal
        ↓
TMF Forecast

rather than assuming every macro variable should directly predict
five-day TMF returns.

---

# 16. Proposed Model Architecture

A future model may use a layered architecture.

## Layer 1 — Macro Regime

Inputs:

- Inflation
- Labor
- Growth
- Monetary policy
- Yield curve
- Financial conditions

Output:

Economic regime representation

Examples:

- Inflationary expansion
- Disinflationary expansion
- Growth slowdown
- Recession risk
- Policy tightening
- Policy easing

## Layer 2 — Rates Regime

Inputs:

- Treasury yield curve
- Nelson-Siegel factors
- Term premium
- Rate volatility
- Treasury supply

Output:

Long-duration Treasury environment

## Layer 3 — Market Signal

Inputs:

- TMF technical features
- Treasury price/yield momentum
- Economic surprises
- Event indicators

Output:

Short-horizon directional probability or expected return

---

# 17. Current Evidence from the TMF Project

Technical Baseline V1 remains the frozen benchmark.

The existing five-day technical logistic model demonstrated weak but
relatively persistent ranking information across the 2023, 2024,
and 2025 validation periods.

Treasury V1, V2A structural curve factors, and V2B Nelson-Siegel
factors are being evaluated as separate information layers.

V2A showed that adding additional yield-curve complexity does not
automatically improve short-horizon prediction.

This result supports a disciplined approach to future macroeconomic
feature development.

---

# 18. Research Rules

The following rules apply to future macroeconomic experiments.

1. Economic hypotheses must be documented before predictive testing.

2. Feature sets should be economically motivated rather than selected
   solely from historical correlation.

3. 2026 remains locked until final model evaluation.

4. Existing 2023-2025 validation folds should not be repeatedly used
   for unrestricted feature selection.

5. Release timestamps must be respected.

6. Revised economic data must not replace historical information that
   would not have been known in real time.

7. Economic surprise should be distinguished from economic level.

8. Slow macro regime information should be distinguished from
   short-horizon trading signals.

9. Negative experimental results must be preserved.

10. Model complexity should increase only when simpler representations
    fail for identifiable reasons.

---

# 19. Initial Research Priority

Based on the economic mechanisms identified in the research, the
initial priority order is:

1. Inflation and inflation expectations
2. Labor-market conditions
3. Monetary-policy expectations
4. Growth and economic activity
5. Term premium / rate volatility
6. Financial conditions
7. Fiscal policy / Treasury supply

Retail sales belongs primarily within the Growth and Economic
Activity factor.

The existing Retail Sales Analytics project can therefore become an
input into the macroeconomic layer rather than an isolated TMF
prediction system.

---
## Real Yields, Inflation Compensation, and Treasury Data

### Motivation

Nominal Treasury yields combine information about real interest rates and
inflation compensation. For a long-duration Treasury instrument such as TMF,
changes in these components may contain more useful information than changes
in nominal yields alone.

A simplified decomposition is:

Nominal Treasury Yield ≈ Real Treasury Yield + Inflation Compensation

This motivates separating movements in long-term nominal Treasury yields into
real-rate and inflation-compensation components.

### Primary Data Sources

The U.S. Treasury and Federal Reserve H.15 release provide daily nominal
constant-maturity Treasury yields across the maturity curve.

Nominal maturities include:

- 1M
- 3M
- 6M
- 1Y
- 2Y
- 3Y
- 5Y
- 7Y
- 10Y
- 20Y
- 30Y

Inflation-indexed (TIPS) constant-maturity yields are available for:

- 5Y
- 7Y
- 10Y
- 20Y
- 30Y

The 20Y and 30Y real yields are particularly relevant to TMF because its
underlying index consists of long-duration U.S. Treasury securities.

### Candidate Decomposition

For maturity m:

inflation_compensation_m =
    nominal_yield_m - real_yield_m

Candidate features:

real_yield_5y
real_yield_10y
real_yield_20y
real_yield_30y

real_yield_20y_change_1d_bp
real_yield_20y_change_5d_bp
real_yield_30y_change_1d_bp
real_yield_30y_change_5d_bp

inflation_compensation_5y
inflation_compensation_10y
inflation_compensation_20y
inflation_compensation_30y

inflation_compensation_20y_change_5d_bp
inflation_compensation_30y_change_5d_bp

These variables are research candidates only. The final predictive feature
set will be pre-specified before validation.

### Important Interpretation

Nominal yield changes should not automatically be interpreted as inflation
changes.

For example:

Rising nominal yield + rising real yield
    -> tightening real financial conditions

Rising nominal yield + rising inflation compensation
    -> increasing inflation compensation

Falling nominal yield + falling real yield
    -> easing real-rate environment

Falling nominal yield + falling inflation compensation
    -> declining inflation compensation

These regimes may have different implications for long-duration Treasury
returns even when the nominal yield movement is similar.

### Data Methodology

Treasury constant-maturity rates are interpolated from daily yield curves
rather than representing the yield of a single security with exactly the
specified maturity.

Treasury changed its yield-curve methodology from a quasi-cubic Hermite
spline methodology to a monotone-convex methodology in December 2021.

Treasury reports that the methodology change had little effect on nominal
CMT rates but somewhat larger effects on real CMT rates.

Because the TMF research sample begins shortly before this methodology
transition, the change will be documented as a potential data-regime issue.

The Treasury-provided HS-versus-MC comparison dataset will be retained for
methodological validation rather than used as a predictive feature source.

### Research Hypothesis

H-MACRO-REAL-1:

Changes in long-duration real Treasury yields contain incremental information
about subsequent TMF returns beyond nominal Treasury yield changes alone.

H-MACRO-INFCOMP-1:

Changes in long-duration inflation compensation contain information distinct
from real-yield changes and may help identify the economic source of nominal
long-rate movements.

H-MACRO-DECOMP-1:

The combination of real-yield direction and inflation-compensation direction
provides a more economically meaningful Treasury regime classification than
nominal yield direction alone.

# 20. Proposed V3 Research Roadmap

## V3.1 — Inflation Layer

Research and construct point-in-time inflation features.

Do not run predictive optimization yet.

## V3.2 — Labor Layer

Research employment and labor-market indicators.

Construct point-in-time features.

## V3.3 — Growth Layer

Integrate the existing Retail Sales Analytics project.

Investigate industrial production, PMI, and other activity measures.

## V3.4 — Monetary Policy Layer

Construct policy-rate and policy-expectation features.

Later extend to Federal Reserve communication.

## V3.5 — Term Premium / Financial Conditions

Investigate whether long-duration risk information explains TMF
behavior not captured by the yield curve.

## V3.6 — Macro Regime Representation

Combine the pre-specified economic factors into a small number of
interpretable regime variables.

## V3.7 — Predictive Evaluation

Only after the macro information set has been defined should it be
tested against the frozen Technical Baseline V1 using the existing
purged chronological validation framework.

2026 remains locked.

---

# 21. Research Sources

The macro-factor framework is motivated by research including:

- Arturo Estrella and Mary R. Trubin,
  "The Yield Curve as a Leading Indicator: Some Practical Issues."

- Arturo Estrella,
  "The Yield Curve as a Leading Indicator: Frequently Asked
  Questions."

- Arturo Estrella and Frederic S. Mishkin,
  "The Yield Curve as a Predictor of U.S. Recessions."

- "What Drives Bond Yields?"

- Research examining inflation, growth, financial stability,
  monetary policy, quantitative easing/tightening, fiscal deficits,
  and Treasury yields.

These sources emphasize several recurring mechanisms:

- Yield-curve information
- Inflation expectations
- Growth expectations
- Monetary-policy expectations
- Term premia
- Financial risk
- Treasury supply and demand

---

# 22. Core Research Question

The central question for V3 is:

> Can point-in-time macroeconomic information improve our
> understanding of the rates regime surrounding TMF without
> overfitting the short-horizon prediction problem?

The objective is not to maximize historical fit.

The objective is to determine whether macroeconomic information
contains stable, economically interpretable, out-of-sample
information that complements the frozen Technical Baseline V1.

## Literature Synthesis: Decomposing Treasury Yield Risk

The literature indicates that nominal Treasury yields should not be treated
as a single economic state variable.

A useful conceptual decomposition is:

Nominal Yield
    = Real Yield
    + Expected Inflation
    + Inflation Risk Premium

Observed breakeven inflation provides:

Breakeven Inflation
    = Nominal Yield - TIPS Yield

but breakeven inflation should not be interpreted as pure expected inflation
because it can contain inflation-risk and liquidity premia.

Research on Treasury return predictability provides evidence that both
time-varying real-rate risk premia and inflation-risk premia contribute to
predictability in nominal Treasury excess returns.

This suggests that TMF research should distinguish between:

1. Nominal-rate movements
2. Real-rate movements
3. Inflation compensation
4. Real term spreads
5. Inflation-compensation term spreads
6. Term/risk-premium conditions
7. Liquidity conditions

rather than treating a change in the nominal 20Y or 30Y yield as a single
homogeneous macroeconomic shock.

### Implication for TMF Modeling

V2A and V2B tested Treasury yield-curve information primarily as direct
predictors of five-day TMF direction.

The literature suggests an additional hypothesis: Treasury macro variables
may be more informative as state or regime variables that condition the
predictive relationship between technical signals and subsequent TMF returns.

Therefore, the next macro research stage should test economic state
classification before adding another unrestricted set of predictors to the
Technical V1 model.