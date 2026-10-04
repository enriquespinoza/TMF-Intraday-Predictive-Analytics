"""HAC/Newey-West inference for Treasury regimes and TMF returns."""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from src.analysis.treasury_regimes import (
    HORIZONS,
    add_treasury_regime,
)


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_treasury_features.csv"
)

OUTPUT_FILE = Path(
    "reports/TMF_treasury_regime_inference.csv"
)

DEVELOPMENT_END = pd.Timestamp("2025-12-31")


def estimate_regime_effect(
    df: pd.DataFrame,
    regime: str,
    horizon: int,
) -> dict:
    """Estimate regime effect using HAC/Newey-West standard errors."""

    target = f"forward_return_{horizon}"

    sample = df[
        [target, "treasury_regime"]
    ].dropna().copy()

    sample["regime_indicator"] = (
        sample["treasury_regime"] == regime
    ).astype(int)

    y = sample[target]

    X = sm.add_constant(
        sample["regime_indicator"]
    )

    model = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": horizon
        },
    )

    coefficient = model.params[
        "regime_indicator"
    ]

    standard_error = model.bse[
        "regime_indicator"
    ]

    t_stat = model.tvalues[
        "regime_indicator"
    ]

    p_value = model.pvalues[
        "regime_indicator"
    ]

    conf_low = (
        coefficient
        - 1.96 * standard_error
    )

    conf_high = (
        coefficient
        + 1.96 * standard_error
    )

    return {
        "treasury_regime": regime,
        "horizon_days": horizon,
        "observations": len(sample),
        "regime_observations": int(
            sample["regime_indicator"].sum()
        ),
        "coefficient": coefficient,
        "hac_standard_error": standard_error,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_95_low": conf_low,
        "ci_95_high": conf_high,
    }


def run_regime_inference(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run HAC inference for each Treasury regime."""

    regimes = sorted(
        regime
        for regime in df[
            "treasury_regime"
        ].dropna().unique()
        if regime != "unclassified"
    )

    results = []

    for regime in regimes:

        for horizon in HORIZONS:

            result = estimate_regime_effect(
                df,
                regime,
                horizon,
            )

            results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    """Run development-sample Treasury regime inference."""

    print(
        "Loading TMF + Treasury dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    development = df[
        df["market_date"] <= DEVELOPMENT_END
    ].copy()

    development = add_treasury_regime(
        development
    )

    print(
        f"Development observations: "
        f"{len(development):,}"
    )

    results = run_regime_inference(
        development
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("BEAR FLATTENING — HAC INFERENCE")
    print()

    bear_flattening = results[
        results["treasury_regime"]
        == "bear_flattening"
    ]

    columns = [
        "horizon_days",
        "regime_observations",
        "coefficient",
        "hac_standard_error",
        "t_stat",
        "p_value",
        "ci_95_low",
        "ci_95_high",
    ]

    print(
        bear_flattening[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()