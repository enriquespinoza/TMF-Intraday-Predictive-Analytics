"""Nelson-Siegel yield-curve features for TMF V2B research."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Treasury maturity grid
# ---------------------------------------------------------

MATURITY_YEARS = {
    "yield_3m": 0.25,
    "yield_6m": 0.50,
    "yield_1y": 1.00,
    "yield_2y": 2.00,
    "yield_3y": 3.00,
    "yield_5y": 5.00,
    "yield_7y": 7.00,
    "yield_10y": 10.00,
    "yield_20y": 20.00,
    "yield_30y": 30.00,
}

YIELD_COLUMNS = list(
    MATURITY_YEARS.keys()
)

MATURITIES = np.array(
    list(MATURITY_YEARS.values()),
    dtype=float,
)


# Diebold-Li commonly uses lambda = 0.0609 when maturity
# is expressed in months.
#
# Our maturity grid is expressed in years, so convert the
# decay parameter to an equivalent per-year value.
NELSON_SIEGEL_LAMBDA = 0.0609 * 12.0


# ---------------------------------------------------------
# Nelson-Siegel loadings
# ---------------------------------------------------------

def nelson_siegel_loadings(
    maturities: np.ndarray,
    decay: float = NELSON_SIEGEL_LAMBDA,
) -> np.ndarray:
    """Return Nelson-Siegel factor loadings.

    Columns correspond to:

    beta_0 -> level
    beta_1 -> slope
    beta_2 -> curvature
    """

    tau = np.asarray(
        maturities,
        dtype=float,
    )

    if np.any(tau <= 0):
        raise ValueError(
            "Nelson-Siegel maturities must be positive."
        )

    if decay <= 0:
        raise ValueError(
            "Nelson-Siegel decay must be positive."
        )

    scaled = decay * tau

    slope_loading = (
        1.0 - np.exp(-scaled)
    ) / scaled

    curvature_loading = (
        slope_loading
        - np.exp(-scaled)
    )

    level_loading = np.ones_like(
        tau
    )

    return np.column_stack(
        [
            level_loading,
            slope_loading,
            curvature_loading,
        ]
    )


# ---------------------------------------------------------
# Single-curve fitting
# ---------------------------------------------------------

def fit_nelson_siegel_curve(
    yields: np.ndarray,
    maturities: np.ndarray = MATURITIES,
    decay: float = NELSON_SIEGEL_LAMBDA,
) -> dict:
    """Fit beta0, beta1, and beta2 to one yield curve."""

    yield_values = np.asarray(
        yields,
        dtype=float,
    )

    maturity_values = np.asarray(
        maturities,
        dtype=float,
    )

    if len(yield_values) != len(
        maturity_values
    ):
        raise ValueError(
            "Yield and maturity arrays must have "
            "the same length."
        )

    if np.isnan(yield_values).any():
        raise ValueError(
            "Yield curve contains missing values."
        )

    loadings = nelson_siegel_loadings(
        maturity_values,
        decay=decay,
    )

    betas, _, _, _ = np.linalg.lstsq(
        loadings,
        yield_values,
        rcond=None,
    )

    fitted_yields = loadings @ betas

    residuals = (
        yield_values
        - fitted_yields
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    return {
        "ns_beta0_level": float(
            betas[0]
        ),
        "ns_beta1_slope": float(
            betas[1]
        ),
        "ns_beta2_curvature": float(
            betas[2]
        ),
        "ns_fit_rmse": rmse,
    }


# ---------------------------------------------------------
# Full Treasury dataset
# ---------------------------------------------------------

def build_nelson_siegel_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit Nelson-Siegel factors to Treasury observations."""

    data = df.copy()

    missing_columns = [
        column
        for column in YIELD_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing Treasury yield columns: "
            + ", ".join(missing_columns)
        )

    # V2B requires the complete maturity grid.
    data = (
        data
        .dropna(
            subset=YIELD_COLUMNS
        )
        .copy()
    )

    fitted = data[
        YIELD_COLUMNS
    ].apply(
        lambda row: pd.Series(
            fit_nelson_siegel_curve(
                row.to_numpy(
                    dtype=float
                )
            )
        ),
        axis=1,
    )

    data = pd.concat(
        [
            data,
            fitted,
        ],
        axis=1,
    )

    # ---------------------------------------------
    # Dynamic Nelson-Siegel factor features
    # ---------------------------------------------

    factor_columns = [
        "ns_beta0_level",
        "ns_beta1_slope",
        "ns_beta2_curvature",
    ]

    for factor in factor_columns:

        for horizon in [
            1,
            5,
            20,
        ]:

            # Betas are expressed in percentage points.
            # Convert changes to basis points.
            data[
                f"{factor}_change_{horizon}d_bp"
            ] = (
                data[factor]
                .diff(horizon)
                * 100
            )

        # 20-observation volatility of daily factor
        # changes, expressed in basis points.
        daily_change_bp = (
            data[factor]
            .diff(1)
            * 100
        )

        data[
            f"{factor}_volatility_20d"
        ] = (
            daily_change_bp
            .rolling(20)
            .std()
        )

    return data