"""Treasury yield-curve feature engineering for TMF research."""

import pandas as pd


# Full Treasury maturity grid used by V2.
YIELD_COLUMNS = [
    "yield_3m",
    "yield_6m",
    "yield_1y",
    "yield_2y",
    "yield_3y",
    "yield_5y",
    "yield_7y",
    "yield_10y",
    "yield_20y",
    "yield_30y",
]

MATURITIES = [
    "3m",
    "6m",
    "1y",
    "2y",
    "3y",
    "5y",
    "7y",
    "10y",
    "20y",
    "30y",
]


def build_treasury_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Treasury yield, change, curve, volatility,
    and V2 structural yield-curve features.

    Features are calculated over valid Treasury observations rather
    than calendar rows containing missing FRED observations.

    Existing V1 Treasury features retain their original definitions.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # Keep only complete Treasury observations.
    # ---------------------------------------------------------

    data = data.dropna(
        subset=YIELD_COLUMNS
    ).copy()

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Yield changes in basis points.
    #
    # diff(5) means five valid Treasury observations ago,
    # not five calendar rows ago.
    # ---------------------------------------------------------

    for maturity in MATURITIES:

        column = f"yield_{maturity}"

        data[
            f"change_{maturity}_1d_bp"
        ] = (
            data[column].diff(1) * 100
        )

        data[
            f"change_{maturity}_5d_bp"
        ] = (
            data[column].diff(5) * 100
        )

        data[
            f"change_{maturity}_20d_bp"
        ] = (
            data[column].diff(20) * 100
        )

    # ---------------------------------------------------------
    # Existing V1 yield-curve slopes.
    #
    # Do not change these definitions. They are retained so V2
    # can be compared directly with the frozen V1 benchmark.
    # ---------------------------------------------------------

    data["curve_10y_2y_bp"] = (
        data["yield_10y"]
        - data["yield_2y"]
    ) * 100

    data["curve_30y_2y_bp"] = (
        data["yield_30y"]
        - data["yield_2y"]
    ) * 100

    data["curve_30y_5y_bp"] = (
        data["yield_30y"]
        - data["yield_5y"]
    ) * 100

    data["curve_30y_10y_bp"] = (
        data["yield_30y"]
        - data["yield_10y"]
    ) * 100

    # ---------------------------------------------------------
    # Existing V1 curve changes.
    # ---------------------------------------------------------

    curve_columns = [
        "curve_10y_2y_bp",
        "curve_30y_2y_bp",
        "curve_30y_5y_bp",
        "curve_30y_10y_bp",
    ]

    for column in curve_columns:

        data[
            f"{column}_change_1d"
        ] = data[column].diff(1)

        data[
            f"{column}_change_5d"
        ] = data[column].diff(5)

        data[
            f"{column}_change_20d"
        ] = data[column].diff(20)

    # ---------------------------------------------------------
    # Existing V1 Treasury volatility.
    #
    # Standard deviation of daily yield changes in basis points.
    # ---------------------------------------------------------

    data[
        "yield_10y_volatility_20d"
    ] = (
        data["change_10y_1d_bp"]
        .rolling(20)
        .std()
    )

    data[
        "yield_30y_volatility_20d"
    ] = (
        data["change_30y_1d_bp"]
        .rolling(20)
        .std()
    )

    # =========================================================
    # V2A STRUCTURAL YIELD-CURVE FEATURES
    # =========================================================

    # ---------------------------------------------------------
    # Level
    #
    # Simple empirical proxy for the overall level of the
    # Treasury curve: equal-weighted mean across all maturities.
    #
    # Units: percentage points.
    # ---------------------------------------------------------

    data["curve_level"] = (
        data[YIELD_COLUMNS].mean(axis=1)
    )

    # ---------------------------------------------------------
    # Slope
    #
    # Long-end minus short-end yield.
    #
    # 30Y - 3M expressed in basis points.
    #
    # Positive = upward-sloping curve.
    # Negative = inverted curve.
    # ---------------------------------------------------------

    data["curve_slope_30y_3m_bp"] = (
        data["yield_30y"]
        - data["yield_3m"]
    ) * 100

    # ---------------------------------------------------------
    # Curvature
    #
    # Simple belly-versus-endpoints measure.
    #
    # 2 * 5Y - 3M - 30Y
    #
    # This is an empirical curvature proxy, not a fitted
    # Nelson-Siegel beta coefficient.
    # ---------------------------------------------------------

    data["curve_curvature_5y_bp"] = (
        (
            2 * data["yield_5y"]
            - data["yield_3m"]
            - data["yield_30y"]
        )
        * 100
    )

    # ---------------------------------------------------------
    # Changes in structural curve factors.
    #
    # Level is stored in percentage points, so multiply its
    # changes by 100 to express them in basis points.
    #
    # Slope and curvature are already expressed in bp.
    # ---------------------------------------------------------

    for horizon in [1, 5, 20]:

        data[
            f"curve_level_change_{horizon}d_bp"
        ] = (
            data["curve_level"].diff(horizon)
            * 100
        )

        data[
            f"curve_slope_change_{horizon}d_bp"
        ] = (
            data["curve_slope_30y_3m_bp"]
            .diff(horizon)
        )

        data[
            f"curve_curvature_change_{horizon}d_bp"
        ] = (
            data["curve_curvature_5y_bp"]
            .diff(horizon)
        )

    # ---------------------------------------------------------
    # Structural-factor volatility.
    #
    # Rolling 20-observation standard deviation of daily
    # factor changes.
    # ---------------------------------------------------------

    data[
        "curve_level_volatility_20d"
    ] = (
        data["curve_level_change_1d_bp"]
        .rolling(20)
        .std()
    )

    data[
        "curve_slope_volatility_20d"
    ] = (
        data["curve_slope_change_1d_bp"]
        .rolling(20)
        .std()
    )

    data[
        "curve_curvature_volatility_20d"
    ] = (
        data["curve_curvature_change_1d_bp"]
        .rolling(20)
        .std()
    )

    return data