"""Treasury yield-curve feature engineering for TMF research."""

import pandas as pd


YIELD_COLUMNS = [
    "yield_2y",
    "yield_5y",
    "yield_10y",
    "yield_30y",
]


def build_treasury_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Treasury yield, change, curve, and volatility features.

    Features are calculated over valid Treasury observations rather
    than calendar rows containing missing FRED observations.
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
    # diff(5) means five Treasury observations ago,
    # not five calendar rows ago.
    # ---------------------------------------------------------

    for maturity in [
        "2y",
        "5y",
        "10y",
        "30y",
    ]:

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
    # Yield curve slopes in basis points.
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
    # Curve changes.
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
    # Treasury volatility.
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

    return data