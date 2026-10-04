"""Episode-level Treasury regime analysis for TMF."""

from pathlib import Path

import pandas as pd

from src.analysis.treasury_regimes import (
    HORIZONS,
    add_treasury_regime,
)


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_treasury_features.csv"
)

OUTPUT_FILE = Path(
    "reports/TMF_treasury_regime_episodes.csv"
)

DEVELOPMENT_END = pd.Timestamp(
    "2025-12-31"
)


def identify_regime_episodes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify contiguous Treasury regime episodes."""

    data = df.copy()

    regime_change = (
        data["treasury_regime"]
        != data["treasury_regime"].shift(1)
    )

    data["regime_episode_id"] = (
        regime_change.cumsum()
    )

    data["regime_episode_start"] = (
        data.groupby(
            "regime_episode_id"
        ).cumcount()
        == 0
    )

    return data


def get_episode_starts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return first observation of each Treasury regime episode."""

    data = identify_regime_episodes(df)

    return data[
        data["regime_episode_start"]
    ].copy()


def summarize_episode_returns(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize forward TMF returns from regime episode starts."""

    rows = []

    for regime, group in episodes.groupby(
        "treasury_regime"
    ):

        if regime == "unclassified":
            continue

        row = {
            "treasury_regime": regime,
            "episodes": len(group),
        }

        for horizon in HORIZONS:

            target = f"forward_return_{horizon}"

            sample = group[target].dropna()

            row[f"n_{horizon}d"] = len(sample)

            row[f"mean_return_{horizon}d"] = (
                sample.mean()
            )

            row[f"median_return_{horizon}d"] = (
                sample.median()
            )

            row[f"positive_rate_{horizon}d"] = (
                (sample > 0).mean()
                if len(sample) > 0
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Run episode-level Treasury regime analysis."""

    print(
        "Loading TMF + Treasury dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["market_date"],
    )

    development = df[
        df["market_date"]
        <= DEVELOPMENT_END
    ].copy()

    development = add_treasury_regime(
        development
    )

    episodes = get_episode_starts(
        development
    )

    summary = summarize_episode_returns(
        episodes
    )

    summary = summary.sort_values(
        "mean_return_5d",
        ascending=False,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Development observations: "
        f"{len(development):,}"
    )

    print(
        f"Independent regime episodes: "
        f"{len(episodes):,}"
    )

    print()
    print("EPISODE-LEVEL REGIME RESULTS")
    print()

    columns = [
        "treasury_regime",
        "episodes",
        "mean_return_5d",
        "median_return_5d",
        "positive_rate_5d",
        "mean_return_10d",
        "mean_return_20d",
    ]

    print(
        summary[
            columns
        ].to_string(
            index=False,
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()