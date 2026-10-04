"""Chronological dataset splitting for TMF research."""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/features/TMF_1d_technical_features.csv"
)

OUTPUT_DIR = Path("data/processed")

DEVELOPMENT_FILE = (
    OUTPUT_DIR / "TMF_1d_development.csv"
)

TEST_FILE = (
    OUTPUT_DIR / "TMF_1d_out_of_sample.csv"
)

TEST_START_DATE = pd.Timestamp(
    "2026-01-01",
    tz="UTC",
)


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically without shuffling."""

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    development = data[
        data["timestamp"] < TEST_START_DATE
    ].copy()

    test = data[
        data["timestamp"] >= TEST_START_DATE
    ].copy()

    return development, test


def main() -> None:
    """Create development and out-of-sample datasets."""

    print("Loading TMF feature dataset...")

    df = pd.read_csv(INPUT_FILE)

    development, test = chronological_split(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development.to_csv(
        DEVELOPMENT_FILE,
        index=False,
    )

    test.to_csv(
        TEST_FILE,
        index=False,
    )

    print()
    print("Chronological split complete.")
    print()

    print(
        f"Development observations: "
        f"{len(development):,}"
    )

    print(
        f"Out-of-sample observations: "
        f"{len(test):,}"
    )

    print()

    print(
        "Development period:",
        development["timestamp"].min(),
        "→",
        development["timestamp"].max(),
    )

    print(
        "Out-of-sample period:",
        test["timestamp"].min(),
        "→",
        test["timestamp"].max(),
    )


if __name__ == "__main__":
    main()
    