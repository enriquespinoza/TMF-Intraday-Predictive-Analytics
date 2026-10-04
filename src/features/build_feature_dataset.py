"""Build the modeling-ready TMF technical feature dataset."""

from pathlib import Path

from src.analysis.forward_returns import add_forward_returns
from src.data.load_market_data import load_ohlcv_csv
from src.features.technical_features import build_technical_features


INPUT_FILE = Path("data/raw/TMF_1d.csv")

OUTPUT_DIR = Path("data/features")
OUTPUT_FILE = OUTPUT_DIR / "TMF_1d_technical_features.csv"


def main() -> None:
    """Build and save the TMF technical feature dataset."""

    print("Loading TMF market data...")

    data = load_ohlcv_csv(INPUT_FILE)

    print(f"Loaded {len(data):,} observations.")

    print("Building technical features...")

    featured = build_technical_features(data)

    print("Adding forward-return targets...")

    featured = add_forward_returns(featured)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    featured.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Feature dataset complete.")
    print(f"Rows:    {len(featured):,}")
    print(f"Columns: {len(featured.columns):,}")
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    