"""Run the TMF directional-score forward-return analysis."""

from pathlib import Path

import pandas as pd

from src.analysis.forward_returns import run_forward_return_analysis
from src.data.load_market_data import load_ohlcv_csv


INPUT_FILE = Path("data/raw/TMF_1d.csv")

OUTPUT_DIR = Path("data/forward_test")

SCORED_OUTPUT = OUTPUT_DIR / "TMF_1d_scored.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "TMF_1d_forward_summary.csv"


def main() -> None:
    """Run the complete daily TMF forward-return experiment."""

    print("Loading TMF market data...")

    tmf_data = load_ohlcv_csv(INPUT_FILE)

    print(f"Loaded {len(tmf_data):,} observations.")

    print("Calculating directional scores and forward returns...")

    scored, summary = run_forward_return_analysis(tmf_data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_csv(
        SCORED_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("Forward-return analysis complete.")
    print()
    print(summary.to_string(index=False))

    print()
    print(f"Scored dataset saved to: {SCORED_OUTPUT}")
    print(f"Summary saved to:        {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()