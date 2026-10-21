
"""Download validated TMF price-history snapshots from Schwab."""

from importlib.resources import path
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from schwab.auth import easy_client


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "schwab"


def create_client():
    load_dotenv(ROOT / ".env")

    key = os.getenv("SCHWAB_CLIENT_ID")
    secret = os.getenv("SCHWAB_CLIENT_SECRET")
    callback = os.getenv("SCHWAB_REDIRECT_URI")

    if not all((key, secret, callback)):
        raise RuntimeError("Missing Schwab environment settings")

    token_path = Path(
        os.getenv("SCHWAB_TOKEN_PATH")
        or (
            Path(os.environ["LOCALAPPDATA"])
            / "TMF-Intraday-Predictive-Analytics"
            / "schwab_token.json"
        )
    )

    token_path.parent.mkdir(parents=True, exist_ok=True)

    return easy_client(
        api_key=key,
        app_secret=secret,
        callback_url=callback,
        token_path=str(token_path),
    )


def normalize_candles(payload):
    candles = payload.get("candles", [])

    if not candles:
        raise ValueError("Schwab returned no candles")

    df = pd.DataFrame(candles)

    required = [
        "datetime", "open", "high",
        "low", "close", "volume"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing candle fields: {missing}")

    df = df[required].copy()

    df["timestamp_utc"] = pd.to_datetime(
        df.pop("datetime"),
        unit="ms",
        utc=True,
    )

    df = df.sort_values("timestamp_utc")

    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("Missing OHLCV values")

    if (
        (df[["open", "high", "low", "close"]] <= 0).any().any()
        or (df["volume"] < 0).any()
        or (df["high"] < df[["open", "close", "low"]].max(axis=1)).any()
        or (df["low"] > df[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError("Invalid OHLCV values")

    df["market_date"] = (
        df["timestamp_utc"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    return df[
        [
            "timestamp_utc", "market_date",
            "open", "high", "low", "close", "volume"
        ]
    ]


def main():
    client = create_client()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=35)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_id = end.strftime("%Y%m%dT%H%M%SZ")

    for interval, method in [
        ("1d", client.get_price_history_every_day),
        ("5m", client.get_price_history_every_five_minutes),
    ]:
        print(f"\nDownloading TMF {interval} history...")

        response = method(
            "TMF",
            start_datetime=start,
            end_datetime=end,
            need_extended_hours_data=False,
        )

        response.raise_for_status()
        df = normalize_candles(response.json())

        if interval == "5m":
            local_times = df["timestamp_utc"].dt.tz_convert(
                "America/New_York"
            )
            minute_of_day = (
                local_times.dt.hour * 60
                + local_times.dt.minute
            )

            # Retain regular-session candle start times.
            df = df.loc[
                (minute_of_day >= 570)
                & (minute_of_day < 960)
            ].copy()

        if df.empty:
            raise ValueError(f"No valid {interval} candles")

        path = OUTPUT_DIR / f"TMF_{interval}_{snapshot_id}.csv"

        with path.open("x", newline="", encoding="utf-8") as handle:
            df.to_csv(handle, index=False)

        print("Rows:", len(df))
        print("First market date:", df["market_date"].iloc[0])
        print("Last market date:", df["market_date"].iloc[-1])
        print("Latest close:", df["close"].iloc[-1])
        print("Saved:", path)

    print("\nSchwab TMF history download complete.")


if __name__ == "__main__":
    main()
