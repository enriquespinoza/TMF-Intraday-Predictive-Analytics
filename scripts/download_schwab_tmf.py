
"""Download validated TMF price-history snapshots from Schwab."""

from importlib.resources import path
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from schwab.auth import easy_client


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "schwab"
HISTORY = ROOT / "data" / "raw" / "TMF_1d.csv"
MIN_DAILY_OVERLAP = 5
INTRADAY_LOOKBACK_DAYS = 35


def resolve_token_path(platform_name=None):
    """Resolve a persistent Schwab token file location."""
    configured = os.getenv("SCHWAB_TOKEN_PATH")

    if configured:
        token_path = Path(configured).expanduser()

        if not token_path.is_absolute():
            raise RuntimeError(
                "SCHWAB_TOKEN_PATH must be absolute"
            )

        return token_path

    platform_name = platform_name or os.name

    if platform_name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA")

        if local_appdata:
            return (
                Path(local_appdata)
                / "TMF-Intraday-Predictive-Analytics"
                / "schwab_token.json"
            )

    raise RuntimeError(
        "SCHWAB_TOKEN_PATH must be configured "
        "for this operating system"
    )


def create_client():
    load_dotenv(ROOT / ".env")

    key = os.getenv("SCHWAB_CLIENT_ID")
    secret = os.getenv("SCHWAB_CLIENT_SECRET")
    callback = os.getenv("SCHWAB_REDIRECT_URI")

    if not all((key, secret, callback)):
        raise RuntimeError("Missing Schwab environment settings")

    token_path = resolve_token_path()

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


def required_history_sessions(history_path=None):
    """Read the frozen historical endpoint; reject malformed session dates."""
    history_path = HISTORY if history_path is None else Path(history_path)
    history = pd.read_csv(history_path, usecols=["timestamp"])
    parsed = pd.to_datetime(history["timestamp"], utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("Historical source contains null timestamps")

    dates = parsed.dt.tz_convert("America/New_York").dt.date.tolist()
    if len(dates) < MIN_DAILY_OVERLAP or dates != sorted(set(dates)):
        raise ValueError("Historical source has insufficient or invalid sessions")
    return tuple(dates[-MIN_DAILY_OVERLAP:])


def request_start_times(end, historical_sessions):
    """Keep the same daily overlap as the Schwab intraday window rolls on."""
    if not historical_sessions or len(historical_sessions) != MIN_DAILY_OVERLAP:
        raise ValueError("Exactly five historical reference sessions are required")
    daily_start = datetime.combine(
        historical_sessions[0], time.min,
        tzinfo=ZoneInfo("America/New_York"),
    )
    intraday_start = end - timedelta(days=INTRADAY_LOOKBACK_DAYS)
    return daily_start, intraday_start


def require_daily_overlap(daily, historical_sessions):
    """Fail before writing if Schwab omitted any frozen reference session."""
    missing = sorted(
        set(historical_sessions).difference(set(daily["market_date"]))
    )
    if missing:
        raise ValueError(
            "Schwab daily history missing required overlap sessions: "
            + ", ".join(str(day) for day in missing)
        )


def main():
    client = create_client()

    end = datetime.now(timezone.utc)
    historical_sessions = required_history_sessions()
    daily_start, intraday_start = request_start_times(
        end, historical_sessions
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_id = end.strftime("%Y%m%dT%H%M%SZ")

    for interval, method, start in [
        ("1d", client.get_price_history_every_day, daily_start),
        ("5m", client.get_price_history_every_five_minutes, intraday_start),
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

        if interval == "1d":
            require_daily_overlap(df, historical_sessions)

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
