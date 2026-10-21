
"""Experimental five-day TMF forecast using Schwab daily data."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import pandas_market_calendars as mcal

import numpy as np
import pandas as pd

from config.v3_final import (
    DEVELOPMENT_END,
    FINAL_V3_FEATURES,
    FINAL_V3_NAME,
)
from src.features.technical_features import build_technical_features
from src.models.directional_classifier import (
    HORIZON,
    TARGET_RETURN,
    build_logistic_model,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/TMF_1d.csv"
HISTORY = ROOT / "data/features/TMF_1d_technical_PAYEMS_features.csv"
PAYEMS = ROOT / "data/features/PAYEMS_event_features.csv"
SCHWAB_DIR = ROOT / "data/raw/schwab"
OUTPUT_DIR = ROOT / "data/forward_test"
NY = "America/New_York"


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def load_prices():
    pinned = os.environ.get("TMF_PINNED_SNAPSHOT")
    if not pinned:
        raise ValueError(
            "An explicitly validated Schwab snapshot is required"
        )

    snapshot = Path(pinned).resolve()

    if (
        snapshot.parent != SCHWAB_DIR.resolve()
        or not snapshot.name.startswith("TMF_1d_")
        or not snapshot.is_file()
    ):
        raise ValueError("Invalid pinned Schwab snapshot")

    old = pd.read_csv(RAW)
    old["market_date"] = (
        pd.to_datetime(old["timestamp"], utc=True)
        .dt.tz_convert(NY)
        .dt.date
    )

    new = pd.read_csv(snapshot)
    new["market_date"] = (
        pd.to_datetime(new["market_date"]).dt.date
    )

    for name, frame in [("Historical", old), ("Schwab", new)]:
        if frame["market_date"].duplicated().any():
            raise ValueError(f"{name} contains duplicate dates")
        if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise ValueError(f"{name} contains missing OHLCV")

    old = old.sort_values("market_date")
    new = new.sort_values("market_date")

    overlap = old[["market_date", "close"]].merge(
        new[["market_date", "close"]],
        on="market_date",
        suffixes=("_old", "_schwab"),
    )

    if overlap.empty:
        raise ValueError("No overlap to reconcile price providers")

    # TMF dividend: ex-date 2026-09-22, $0.35062.
    # Yahoo historical prices are backward-adjusted before
    # this event; Schwab overlapping prices are unadjusted.
    ex_date = pd.Timestamp("2026-09-22").date()

    before = overlap.loc[
        overlap["market_date"] < ex_date
    ].copy()

    after = overlap.loc[
        overlap["market_date"] >= ex_date
    ].copy()

    if len(after) < 5:
        raise ValueError("Insufficient post-dividend overlap")

    if after["market_date"].max() != old["market_date"].max():
        raise ValueError("Recent historical endpoint not reconciled")

    # Verify the pre-event prices use a stable adjustment.
    if not before.empty:
        adjustment_ratio = (
            before["close_old"] / before["close_schwab"]
        )

        if (
            adjustment_ratio.max()
            - adjustment_ratio.min()
        ) > 1e-5:
            raise ValueError(
                "Pre-dividend adjustment ratio is inconsistent"
            )

        print(
            "Pre-dividend adjustment factor:",
            round(float(adjustment_ratio.median()), 6)
        )

    # Require near-exact agreement after the dividend.
    after_diff = (
        after["close_old"]
        - after["close_schwab"]
    ).abs()

    max_recent_diff = float(after_diff.max())

    print(
        "Post-dividend overlapping sessions:",
        len(after)
    )
    print(
        "Maximum post-dividend close difference:",
        f"${max_recent_diff:.6f}"
    )

    if max_recent_diff > 0.001:
        raise ValueError(
            "Recent Schwab and historical prices disagree"
        )

    print("Dividend-aware reconciliation: PASSED")
    
    last_old_date = old["market_date"].max()
    appended = new.loc[new["market_date"] > last_old_date].copy()

    if appended.empty:
        raise ValueError("No newer Schwab daily candles available")

    appended["timestamp"] = appended["market_date"].astype(str)

    columns = [
        "timestamp", "open", "high",
        "low", "close", "volume", "market_date"
    ]

    combined = pd.concat(
        [old[columns], appended[columns]],
        ignore_index=True,
    ).sort_values("market_date").reset_index(drop=True)

    if combined["market_date"].duplicated().any():
        raise ValueError("Duplicate dates after combining data")

    return combined, snapshot, len(appended)


def latest_payems(as_of_date):
    events = pd.read_csv(PAYEMS)

    release_at = pd.to_datetime(
        events["event_release_date"].astype(str)
        + " "
        + events["event_release_time"].astype(str),
        errors="raise",
    )

    # These event records use New York release times.
    zones = events["event_release_timezone"]
    if zones.ne(NY).any():
        raise ValueError("Unsupported PAYEMS release timezone")

    release_at = release_at.dt.tz_localize(NY)

    market_close = pd.Timestamp(as_of_date).tz_localize(
        NY
    ) + pd.Timedelta(hours=16)

    available = events.loc[release_at <= market_close].copy()

    if available.empty:
        raise ValueError("No available PAYEMS release")

    available["_release_at"] = release_at.loc[available.index]
    available = available.sort_values("_release_at")

    return available.iloc[-1]


def main():
    if HORIZON != 5:
        raise ValueError("Unexpected frozen model horizon")

    prices, snapshot, appended_count = load_prices()

    technical = build_technical_features(prices.copy())

    reference = pd.read_csv(HISTORY, parse_dates=["market_date"])
    reference = reference.sort_values("market_date")

    # Reproduce previously recorded technical features.
    audit_date = reference["market_date"].max().date()
    old_row = reference.loc[
        reference["market_date"].dt.date == audit_date
    ].iloc[-1]

    audit_row = technical.loc[
        technical["market_date"] == audit_date
    ].iloc[-1]

    tech_features = [
        "return_5d",
        "rsi_14",
        "volatility_20d",
        "relative_volume",
        "distance_from_20d_high",
    ]

    for feature in tech_features:
        if not np.isclose(
            float(old_row[feature]),
            float(audit_row[feature]),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                f"Historical feature parity failed: {feature}"
            )

    latest = technical.iloc[-1]
    forecast_date = latest["market_date"]

    expected_origin = os.environ.get("TMF_EXPECTED_SESSION")
    if (
        not expected_origin
        or str(forecast_date) != expected_origin
    ):
        raise ValueError("Forecast origin does not match validated session")

    if pd.Timestamp(forecast_date) <= reference["market_date"].max():
        raise ValueError("Forecast is not newer than frozen features")

    payroll = latest_payems(forecast_date)

    feature_values = {
        name: latest[name] for name in tech_features
    }

    macro_features = [
        "payroll_change_1m",
        "payroll_revision_total",
        "payroll_broad_revision_event",
    ]

    for name in macro_features:
        feature_values[name] = payroll[name]

    features = list(FINAL_V3_FEATURES)

    if set(features) != set(feature_values):
        raise ValueError("Frozen V3 feature mismatch")

    X_latest = pd.DataFrame(
        [{name: feature_values[name] for name in features}]
    ).apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X_latest.to_numpy(dtype=float)).all():
        raise ValueError("Invalid forecast input feature")

    cutoff = pd.Timestamp(DEVELOPMENT_END) + pd.Timedelta(days=1)

    reference["target_end_date"] = (
        reference["market_date"].shift(-HORIZON)
    )

    eligible = (
        (reference["market_date"] < cutoff)
        & (reference["target_end_date"] < cutoff)
        & reference[TARGET_RETURN].notna()
    )

    train = reference.loc[eligible].copy()
    X_train = train[features].apply(
        pd.to_numeric, errors="raise"
    )
    y_train = (train[TARGET_RETURN] > 0).astype(int)

    if y_train.nunique() != 2:
        raise ValueError("Training requires both target classes")

    model = build_logistic_model()
    model.fit(X_train, y_train)

    probability = float(model.predict_proba(X_latest)[0, 1])

    nyse = mcal.get_calendar("NYSE")
    sessions = nyse.valid_days(
        start_date=pd.Timestamp(forecast_date)
        + pd.Timedelta(days=1),
        end_date=pd.Timestamp(forecast_date)
        + pd.Timedelta(days=21),
    )

    if len(sessions) < HORIZON:
        raise ValueError("Insufficient future NYSE sessions")

    target_date = sessions[HORIZON - 1].date()

    issued_at = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for existing in OUTPUT_DIR.glob(
        f"TMF_V3_SCHWAB_{forecast_date}_*.json"
    ):
        previous = json.loads(
            existing.read_text(encoding="utf-8")
        )
        if (
            previous.get("model") == FINAL_V3_NAME
            and previous.get("forecast_origin") == str(forecast_date)
            and previous.get("horizon_trading_days") == HORIZON
        ):
            raise FileExistsError(
                f"Forecast already exists: {existing}"
            )

    result = {
        "model": FINAL_V3_NAME,
        "status": "experimental",
        "issued_at_utc": issued_at.isoformat(),
        "forecast_origin": str(forecast_date),
        "forecast_target_date": str(target_date),
        "horizon_trading_days": HORIZON,
        "tmf_close": float(latest["close"]),
        "probability_positive_return": probability,
        "training_base_rate": float(y_train.mean()),
        "training_rows": len(train),
        "latest_training_date": str(
            train["market_date"].max().date()
        ),
        "payems_release_date": str(payroll["event_release_date"]),
        "feature_values": {
            name: float(X_latest.iloc[0][name])
            for name in features
        },
        "source_snapshot": snapshot.name,
        "source_sha256": file_hash(snapshot),
        "training_features_sha256": file_hash(HISTORY),
        "historical_feature_parity": "passed",
        "order_generated": False,
    }

    filename = (
        f"TMF_V3_SCHWAB_{forecast_date}_"
        f"{FINAL_V3_NAME}_{HORIZON}d.json"
    )
    path = OUTPUT_DIR / filename

    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("TMF V3 — SCHWAB EXPERIMENTAL FORECAST")
    print("=" * 60)
    print("Model:", FINAL_V3_NAME)
    print("Forecast origin:", forecast_date)
    print("Target date:", target_date)
    print("TMF close:", round(float(latest["close"]), 4))
    print("New Schwab sessions:", appended_count)
    print("Historical feature parity: PASSED")
    print("PAYEMS release:", payroll["event_release_date"])
    print("Training rows:", len(train))
    print(f"Probability TMF rises: {probability:.2%}")
    print(f"Training base rate: {y_train.mean():.2%}")
    print("Research only — no order generated")
    print("Saved:", path)


if __name__ == "__main__":
    main()
