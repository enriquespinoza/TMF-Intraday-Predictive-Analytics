"""Select the last completed NYSE session without backdating forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandas_market_calendars as mcal


NY = "America/New_York"
FINALIZATION_DELAY = pd.Timedelta(minutes=30)
CALENDAR = mcal.get_calendar("NYSE")


@dataclass(frozen=True)
class SessionWindow:
    status: str
    session: date | None
    now_utc: pd.Timestamp
    finalized_at_utc: pd.Timestamp | None
    next_open_utc: pd.Timestamp | None


def resolve_market_session(now=None) -> SessionWindow:
    """Return READY only between close+30 minutes and next NYSE open.

    A READY origin may be yesterday or Friday, but cannot already have
    been followed by an opened NYSE trading session. All comparisons
    use the exchange's actual UTC calendar, including DST/early closes.
    """
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        now = pd.Timestamp(now)
        if pd.isna(now) or now.tzinfo is None:
            raise ValueError("now must be a timezone-aware timestamp")
        now = now.tz_convert("UTC")

    local_day = now.tz_convert(NY).date()
    schedule = CALENDAR.schedule(
        start_date=local_day - pd.Timedelta(days=21),
        end_date=local_day + pd.Timedelta(days=21),
    )
    if schedule.empty:
        raise RuntimeError("NYSE calendar returned no sessions")

    # If today's exchange close occurred but data has not finalized,
    # report NOT_READY instead of attempting yesterday's expired run.
    pending = schedule.loc[
        (schedule["market_close"] <= now)
        & (now < schedule["market_close"] + FINALIZATION_DELAY)
    ]
    if not pending.empty:
        item = pending.iloc[-1]
        return SessionWindow(
            "NOT_READY", pending.index[-1].date(), now,
            item["market_close"] + FINALIZATION_DELAY, None,
        )

    completed = schedule.loc[
        schedule["market_close"] + FINALIZATION_DELAY <= now
    ]
    if completed.empty:
        return SessionWindow("NOT_READY", None, now, None, None)

    last_day = completed.index[-1]
    finalized_at = (
        completed.iloc[-1]["market_close"] + FINALIZATION_DELAY
    )
    next_sessions = schedule.loc[schedule.index > last_day]
    if next_sessions.empty:
        raise RuntimeError("NYSE calendar cannot resolve next market open")

    next_open = next_sessions.iloc[0]["market_open"]
    status = "READY" if now < next_open else "MISSED_WINDOW"
    return SessionWindow(
        status, last_day.date(), now, finalized_at, next_open,
    )


def require_issuance_window(session, now=None) -> SessionWindow:
    """Fail closed if a forecast origin is not currently prospective."""
    window = resolve_market_session(now=now)
    if window.status != "READY" or str(window.session) != str(session):
        raise ValueError(
            f"{window.status}: prospective post-close window is not "
            f"open for origin {session}; current origin={window.session}"
        )
    return window
