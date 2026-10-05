"""Configuration for V3 macroeconomic data series.

This module defines the macroeconomic series permitted in the V3 research
pipeline. Keeping the series registry separate from downloading and feature
engineering helps preserve the ex-ante research specification.

Important:
    2026 remains locked for final out-of-sample evaluation.
"""

from __future__ import annotations

from typing import Any


MACRO_SERIES: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Labor market: demand
    # ------------------------------------------------------------------
    "nonfarm_payrolls": {
        "series_id": "PAYEMS",
        "name": "Total Nonfarm Payroll Employment",
        "category": "labor_demand",
        "source": "FRED",
        "frequency": "monthly",
        "units": "thousands_of_persons",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "job_openings": {
        "series_id": "JTSJOL",
        "name": "Total Nonfarm Job Openings",
        "category": "labor_demand",
        "source": "FRED",
        "frequency": "monthly",
        "units": "thousands",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "job_openings_rate": {
        "series_id": "JTSJOR",
        "name": "Total Nonfarm Job Openings Rate",
        "category": "labor_demand",
        "source": "FRED",
        "frequency": "monthly",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },

    # ------------------------------------------------------------------
    # Labor market: tightness / slack
    # ------------------------------------------------------------------
    "unemployment_rate": {
        "series_id": "UNRATE",
        "name": "Civilian Unemployment Rate",
        "category": "labor_slack",
        "source": "FRED",
        "frequency": "monthly",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "u6_unemployment_rate": {
        "series_id": "U6RATE",
        "name": "U-6 Unemployment Rate",
        "category": "labor_slack",
        "source": "FRED",
        "frequency": "monthly",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "quits_rate": {
        "series_id": "JTSQUR",
        "name": "Total Nonfarm Quits Rate",
        "category": "labor_tightness",
        "source": "FRED",
        "frequency": "monthly",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "labor_force_participation": {
        "series_id": "CIVPART",
        "name": "Labor Force Participation Rate",
        "category": "labor_supply",
        "source": "FRED",
        "frequency": "monthly",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "part_time_economic_reasons": {
        "series_id": "LNS12032194",
        "name": "Employed Part Time for Economic Reasons",
        "category": "labor_slack",
        "source": "FRED",
        "frequency": "monthly",
        "units": "thousands_of_persons",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },

    # ------------------------------------------------------------------
    # Labor market: hours / wages
    # ------------------------------------------------------------------
    "average_weekly_hours": {
        "series_id": "AWHAETP",
        "name": "Average Weekly Hours of All Employees",
        "category": "labor_hours",
        "source": "FRED",
        "frequency": "monthly",
        "units": "hours",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "average_hourly_earnings": {
        "series_id": "CES0500000003",
        "name": "Average Hourly Earnings of All Employees",
        "category": "wages",
        "source": "FRED",
        "frequency": "monthly",
        "units": "dollars_per_hour",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "employment_cost_index": {
        "series_id": "ECIWAG",
        "name": "Employment Cost Index: Wages and Salaries",
        "category": "wages",
        "source": "FRED",
        "frequency": "quarterly",
        "units": "index",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },

    # ------------------------------------------------------------------
    # Consumer / growth
    # ------------------------------------------------------------------
    "retail_sales": {
        "series_id": "RSAFS",
        "name": "Advance Retail Sales: Retail and Food Services",
        "category": "consumer_growth",
        "source": "FRED",
        "frequency": "monthly",
        "units": "millions_of_dollars",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "industrial_production": {
        "series_id": "INDPRO",
        "name": "Industrial Production Index",
        "category": "growth",
        "source": "FRED",
        "frequency": "monthly",
        "units": "index",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },

    # ------------------------------------------------------------------
    # Inflation
    # ------------------------------------------------------------------
    "cpi": {
        "series_id": "CPIAUCSL",
        "name": "Consumer Price Index",
        "category": "inflation",
        "source": "FRED",
        "frequency": "monthly",
        "units": "index",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "core_cpi": {
        "series_id": "CPILFESL",
        "name": "Core Consumer Price Index",
        "category": "inflation",
        "source": "FRED",
        "frequency": "monthly",
        "units": "index",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "pce_price_index": {
        "series_id": "PCEPI",
        "name": "PCE Price Index",
        "category": "inflation",
        "source": "FRED",
        "frequency": "monthly",
        "units": "index",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },
    "core_pce_price_index": {
        "series_id": "PCEPILFE",
        "name": "Core PCE Price Index",
        "category": "inflation",
        "source": "FRED",
        "frequency": "monthly",
        "units": "index",
        "transformation": "level",
        "requires_release_alignment": True,
        "requires_vintage_tracking": True,
    },

    # ------------------------------------------------------------------
    # Market-based inflation compensation
    # ------------------------------------------------------------------
    "breakeven_5y": {
        "series_id": "T5YIE",
        "name": "5-Year Breakeven Inflation Rate",
        "category": "inflation_compensation",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },
    "breakeven_10y": {
        "series_id": "T10YIE",
        "name": "10-Year Breakeven Inflation Rate",
        "category": "inflation_compensation",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },
    "forward_inflation_5y5y": {
        "series_id": "T5YIFR",
        "name": "5-Year, 5-Year Forward Inflation Expectation Rate",
        "category": "inflation_compensation",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },

    # ------------------------------------------------------------------
    # Real Treasury yields / TIPS
    # ------------------------------------------------------------------
    "real_yield_5y": {
        "series_id": "DFII5",
        "name": "5-Year Treasury Inflation-Indexed Yield",
        "category": "real_rates",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },
    "real_yield_10y": {
        "series_id": "DFII10",
        "name": "10-Year Treasury Inflation-Indexed Yield",
        "category": "real_rates",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },
    "real_yield_20y": {
        "series_id": "DFII20",
        "name": "20-Year Treasury Inflation-Indexed Yield",
        "category": "real_rates",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },
    "real_yield_30y": {
        "series_id": "DFII30",
        "name": "30-Year Treasury Inflation-Indexed Yield",
        "category": "real_rates",
        "source": "FRED",
        "frequency": "daily",
        "units": "percent",
        "transformation": "level",
        "requires_release_alignment": False,
        "requires_vintage_tracking": False,
    },
}


def get_series_by_category(
    category: str,
) -> dict[str, dict[str, Any]]:
    """Return macro series belonging to a specific category."""

    return {
        name: config
        for name, config in MACRO_SERIES.items()
        if config["category"] == category
    }


def get_fred_series_ids() -> dict[str, str]:
    """Return mapping of internal names to FRED series IDs."""

    return {
        name: config["series_id"]
        for name, config in MACRO_SERIES.items()
        if config["source"] == "FRED"
    }