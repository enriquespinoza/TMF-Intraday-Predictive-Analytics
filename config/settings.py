"""Project-wide configuration for TMF Intraday Predictive Analytics."""

from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DATA_DIR = DATA_DIR / "features"
FORWARD_TEST_DATA_DIR = DATA_DIR / "forward_test"

# Research directories
MODEL_DIR = PROJECT_ROOT / "models"
BACKTEST_DIR = PROJECT_ROOT / "backtests"
REPORT_DIR = PROJECT_ROOT / "reports"

# Market configuration
MARKET_TIMEZONE = os.getenv("MARKET_TIMEZONE", "America/New_York")

# Application configuration
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Schwab API configuration
SCHWAB_CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID")
SCHWAB_CLIENT_SECRET = os.getenv("SCHWAB_CLIENT_SECRET")
SCHWAB_REDIRECT_URI = os.getenv("SCHWAB_REDIRECT_URI")
