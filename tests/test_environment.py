"""Basic environment and repository configuration tests."""

from config import settings


def test_project_root_exists():
    assert settings.PROJECT_ROOT.exists()


def test_data_directories_exist():
    assert settings.RAW_DATA_DIR.exists()
    assert settings.PROCESSED_DATA_DIR.exists()
    assert settings.FEATURE_DATA_DIR.exists()
    assert settings.FORWARD_TEST_DATA_DIR.exists()


def test_market_timezone():
    assert settings.MARKET_TIMEZONE == "America/New_York"
