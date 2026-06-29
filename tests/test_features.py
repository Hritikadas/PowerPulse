"""Basic smoke tests for feature engineering."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.engineer import add_time_features


def make_sample_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "hour": [10, 10, 10, 10, 10],
        "mcp":  [4.5, 4.7, 4.2, 5.0, 4.8],
        "mcv":  [100, 110, 95, 120, 105],
        "buy_volume":  [60, 65, 55, 70, 60],
        "sell_volume": [60, 65, 55, 70, 60],
        "exchange": ["IEX"] * 5,
    })


def test_time_features_added():
    df = add_time_features(make_sample_df())
    assert "day_of_week" in df.columns
    assert "is_weekend" in df.columns
    assert "is_peak" in df.columns
    assert df["is_weekend"].isin([0, 1]).all()
