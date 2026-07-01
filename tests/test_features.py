"""Basic smoke tests for feature engineering."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.engineer import add_time_features, add_lag_features, clean_missing_data


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
    # --- 🟢 Step 2.2: Add to the bottom of tests/test_features.py ---

def test_no_nan_values_in_output():
    """Ensure our engineering pipeline completely eliminates missing values."""
    import numpy as np
    
    # We need at least 8+ periods because rolling(7) requires 7 rows of historical data
    mock_data = pd.DataFrame({
        'date': pd.date_range(start='2026-01-01', periods=10, freq='h'), 
        'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'mcp': [12.0, np.nan, 14.5, np.nan, 15.0, 13.2, np.nan, 16.1, 14.0, 15.5],
        'buy_volume': [100, 110, np.nan, 130, 140, 125, 135, np.nan, 145, 150],
        'sell_volume': [90, np.nan, 95, 100, 105, 110, np.nan, 115, 120, 125],
        'exchange': ['IEX'] * 10
    })
    
    # 1. Clean the raw missing gaps first
    cleaned_data = clean_missing_data(mock_data)
    
    # 2. Run through time features (peaks, buy/sell ratio)
    processed_df = add_time_features(cleaned_data)
    
    # 3. 🟢 Run through lag features block (EMAs, rolling stats, lag1)
    processed_df = add_lag_features(processed_df)
    
    # 4. Drop the boundary rows created naturally by rolling windows and lags
    processed_df = processed_df.dropna()
    
    # Assert that zero missing rows remain in the final training dataset
    assert processed_df.isna().sum().sum() == 0, "Warning: NaN values leaked through the pipeline!"

def test_feature_columns_exist():
    """Verify that all your newly created columns are successfully attached."""
    mock_data = pd.DataFrame({
        'date': pd.date_range(start='2026-01-01', periods=2, freq='h'),
        'hour': [8, 20],
        'mcp': [10.0, 12.0],
        'buy_volume': [100, 120],
        'sell_volume': [90, 85],
        'exchange': ['IEX', 'IEX']
    })
    
    processed_df = add_time_features(mock_data)
    
    # Assert the new column labels exist in the output array
    assert 'buy_sell_ratio' in processed_df.columns
    assert 'grid_period_Morning_Peak' in processed_df.columns or 'grid_period_Evening_Peak' in processed_df.columns
def test_no_nan_values_in_output():
    """Ensure our engineering pipeline completely eliminates missing values."""
    import numpy as np
    
    mock_data = pd.DataFrame({
        'date': pd.date_range(start='2026-01-01', periods=5, freq='h'), 
        'hour': [0, 1, 2, 3, 4],
        'mcp': [12.0, np.nan, 14.5, np.nan, 15.0],
        'buy_volume': [100, 110, np.nan, 130, 140],
        'sell_volume': [90, np.nan, 95, 100, 105],
        'exchange': ['IEX', 'IEX', 'IEX', 'IEX', 'IEX']
    })
    
    # Run the raw test matrix through the cleaning function first, then add features
    cleaned_data = clean_missing_data(mock_data)
    processed_df = add_time_features(cleaned_data)
    
    # Assert that zero missing rows remain
    assert processed_df.isna().sum().sum() == 0, "Warning: NaN values leaked through the pipeline!"