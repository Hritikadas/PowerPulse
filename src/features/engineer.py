"""
engineer.py
-----------
Adds time-series features to the master dataset.

Usage:
    python src/features/engineer.py
Input:  data/processed/master.csv
Output: data/processed/master_features.csv
"""

import pandas as pd
from pathlib import Path

INPUT  = Path("data/processed/master.csv")
OUTPUT = Path("data/processed/master_features.csv")
def clean_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts data and cleanly interpolates gaps."""
    # 🟢 ADD THIS LINE RIGHT HERE: Convert date column from text to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date') 
    df['mcp'] = df['mcp'].interpolate(method='linear').ffill().bfill()
    df['buy_volume'] = df['buy_volume'].interpolate(method='linear').ffill().bfill()
    df['sell_volume'] = df['sell_volume'].interpolate(method='linear').ffill().bfill()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Your Original Code ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak"]     = df["hour"].between(9, 22).astype(int)
    
    # --- 🟢 Step 1.2: Indian Grid Peak vs Off-Peak Indicators ---
    def get_grid_period(hour):
        if 6 <= hour <= 10:
            return 'Morning_Peak'
        elif 18 <= hour <= 22:
            return 'Evening_Peak'
        else:
            return 'Off_Peak'

    df['grid_period'] = df['hour'].apply(get_grid_period)
    # One-hot encode them into 0 and 1 columns for the ML model
    df = pd.get_dummies(df, columns=['grid_period'], drop_first=False)

    # --- 🟢 Step 1.3: Volume-Based Interaction Ratio ---
    # Note: Using lower-case column names to match your script's style
    df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-5)
    
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for exchange, grp in df.groupby("exchange"):
        # Calculate base daily average metrics per exchange
        daily = grp.groupby("date")["mcp"].mean().rename("daily_avg_mcp").reset_index()
        
        # Original features
        daily["mcp_lag1"] = daily["daily_avg_mcp"].shift(1)
        daily["mcp_roll7_mean"] = daily["daily_avg_mcp"].rolling(7).mean()
        daily["mcp_roll7_std"]  = daily["daily_avg_mcp"].rolling(7).std()
        
        # 🔥 Your Advanced Features: 12-hour and 24-hour Exponential Moving Averages
        daily['mcp_ema_12'] = daily['daily_avg_mcp'].ewm(span=12, adjust=False).mean()
        daily['mcp_ema_24'] = daily['daily_avg_mcp'].ewm(span=24, adjust=False).mean()
        
        # Merge the newly engineered features back into the exchange's main group data
        grp = pd.merge(grp, daily, on="date", how="left")
        frames.append(grp)
        
    # Recombine all the exchanges into a single clean master dataframe
    return pd.concat(frames, ignore_index=True)


def main():
    df = pd.read_csv(INPUT, parse_dates=["date"])
    # --- 🟢 Step 2.1: Smart Time-of-Day Interpolation ---
    # Sort chronologically to make sure time-based gaps line up correctly
    df = df.sort_values(by='date') 

# Fill missing spaces cleanly using time-weighted progression
    df['mcp'] = df['mcp'].interpolate(method='linear').ffill().bfill()
    df['buy_volume'] = df['buy_volume'].interpolate(method='linear').ffill().bfill()
    df['sell_volume'] = df['sell_volume'].interpolate(method='linear').ffill().bfill()
    df = add_time_features(df)
    df = add_lag_features(df)
    df.to_csv(OUTPUT, index=False)
    print(f"Feature dataset saved -> {OUTPUT}  (shape: {df.shape})")


# if __name__ == "__main__":
#     df = pd.read_csv("data/processed/master.csv")
    
#     # Clean the data using our new function first!
#     df = clean_missing_data(df)
    
#     df = add_time_features(df)
#     df = add_lag_features(df)
#     df.to_csv(OUTPUT, index=False)
#     main()
# DELETE the if __name__ == "__main__" block that's there now
# Replace with just this:

if __name__ == "__main__":
    main()