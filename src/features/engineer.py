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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek        # 0=Mon
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak"]     = df["hour"].between(9, 22).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for exchange, grp in df.groupby("exchange"):
        daily = grp.groupby("date")["mcp"].mean().rename("daily_avg_mcp").reset_index()
        daily["mcp_lag1"]       = daily["daily_avg_mcp"].shift(1)
        daily["mcp_roll7_mean"] = daily["daily_avg_mcp"].rolling(7).mean()
        daily["mcp_roll7_std"]  = daily["daily_avg_mcp"].rolling(7).std()
        grp = grp.merge(
            daily[["date", "mcp_lag1", "mcp_roll7_mean", "mcp_roll7_std"]],
            on="date", how="left"
        )
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def main():
    df = pd.read_csv(INPUT, parse_dates=["date"])
    df = add_time_features(df)
    df = add_lag_features(df)
    df.to_csv(OUTPUT, index=False)
    print(f"Feature dataset saved -> {OUTPUT}  (shape: {df.shape})")


if __name__ == "__main__":
    main()
