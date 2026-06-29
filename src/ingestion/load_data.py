"""
load_data.py
------------
Loads and merges raw IEX and HPX CSV/Excel reports into a unified master DataFrame.

Usage:
    python src/ingestion/load_data.py
Output:
    data/processed/master.csv
"""

import pandas as pd
from pathlib import Path

RAW_IEX = Path("data/raw/iex")
RAW_HPX = Path("data/raw/hpx")
OUTPUT   = Path("data/processed/master.csv")


def load_iex() -> pd.DataFrame:
    """Read all IEX DAM CSVs from data/raw/iex/ and concatenate."""
    files = sorted(RAW_IEX.glob("iex_dam_*.csv"))
    if not files:
        raise FileNotFoundError(f"No IEX files found in {RAW_IEX}. Download data first.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # TODO: rename columns to match schema: date, hour, mcp, mcv, buy_volume, sell_volume
        df["exchange"] = "IEX"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_hpx() -> pd.DataFrame:
    """Read all HPX DAM CSVs from data/raw/hpx/ and concatenate."""
    files = sorted(RAW_HPX.glob("hpx_dam_*.csv"))
    if not files:
        print("Warning: No HPX files found. Skipping cross-exchange data.")
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["exchange"] = "HPX"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.dropna(subset=["mcp"])
    df = df.drop_duplicates()
    df = df.sort_values(["date", "exchange", "hour"]).reset_index(drop=True)
    return df


def main():
    iex = load_iex()
    hpx = load_hpx()
    master = pd.concat([iex, hpx], ignore_index=True) if not hpx.empty else iex
    master = clean(master)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT, index=False)
    print(f"Saved {len(master):,} rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
