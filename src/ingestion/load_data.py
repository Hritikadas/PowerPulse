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

# Maps the exact column names in the raw IEX export to our internal schema.
IEX_COLUMN_MAP = {
    "Date": "date",
    "Purchase Bid (MWh)": "buy_volume",
    "Sell Bid (MWh)": "sell_volume",
    "MCV (MWh)": "mcv",
    "Final Scheduled Volume (MWh)": "final_scheduled_volume",
    "MCP (Rs/MWh) *": "mcp_rs_mwh",
    "MCP (Rs/MWh)": "mcp_rs_mwh",       # fallback if the trailing " *" isn't present
    "Weighted MCP (Rs/MWh)": "weighted_mcp_rs_mwh",
}


def load_iex() -> pd.DataFrame:
    """Read all IEX DAM exports from data/raw/iex/ and concatenate.

    Handles both .csv and .xlsx, and skips the 4-row IEX banner so the
    real header ("Date", "Hour", "MCP (Rs/MWh) *", ...) is used.
    """
    csv_files = sorted(RAW_IEX.glob("iex_dam_*.csv"))
    xlsx_files = sorted(RAW_IEX.glob("iex_dam_*.xlsx"))
    files = csv_files + xlsx_files
    if not files:
        raise FileNotFoundError(f"No IEX files found in {RAW_IEX}. Download data first.")

    frames = []
    for f in files:
        if f.suffix == ".xlsx":
            df = pd.read_excel(f, skiprows=0)
        else:
            df = pd.read_csv(f, skiprows=0)

        # Drop any fully-empty trailing rows/columns the export sometimes adds
        df = df.dropna(how="all")
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

        # IEX repeats a daily Summary block (Total/Max/Min/Avg rows) after each
        # day's 24 hourly rows, including a re-stated header row ("Date, Summary, ...").
        # Keep only real hourly rows: the "Hour" column must be a number 1-24.
        df["hour"] = pd.to_numeric(df["Hour"], errors="coerce")
        df = df[df["hour"].between(1, 24)]
        df = df.drop(columns=["Hour"])

        df = df.rename(columns=IEX_COLUMN_MAP)

        # Force numeric dtypes (mixed string/blank cells can sneak through from
        # any malformed rows the export includes).
        numeric_cols = ["buy_volume", "sell_volume", "mcv",
                         "final_scheduled_volume", "mcp_rs_mwh", "weighted_mcp_rs_mwh"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # MCP is reported in Rs/MWh; convert to ₹/kWh to match the proposal's schema (divide by 1000)
        df["mcp"] = df["mcp_rs_mwh"] / 1000

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

    # Keep a clean, consistent set of columns for downstream steps.
    keep_cols = ["date", "hour", "exchange", "mcp", "mcv", "buy_volume", "sell_volume"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


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