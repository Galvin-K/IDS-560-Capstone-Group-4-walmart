"""Export historical M5 sell_price by store and item as of 2015-12-31.

Run once in the repository root:
    python export_m5_prices.py

Reads M5/datasets/calendar.csv and M5/datasets/sell_prices.csv.
Writes forecasting/m5_prices_asof_2015_12_31.csv.gz.
No future price observations are used. Prices are SALES prices, not supplier costs.
"""
from pathlib import Path
import argparse
import pandas as pd


def export_prices(calendar_path: Path, price_path: Path, output_path: Path,
                  as_of: str = "2015-12-31") -> tuple[int, int]:
    cal = pd.read_csv(calendar_path, usecols=["date", "wm_yr_wk"])
    cal["date"] = pd.to_datetime(cal["date"], errors="raise")
    selected = cal.loc[cal["date"].eq(pd.Timestamp(as_of)), "wm_yr_wk"]
    if len(selected) != 1:
        raise ValueError(f"Calendar must identify exactly one M5 week for {as_of}")
    last_week = int(selected.iloc[0])
    # Keep only the newest observed M5 week per item-store on/before the decision date.
    best = None
    for chunk in pd.read_csv(
        price_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        chunksize=250_000,
    ):
        chunk["wm_yr_wk"] = pd.to_numeric(chunk["wm_yr_wk"], errors="coerce")
        chunk["sell_price"] = pd.to_numeric(chunk["sell_price"], errors="coerce")
        chunk = chunk.loc[
            chunk["wm_yr_wk"].le(last_week)
            & chunk["sell_price"].gt(0)
            & chunk["sell_price"].notna()
        ]
        if chunk.empty:
            continue
        chunk = chunk.sort_values("wm_yr_wk").drop_duplicates(
            ["store_id", "item_id"], keep="last"
        )
        if best is None:
            best = chunk
        else:
            best = pd.concat([best, chunk], ignore_index=True).sort_values(
                "wm_yr_wk"
            ).drop_duplicates(["store_id", "item_id"], keep="last")
    if best is None or best.empty:
        raise ValueError("No historical M5 prices found on or before the decision date")
    best = best.sort_values(["store_id", "item_id"]).reset_index(drop=True)
    best.insert(0, "as_of_date", as_of)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output_path, index=False, compression="gzip")
    return len(best), last_week


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path("M5/datasets/calendar.csv"))
    parser.add_argument("--prices", type=Path, default=Path("M5/datasets/sell_prices.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "forecasting/m5_prices_asof_2015_12_31.csv.gz"))
    args = parser.parse_args()
    rows, week = export_prices(args.calendar, args.prices, args.output)
    print(f"Saved {rows:,} M5 store-product prices as of 2015-12-31 (week {week}) to {args.output}")


if __name__ == "__main__":
    main()
