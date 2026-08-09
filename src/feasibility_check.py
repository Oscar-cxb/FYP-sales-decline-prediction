"""
feasibility_check.py

Throwaway script - run this BEFORE rebuilding anything at seller level.
Checks whether the olist data actually supports a seller-month decline
target, because if sellers dont have enough monthly history or the
decline rate is lopsided the way customer repurchase was (99.4% one
sided), theres no point rebuilding around it.

nothing here gets used by the pipeline, its just a go/no-go check.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")

# a seller needs at least this many months of history to be usable -
# cant say much about a decline trend from one or two months
MIN_MONTHS = 6

# candidate thresholds for what counts as a "decline"
THRESHOLDS = [0.10, 0.20, 0.30]


def build_seller_months():
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # only counting orders that actually completed - cancelled/unavailable
    # ones arent revenue
    ok = orders[orders["order_status"].isin(["delivered", "shipped", "invoiced"])]
    print(f"orders total: {len(orders)}, usable status: {len(ok)}")

    df = items.merge(ok[["order_id", "order_purchase_timestamp"]], on="order_id", how="inner")
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M")

    panel = df.groupby(["seller_id", "month"]).agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique"),
    ).reset_index()

    return panel


def main():
    panel = build_seller_months()

    print(f"\nseller-month rows: {len(panel):,}")
    print(f"distinct sellers: {panel['seller_id'].nunique():,}")
    print(f"month range: {panel['month'].min()} to {panel['month'].max()}")

    months_per_seller = panel.groupby("seller_id").size()
    print(f"\nmonths of history per seller:")
    print(months_per_seller.describe())

    keep = months_per_seller[months_per_seller >= MIN_MONTHS].index
    panel = panel[panel["seller_id"].isin(keep)]
    print(f"\nsellers with >= {MIN_MONTHS} months: {len(keep):,}")
    print(f"seller-month rows kept: {len(panel):,}")

    if len(panel) == 0:
        print("\nNO USABLE DATA - stop here, dont rebuild")
        return

    # next month's revenue for the same seller
    panel = panel.sort_values(["seller_id", "month"])
    panel["next_revenue"] = panel.groupby("seller_id")["revenue"].shift(-1)

    # rows where theres no next month cant be labelled, drop them
    labelled = panel.dropna(subset=["next_revenue"]).copy()
    print(f"rows with a next month to compare against: {len(labelled):,}")

    labelled["pct_change"] = (labelled["next_revenue"] - labelled["revenue"]) / labelled["revenue"]

    print(f"\nmonth-over-month revenue change:")
    print(labelled["pct_change"].describe())

    print(f"\ndecline rate at each threshold:")
    for t in THRESHOLDS:
        rate = (labelled["pct_change"] <= -t).mean()
        n = (labelled["pct_change"] <= -t).sum()
        verdict = "usable" if 0.15 <= rate <= 0.60 else "TOO LOPSIDED"
        print(f"  drop of more than {t*100:.0f}%: {n:,} rows ({rate*100:.1f}%)  -> {verdict}")

    # sanity: is there actually variation, or does everything just grow?
    print(f"\nsellers whose revenue ever declined by >20%: "
          f"{labelled[labelled['pct_change'] <= -0.2]['seller_id'].nunique():,} "
          f"of {labelled['seller_id'].nunique():,}")


if __name__ == "__main__":
    main()