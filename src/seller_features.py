"""
seller_features.py  (v2 - rebuilt)

Builds a seller-month panel and the sales decline label.

WHY v2
The first version gave momentum features (change vs last month, rolling
averages) to the Trading block ONLY. Every other block got a flat
snapshot. Then the ablation said "trading dominates" - but that was
never a fair fight, i'd handed one block the ability to see trends and
denied it to the rest.

And trends are exactly where the signal should live. "average review
score is 4.1" says almost nothing. "review score fell from 4.6 to 4.1
over three months" is an actual early warning. Same for RFM - rising
recency month over month means customers are drifting away, which IS
the decline mechanism this project is about.

so v2 gives every block the same treatment:
    level    - the value this month
    delta    - change vs last month
    roll3    - 3 month rolling average
    trend    - slope over the last 3 months

WHY SELLER LEVEL AT ALL
Customer level decline is unmeasurable on olist - 99.4% of customers
never buy again regardless of satisfaction, so "did they come back"
just measures the marketplace's repeat rate. Sellers have real monthly
revenue that rises and falls, and a ~37% decline rate. thats a
learnable target and its literally "sales decline".

NO LEAKAGE RULES
  - every feature for month m uses ONLY data up to the end of month m
  - deltas/rolling are computed with shift() inside each seller, so
    they look backwards only
  - the label is the ONLY thing that reads month m+1
  - the final month of the dataset is dropped (collection stops mid
    september, every seller would look like they collapsed)
  - tiny sellers filtered out or R$50 -> R$200 reads as a 300% swing

run first, before seller_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

MIN_MONTHS = 6
MIN_MONTHLY_REVENUE = 100.0
DECLINE_THRESHOLD = 0.20
WINDOW_DAYS = 90
USABLE_STATUS = ["delivered", "shipped", "invoiced"]

# every one of these gets level + delta + roll3 + trend
MOMENTUM_BASE = [
    "revenue", "orders", "customers", "avg_item_price",
    "rfm_recency", "rfm_frequency", "rfm_monetary",
    "avg_review_score", "pct_negative_reviews",
    "avg_delivery_delay", "pct_late_deliveries",
]


def load_orders():
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")

    for c in ["order_purchase_timestamp", "order_delivered_customer_date",
              "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    orders = orders[orders["order_status"].isin(USABLE_STATUS)]

    df = items.merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp",
                "order_delivered_customer_date", "order_estimated_delivery_date"]],
        on="order_id", how="inner")
    df = df.merge(customers[["customer_id", "customer_unique_id"]],
                  on="customer_id", how="left")

    rev = reviews.groupby("order_id")["review_score"].mean().reset_index()
    df = df.merge(rev, on="order_id", how="left")

    df["delay_days"] = (df["order_delivered_customer_date"]
                        - df["order_estimated_delivery_date"]).dt.days
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M")

    print(f"order-item rows after status filter: {len(df):,}")
    return df


def build_monthly_panel(df):
    return df.groupby(["seller_id", "month"]).agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_unique_id", "nunique"),
        avg_item_price=("price", "mean"),
    ).reset_index()


def window_features(df, months):
    """RFM + sentiment + delivery over a 90 day trailing window.

    looping the months (only ~22 of them) instead of doing something
    vectorised and clever - slower but its obvious by inspection that
    nothing from the future gets in, which matters more here
    """
    out = []
    for m in months:
        month_end = m.to_timestamp(how="end")
        w = df[(df["order_purchase_timestamp"] > month_end - pd.Timedelta(days=WINDOW_DAYS))
               & (df["order_purchase_timestamp"] <= month_end)]
        if w.empty:
            continue

        last = w.groupby(["seller_id", "customer_unique_id"])[
            "order_purchase_timestamp"].max().reset_index()
        last["r"] = (month_end - last["order_purchase_timestamp"]).dt.days
        rec = last.groupby("seller_id")["r"].mean().rename("rfm_recency")

        per_cust = w.groupby(["seller_id", "customer_unique_id"]).agg(
            o=("order_id", "nunique"), s=("price", "sum")).reset_index()
        freq = per_cust.groupby("seller_id")["o"].mean().rename("rfm_frequency")
        mon = per_cust.groupby("seller_id")["s"].mean().rename("rfm_monetary")

        sent = w.groupby("seller_id")["review_score"].mean().rename("avg_review_score")
        neg = (w.assign(n=(w["review_score"] <= 2).astype(float))
                 .groupby("seller_id")["n"].mean().rename("pct_negative_reviews"))
        delay = w.groupby("seller_id")["delay_days"].mean().rename("avg_delivery_delay")
        late = (w.assign(l=(w["delay_days"] > 0).astype(float))
                  .groupby("seller_id")["l"].mean().rename("pct_late_deliveries"))

        block = pd.concat([rec, freq, mon, sent, neg, delay, late], axis=1).reset_index()
        block["month"] = m
        out.append(block)

    return pd.concat(out, ignore_index=True)


def slope(series):
    # least squares slope over whatever values are present. used on a
    # 3 month rolling window so its "which way has this been heading"
    y = series.dropna().values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def add_momentum(panel):
    # THE v2 CHANGE - this now runs over every feature block, not just
    # the trading columns
    panel = panel.sort_values(["seller_id", "month"]).copy()
    g = panel.groupby("seller_id")

    for col in MOMENTUM_BASE:
        prev = g[col].shift(1)

        # absolute change - works for any column
        panel[f"d_{col}"] = panel[col] - prev

        # percentage change only where a ratio actually means something
        # (a % change in "average review score" is odd, an absolute drop
        # of 0.5 stars is the intuitive thing)
        if col in ["revenue", "orders", "customers", "rfm_monetary"]:
            panel[f"pct_{col}"] = (panel[col] - prev) / prev.replace(0, np.nan)

        panel[f"roll3_{col}"] = g[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
        panel[f"trend_{col}"] = g[col].transform(
            lambda s: s.rolling(3, min_periods=2).apply(slope, raw=False))

    panel["months_active"] = g.cumcount() + 1
    return panel


def add_label(panel):
    panel = panel.sort_values(["seller_id", "month"]).copy()
    panel["next_revenue"] = panel.groupby("seller_id")["revenue"].shift(-1)
    panel = panel.dropna(subset=["next_revenue"])
    panel["pct_change_next"] = (panel["next_revenue"] - panel["revenue"]) / panel["revenue"]
    panel["decline"] = (panel["pct_change_next"] <= -DECLINE_THRESHOLD).astype(int)

    # these two ARE the answer - dropping them so they cant leak by accident
    return panel.drop(columns=["next_revenue", "pct_change_next"])


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_orders()

    last_month = df["month"].max()
    df = df[df["month"] < last_month]
    print(f"dropped final partial month ({last_month})")

    panel = build_monthly_panel(df)
    print(f"seller-months before filtering: {len(panel):,}")

    panel = panel[panel["revenue"] >= MIN_MONTHLY_REVENUE]
    counts = panel.groupby("seller_id").size()
    keep = counts[counts >= MIN_MONTHS].index
    panel = panel[panel["seller_id"].isin(keep)]
    print(f"after filters: {len(panel):,} rows, {len(keep):,} sellers")

    months = sorted(panel["month"].unique())
    panel = panel.merge(window_features(df, months), on=["seller_id", "month"], how="left")

    # fill the window features BEFORE momentum, otherwise the deltas
    # inherit the gaps and spread them further
    for col in ["rfm_recency", "rfm_frequency", "rfm_monetary", "avg_review_score",
                "pct_negative_reviews", "avg_delivery_delay", "pct_late_deliveries"]:
        n = panel[col].isna().sum()
        if n:
            print(f"  filling {n} missing {col} with median")
            panel[col] = panel[col].fillna(panel[col].median())

    panel = add_momentum(panel)
    panel = add_label(panel)

    # first month of each seller has no previous month so deltas are NaN.
    # 0 = "no change observed yet", which is the honest reading
    feat_cols = [c for c in panel.columns if c not in ["seller_id", "month", "decline"]]
    panel[feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\nfinal panel: {len(panel):,} seller-months, {panel['seller_id'].nunique():,} sellers")
    print(f"features: {len(feat_cols)} (v1 had 15)")
    print(f"decline rate: {panel['decline'].mean()*100:.1f}%")
    print(f"months: {panel['month'].min()} to {panel['month'].max()}")

    panel["month"] = panel["month"].astype(str)
    out = PROCESSED_DIR / "seller_panel.csv"
    panel.to_csv(out, index=False)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()