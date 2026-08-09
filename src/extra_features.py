"""
extra_features.py

Adds product category + payment behaviour as extra features, based on
what Matuszelanski & Kopczewska (2022) found on this SAME dataset -
they showed churn was linked to payment value, product category and
shipping cost, even without RFM. worth trying since my own model so
far only really has delivery_delay doing the heavy lifting (SHAP
importance 1.58, way ahead of everything else) - these might give it
more to work with.

pulling straight from the raw csvs instead of merged_dataset.csv,
cleaner than going back and editing data_pipeline.py at this point

run after data_pipeline.py (uses same OBSERVATION_CUTOFF as
rfm_features.py / sentiment.py - gotta stay consistent or its the
same leakage problem all over again)
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OBSERVATION_CUTOFF = pd.Timestamp("2018-06-01")

# too many product categories to one-hot encode all of them (70+), only
# keeping the top N and bucketing the rest as "other" - classifier.py
# does the actual one-hot encoding, this file just figures out the label
TOP_N_CATEGORIES = 10


def build_order_level_features():
    payments = pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv")
    order_items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")

    # some orders have multiple payment rows (installments), aggregate to one row per order
    payment_agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
    ).reset_index()

    # category per item, just taking whichever category shows up per order item row
    item_category = order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    item_category = item_category[["order_id", "product_category_name"]]

    orders_slim = orders[["order_id", "customer_id", "order_purchase_timestamp"]].copy()
    orders_slim["order_purchase_timestamp"] = pd.to_datetime(orders_slim["order_purchase_timestamp"])

    df = orders_slim.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
    df = df.merge(payment_agg, on="order_id", how="left")
    df = df.merge(item_category, on="order_id", how="left")

    before = len(df)
    df = df[df["order_purchase_timestamp"] <= OBSERVATION_CUTOFF]
    print(f"trimmed {before - len(df)} rows after observation cutoff ({OBSERVATION_CUTOFF.date()})")

    return df


def aggregate_to_customer(df):
    # figure out top N categories overall first, everything else becomes "other"
    top_categories = df["product_category_name"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    print(f"top {TOP_N_CATEGORIES} categories: {top_categories}")

    df["product_category_name"] = df["product_category_name"].where(
        df["product_category_name"].isin(top_categories), "other"
    )

    def mode_or_unknown(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else "unknown"

    agg = df.groupby("customer_unique_id").agg(
        dominant_category=("product_category_name", mode_or_unknown),
        dominant_payment_type=("payment_type", mode_or_unknown),
        avg_installments=("payment_installments", "mean"),
        avg_payment_value=("total_payment_value", "mean"),
    ).reset_index()

    return agg


def main():
    order_df = build_order_level_features()
    agg = aggregate_to_customer(order_df)

    print(f"customers with extra features: {len(agg)}")
    print(agg["dominant_category"].value_counts())
    print(agg["dominant_payment_type"].value_counts())
    print(agg[["avg_installments", "avg_payment_value"]].describe())

    out_path = PROCESSED_DIR / "extra_features.csv"
    agg.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()