"""
data_pipeline.py

Loads the raw Olist csvs, merges them into one dataset, cleans up
missing values and outliers. This is basically Chapter 3 (data
preprocessing) turned into actual code.

run this first before anything else, everything downstream needs
data/processed/merged_dataset.csv to exist
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def load_raw_data():
    # only loading the 5 csvs we actually need, ignoring geolocation/sellers/payments for now
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    order_items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")

    print(f"orders: {orders.shape}")
    print(f"order_items: {order_items.shape}")
    print(f"customers: {customers.shape}")
    print(f"reviews: {reviews.shape}")
    print(f"products: {products.shape}")

    return orders, order_items, customers, reviews, products


def merge_datasets(orders, order_items, customers, reviews, products):
    # start with orders + order items on order_id
    df = order_items.merge(orders, on="order_id", how="left")

    # bring in customer info
    df = df.merge(customers, on="customer_id", how="left")

    # reviews - some orders have more than one review so this can duplicate rows a bit,
    # keeping it simple for now and dealing with dupes later if it becomes an issue
    df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

    # product category
    df = df.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")

    print(f"merged shape before cleaning: {df.shape}")
    return df


def handle_missing_values(df):
    # order_purchase_timestamp and price are critical, cant really do RFM without them
    before = len(df)
    df = df.dropna(subset=["order_purchase_timestamp", "price"])
    print(f"dropped {before - len(df)} rows missing timestamp/price")

    # review_score missing just means no review left, not really a data quality issue
    # leave as NaN for now, sentiment.py will handle it

    # product category missing - fill with unknown instead of dropping, dont want to lose the sale
    df["product_category_name"] = df["product_category_name"].fillna("unknown")

    return df


def handle_outliers(df):
    # was using IQR here before but that was nuking ~7.5% of rows including all
    # the genuinely high-value orders (max price is 6735, thats not an error, thats
    # just an expensive item). those high spenders matter a lot for RFM monetary
    # score so cant just drop them.
    # only removing actual bad data now - negative or zero price doesnt make sense
    before = len(df)
    df = df[df["price"] > 0]
    print(f"removed {before - len(df)} rows with price <= 0 (data errors, not real outliers)")

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orders, order_items, customers, reviews, products = load_raw_data()
    df = merge_datasets(orders, order_items, customers, reviews, products)
    df = handle_missing_values(df)
    df = handle_outliers(df)

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    out_path = OUT_DIR / "merged_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()