"""
rfm_features.py

Computes RFM (recency, frequency, monetary) + delivery delay per
customer. This is basically section 3.4.1 "Feature Engineering" from
the IR, turned into code.

note: olist has customer_id (changes per order) AND customer_unique_id
(same person across orders). gotta group by customer_unique_id or
frequency will be wrong for repeat customers - this tripped me up
when i first looked at the dataset docs

IMPORTANT: the dataset basically stops mid-Sept 2018 (this is even
in my own IR Graph 5 - the order volume just falls off a cliff at the
end, its not a real decline its just where data collection ended).
if we calc recency against the actual last date in the data, everyone
near the end looks "at risk" just because we stopped observing them,
not because they actually churned. this is called right-censoring.
fix: cut the observation window off a few months before the real end
so recency is measured against a point where we still have a fair
amount of runway to see if someone came back or not.

also added avg_delivery_delay as an extra feature (see
compute_delivery_delay) since frequency is basically flat for this
dataset and doesnt give the model much to work with on its own.

NOTE ON at_risk LABEL - this file used to define at_risk right here,
but that went through two broken versions before landing on the
right approach, worth explaining since its all in one place now:

  v1: at_risk = recency > 180 days. broken - recency is also a model
      feature so the model just rediscovers the threshold. fake
      perfect looking results.

  v2: at_risk = combined recency/frequency/monetary quintile score.
      still broken - same features, just combined into a fancier
      formula. classifier still hit ~99.7% F1 because tree models are
      really good at learning threshold combinations like that.

  v3: at_risk = no purchase again after the cutoff (real future
      behaviour, not derived from the same features). this actually
      fixed the leakage... but then at_risk came out 99.4% because
      olist is a one-shot marketplace, 90%+ of customers never buy
      again regardless of how happy they were. not usable as a label,
      just measures the marketplace's repeat-purchase rate not risk.

  FINAL: at_risk is now defined in classifier.py instead, based on
  sentiment (did the customer leave a negative review). that needs
  sentiment_features.csv which this file doesnt touch, so this file
  just does features now - recency, frequency, monetary, delivery
  delay. no label here anymore.

run after data_pipeline.py
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# cutting off the last ~3 months of the dataset (order volume was still healthy
# through May 2018 per my own EDA, drops off after that / ends properly in Sept)
# so we dont mistake "data ran out" for "customer churned"
OBSERVATION_CUTOFF = pd.Timestamp("2018-06-01")


def compute_recency(df, reference_date):
    last_purchase = df.groupby("customer_unique_id")["order_purchase_timestamp"].max()
    recency = (reference_date - last_purchase).dt.days
    return recency.rename("recency")


def compute_frequency(df):
    freq = df.groupby("customer_unique_id")["order_id"].nunique()
    return freq.rename("frequency")


def compute_monetary(df):
    monetary = df.groupby("customer_unique_id")["price"].sum()
    return monetary.rename("monetary")


def compute_delivery_delay(df):
    # frequency barely does anything here since 90%+ customers only order once
    # (my IR EDA already showed this, and Matuszelanski & Kopczewska 2022 found
    # the same thing on this exact dataset - RFM frequency isnt a strong churn
    # signal for a one-shot marketplace like olist)
    # adding delivery delay as a feature that doesnt depend on repeat purchases at all,
    # late delivery is a well known dissatisfaction/churn driver in ecommerce lit
    df = df.copy()
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    # only makes sense for orders that were actually delivered
    delivered = df.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    delivered = delivered.copy()
    delivered["delay_days"] = (delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]).dt.days

    avg_delay = delivered.groupby("customer_unique_id")["delay_days"].mean()
    return avg_delay.rename("avg_delivery_delay")


def main():
    df = pd.read_csv(PROCESSED_DIR / "merged_dataset.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    before = len(df)
    df = df[df["order_purchase_timestamp"] <= OBSERVATION_CUTOFF]
    print(f"trimmed {before - len(df)} rows after observation cutoff ({OBSERVATION_CUTOFF.date()})")

    reference_date = OBSERVATION_CUTOFF + pd.Timedelta(days=1)
    print(f"reference date for recency calc: {reference_date}")

    recency = compute_recency(df, reference_date)
    frequency = compute_frequency(df)
    monetary = compute_monetary(df)
    delivery_delay = compute_delivery_delay(df)

    rfm = pd.concat([recency, frequency, monetary, delivery_delay], axis=1).reset_index()

    # some customers have no delivered orders in this window (still in transit,
    # cancelled etc) so avg_delivery_delay will be NaN for them - fill with the
    # median delay rather than dropping these customers entirely
    median_delay = rfm["avg_delivery_delay"].median()
    missing_delay = rfm["avg_delivery_delay"].isna().sum()
    print(f"filling {missing_delay} customers with no delivery data using median delay ({median_delay:.1f} days)")
    rfm["avg_delivery_delay"] = rfm["avg_delivery_delay"].fillna(median_delay)

    # no at_risk label here anymore - see note at top of file. label gets
    # defined in classifier.py once sentiment data is available to join with

    print(f"total customers: {len(rfm)}")
    print(rfm.describe())

    out_path = PROCESSED_DIR / "rfm_features.csv"
    rfm.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()