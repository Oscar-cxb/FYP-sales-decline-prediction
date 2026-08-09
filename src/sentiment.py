"""
sentiment.py

Derives sentiment per customer two ways:
1. from review_score (1-2 negative, 3 neutral, 4-5 positive) - this is
   the main one, exactly what i specced in the IR (section 2.2.2)
2. from the actual review comment text using TextBlob polarity, as a
   supplementary signal

BIG CAVEAT (need to mention this in the report): olist reviews are in
PORTUGUESE. TextBlob is built for english and its sentiment lexicon
doesnt really work on portuguese text - i even flagged this myself in
my IR's library comparison table ("English primarily" for both
TextBlob and VADER). so the text-based polarity score is unreliable
here, review_score is the trustworthy signal, text polarity is just
a nice-to-have on top, not something to lean on for the report's
main sentiment claims.

ALSO IMPORTANT: this needs the same observation cutoff as
rfm_features.py (2018-06-01). if we dont filter by date here too,
we'd be using reviews written AFTER the point were measuring recency
from - thats basically leaking future info into a "risk as of June"
prediction. same idea as the right-censoring fix, just a different
spot it can sneak in.

run after data_pipeline.py (needs merged_dataset.csv for the
order_id -> customer_unique_id mapping)
"""

import pandas as pd
from pathlib import Path
from textblob import TextBlob

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# has to match rfm_features.py's cutoff, otherwise sentiment and RFM
# would be measuring two different time windows
OBSERVATION_CUTOFF = pd.Timestamp("2018-06-01")


def score_from_review_rating(reviews):
    # matches what i wrote in the IR - 1,2 = negative, 3 = neutral, 4,5 = positive
    def label(score):
        if score <= 2:
            return "negative"
        elif score == 3:
            return "neutral"
        else:
            return "positive"

    reviews["sentiment_label"] = reviews["review_score"].apply(label)
    return reviews


def score_from_review_text(reviews):
    # only run textblob on rows that actually have comment text, most are blank
    has_text = reviews["review_comment_message"].notna()
    print(f"{has_text.sum()} / {len(reviews)} reviews have comment text ({has_text.mean()*100:.1f}%)")

    def get_polarity(text):
        if pd.isna(text):
            return None
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return None

    reviews["text_polarity"] = reviews["review_comment_message"].apply(get_polarity)
    return reviews


def main():
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    print(f"reviews loaded: {reviews.shape}")

    reviews = score_from_review_rating(reviews)
    reviews = score_from_review_text(reviews)

    # need customer_unique_id AND the order timestamp this time, so we can
    # filter out reviews tied to orders placed after the observation cutoff
    merged = pd.read_csv(
        PROCESSED_DIR / "merged_dataset.csv",
        usecols=["order_id", "customer_unique_id", "order_purchase_timestamp"],
    )
    merged["order_purchase_timestamp"] = pd.to_datetime(merged["order_purchase_timestamp"])
    merged = merged.drop_duplicates(subset="order_id")

    reviews = reviews.merge(merged, on="order_id", how="inner")
    print(f"reviews matched to customers: {len(reviews)}")

    before = len(reviews)
    reviews = reviews[reviews["order_purchase_timestamp"] <= OBSERVATION_CUTOFF]
    print(f"trimmed {before - len(reviews)} reviews after observation cutoff ({OBSERVATION_CUTOFF.date()})")

    # aggregate to customer level - a customer can have multiple reviews
    agg = reviews.groupby("customer_unique_id").agg(
        avg_review_score=("review_score", "mean"),
        pct_negative_reviews=("sentiment_label", lambda x: (x == "negative").mean()),
        avg_text_polarity=("text_polarity", "mean"),
    ).reset_index()

    # customers with no comment text at all will have NaN polarity, fill with 0
    # (neutral) rather than dropping - remember this is the less reliable
    # signal anyway because of the portuguese language issue
    agg["avg_text_polarity"] = agg["avg_text_polarity"].fillna(0)

    print(agg.describe())

    # honest check: how many of the actual RFM customers have ANY review at all
    # within the observation window (not the text-comment %, the review_score %)
    rfm_customers = pd.read_csv(PROCESSED_DIR / "rfm_features.csv", usecols=["customer_unique_id"])
    matched = rfm_customers["customer_unique_id"].isin(agg["customer_unique_id"]).sum()
    print(f"{matched} / {len(rfm_customers)} RFM customers have at least one review "
          f"({matched/len(rfm_customers)*100:.1f}%) - the rest get filled with dataset "
          f"averages in classifier.py, not dropped")

    out_path = PROCESSED_DIR / "sentiment_features.csv"
    agg.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()