"""
sentiment_validation.py

Measures how well TextBlob actually performs on the olist review text,
instead of just asserting that it doesnt work.

WHY THIS EXISTS
Chapter 2 reviews TextBlob, VADER and BERT in some detail, but the
sentiment features that ended up in the model are derived from the
numeric review score rather than from any NLP output. That gap needs
evidence behind it, not a claim. This script produces the number:
how often TextBlob's polarity agrees with the star rating the customer
actually gave.

The star rating is treated as ground truth here. A customer who awards
one star has told us their sentiment directly - if a sentiment analyser
reads their comment and calls it positive, the analyser is wrong.

Run once, quote the output in Chapter 3 and Chapter 5. Nothing else in
the pipeline depends on it.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")

# TextBlob calls anything above this positive and anything below the
# negative of it negative. 0.1 is the usual convention rather than 0,
# which would force every neutral review into one camp or the other
POLARITY_CUTOFF = 0.1

# words that appear constantly in portuguese and almost never in english,
# used to confirm what language the reviews are actually in
PT_MARKERS = ["não", "muito", "produto", "chegou", "prazo", "entrega",
              "recomendo", "veio", "antes", "compra", "gostei", "bem",
              "está", "para", "com", "que", "mas", "foi"]


def score_to_class(score):
    # the mapping from the IR - 1 and 2 negative, 3 neutral, 4 and 5 positive
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def polarity_to_class(p):
    if p > POLARITY_CUTOFF:
        return "positive"
    if p < -POLARITY_CUTOFF:
        return "negative"
    return "neutral"


def detect_language(texts, sample=3000):
    """rough check that the review text is portuguese, not english."""
    hits = 0
    checked = 0
    for t in texts[:sample]:
        words = str(t).lower().split()
        if not words:
            continue
        checked += 1
        if any(m in words for m in PT_MARKERS):
            hits += 1
    return hits / checked if checked else 0.0


def main():
    try:
        from textblob import TextBlob
    except ImportError:
        print("textblob is not installed. run: pip install textblob")
        sys.exit(1)

    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    total = len(reviews)

    with_text = reviews.dropna(subset=["review_comment_message"]).copy()
    with_text = with_text[with_text["review_comment_message"].str.strip() != ""]

    print(f"reviews in the dataset:        {total:,}")
    print(f"reviews with comment text:     {len(with_text):,} "
          f"({len(with_text)/total*100:.1f}%)")

    pt_rate = detect_language(with_text["review_comment_message"].tolist())
    print(f"comments containing common portuguese words: {pt_rate*100:.1f}%")

    print("\nrunning textblob on the comment text...")
    with_text["polarity"] = with_text["review_comment_message"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity)

    zero = (with_text["polarity"] == 0).mean()
    print(f"comments textblob scored exactly 0.0: {zero*100:.1f}%")
    print("  (a polarity of exactly zero means textblob recognised no "
          "sentiment-bearing words at all)")

    print(f"\npolarity distribution:")
    print(with_text["polarity"].describe().round(4).to_string())

    with_text["textblob_class"] = with_text["polarity"].apply(polarity_to_class)
    with_text["actual_class"] = with_text["review_score"].apply(score_to_class)

    agree = (with_text["textblob_class"] == with_text["actual_class"]).mean()

    # what youd get by always guessing the most common class - the number
    # that actually matters, since agreement on its own can look respectable
    # simply because most reviews are positive
    majority = with_text["actual_class"].value_counts(normalize=True).max()
    majority_label = with_text["actual_class"].value_counts().idxmax()

    print(f"\n{'='*60}")
    print(f"agreement with the star rating:     {agree*100:.1f}%")
    print(f"always guessing '{majority_label}':  {majority*100:.1f}%")
    print(f"difference:                         {(agree-majority)*100:+.1f} points")
    print(f"{'='*60}")

    if agree <= majority:
        print("\ntextblob performs no better than guessing the most common class.")
    else:
        print(f"\ntextblob beats the majority guess by {(agree-majority)*100:.1f} points.")

    print("\nwhere it goes wrong - actual class vs textblob class:")
    matrix = pd.crosstab(with_text["actual_class"], with_text["textblob_class"],
                         normalize="index").round(3)
    print(matrix.to_string())

    # the case that matters most: unhappy customers read as happy
    neg = with_text[with_text["actual_class"] == "negative"]
    if len(neg):
        missed = (neg["textblob_class"] != "negative").mean()
        as_pos = (neg["textblob_class"] == "positive").mean()
        print(f"\nof reviews scoring 1 or 2 stars:")
        print(f"  textblob failed to call them negative: {missed*100:.1f}%")
        print(f"  textblob called them POSITIVE:         {as_pos*100:.1f}%")

    REPORTS_DIR.mkdir(exist_ok=True)
    summary = pd.DataFrame([{
        "reviews_total": total,
        "reviews_with_text": len(with_text),
        "pct_with_text": round(len(with_text) / total * 100, 1),
        "pct_portuguese_markers": round(pt_rate * 100, 1),
        "pct_polarity_exactly_zero": round(zero * 100, 1),
        "agreement_with_star_rating": round(agree * 100, 1),
        "majority_class_baseline": round(majority * 100, 1),
        "lift_over_baseline": round((agree - majority) * 100, 1),
    }])
    summary.to_csv(REPORTS_DIR / "sentiment_validation.csv", index=False)
    matrix.to_csv(REPORTS_DIR / "sentiment_confusion.csv")
    print(f"\nsaved to reports/sentiment_validation.csv and sentiment_confusion.csv")


if __name__ == "__main__":
    main()