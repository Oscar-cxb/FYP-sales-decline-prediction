"""
review_examples.py

Builds the per-review table behind the raw review card on the dashboard.

WHY THIS EXISTS
sentiment_validation.py and nlp_lexicon.py both report aggregates - "TextBlob
scored exactly zero on X% of comments", "the induced lexicon agreed with the
star rating 77.8% of the time". Those are the right numbers for the chapters,
but they are summary statistics, and a marker asking "show me one" currently
has nothing to look at. This writes a sample of individual reviews with both
scores attached, so the claim can be inspected one row at a time.

Nothing in the model depends on this file. It is evidence, not a feature.

HOW THE SAMPLE IS DRAWN
Stratified by star rating with a fixed seed, NOT filtered to the cases that
make the lexicon look good. That matters: if the table only held reviews
where TextBlob failed and the lexicon succeeded, it would be an illustration
of a conclusion rather than evidence for one. The rows where TextBlob scored
exactly 0.0 are flagged in a column instead, so they can be sorted to the top
in the dashboard without the sample itself being rigged.

Scoring is imported from nlp_lexicon rather than reimplemented here - the
tokeniser and the mean-of-hits scoring have to match exactly or the numbers
in this table will quietly disagree with the ones in nlp_results.csv.

run after nlp_lexicon.py (needs reports/nlp_lexicon.csv)

    python src/review_examples.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nlp_lexicon import tokenise, score_comment

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# per star rating. 5 x 60 = 300 rows, which keeps the csv small enough to
# commit and the dash table responsive without paging through thousands
PER_STAR = 60

# comments shorter than this are things like "bom" - true to the dataset,
# but they make a poor demonstration of anything
MIN_CHARS = 25


def load_lexicon():
    lx = pd.read_csv(REPORTS_DIR / "nlp_lexicon.csv")
    return dict(zip(lx["word"], lx["polarity"]))


def main():
    try:
        from textblob import TextBlob
    except ImportError:
        print("textblob is not installed. run: pip install textblob")
        sys.exit(1)

    lex_path = REPORTS_DIR / "nlp_lexicon.csv"
    if not lex_path.exists():
        print(f"missing {lex_path} - run src/nlp_lexicon.py first")
        sys.exit(1)

    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    df = reviews.dropna(subset=["review_comment_message"]).copy()
    df["review_comment_message"] = df["review_comment_message"].str.strip()
    df = df[df["review_comment_message"].str.len() >= MIN_CHARS]
    print(f"reviews with usable comment text: {len(df):,}")

    # stratified sample, fixed seed. taking min() per group so a thin star
    # band doesnt blow up on a sample larger than the group.
    #
    # deliberately a plain loop rather than groupby(...).apply(). from pandas
    # 2.2 the grouping column is excluded from the frame handed to apply, and
    # in pandas 3 it is gone for good - which drops review_score out of the
    # result and blows up two lines later on a KeyError. iterating the groups
    # keeps every column and behaves the same on either version.
    parts = []
    for _, group in df.groupby("review_score"):
        parts.append(group.sample(min(PER_STAR, len(group)),
                                  random_state=RANDOM_STATE))
    df = pd.concat(parts).reset_index(drop=True)
    print(f"sampled: {len(df):,} rows across {df['review_score'].nunique()} star bands")

    print("scoring with textblob...")
    df["textblob_polarity"] = df["review_comment_message"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity)

    print("scoring with the induced lexicon...")
    lexicon = load_lexicon()
    df["tokens"] = df["review_comment_message"].apply(tokenise)
    df["lex_score"] = df["tokens"].apply(lambda t: score_comment(t, lexicon))

    # which lexicon words actually fired, so the modal can show the working
    # rather than just a number
    df["lex_words"] = df["tokens"].apply(
        lambda t: ", ".join(dict.fromkeys(w for w in t if w in lexicon)) or "")

    df["textblob_zero"] = df["textblob_polarity"] == 0
    df["n_chars"] = df["review_comment_message"].str.len()

    out = df[["review_id", "order_id", "review_score", "review_comment_message",
              "textblob_polarity", "lex_score", "lex_words",
              "textblob_zero", "n_chars"]].reset_index(drop=True)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "review_examples.csv"
    out.to_csv(out_path, index=False)

    zero = out["textblob_zero"].mean()
    scored = out["lex_score"].notna().mean()
    print(f"\nTextBlob returned exactly 0.0 on {zero*100:.1f}% of the sample")
    print(f"the induced lexicon could score {scored*100:.1f}% of it")
    print(f"saved {len(out):,} rows to {out_path}")


if __name__ == "__main__":
    main()