"""
nlp_lexicon.py

Builds a Portuguese sentiment lexicon from the olist reviews themselves,
then uses it to classify the comment text.

WHY BUILD ONE RATHER THAN IMPORT ONE
TextBlob scored 14.8% agreement against a 64.7% majority baseline on
this text (see sentiment_validation.py) because its lexicon is English
and the reviews are Portuguese. The obvious fix is to import a
portuguese lexicon, but there is a better option sitting in the data:
every comment already carries a star rating, so the dataset labels
itself. Words that show up far more often in 1-2 star comments than in
4-5 star ones are negative words, by definition, in this domain.

This is lexicon induction - a standard NLP technique. The advantage over
an off-the-shelf lexicon is that it learns marketplace vocabulary that a
general dictionary would miss, and it needs no model download.

NO LEAKAGE
The lexicon is built on a training split and evaluated on a held-out
split. Building it on all the data and then scoring that same data would
be circular - the lexicon would have already seen the answers.

METHOD
  1. tokenise, lowercase, strip punctuation and portuguese stopwords
  2. count each word in negative comments and positive comments
  3. score each word by log odds ratio, with smoothing so a word seen
     twice doesnt outrank one seen five hundred times
  4. keep the strongest words in each direction as the lexicon
  5. score a comment as the mean polarity of its known words

run after data_pipeline.py, before or after seller_model.py - nothing
else depends on it
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
TRAIN_FRAC = 0.7

# a word must appear this often to earn a place - below this the log odds
# is dominated by noise
MIN_COUNT = 20

# how many words to keep in each direction
LEXICON_SIZE = 250

# comments shorter than this after cleaning carry too little to score
MIN_TOKENS = 2

# common portuguese function words. these appear everywhere regardless of
# sentiment, so they only add noise
STOPWORDS = {
    "a","o","e","de","do","da","em","um","uma","os","as","no","na","por",
    "para","com","que","se","na","mais","como","mas","foi","ao","ele","ela",
    "das","dos","seu","sua","ou","ser","quando","muito","ja","eu","tambem",
    "so","pelo","pela","ate","isso","entre","depois","sem","mesmo","aos",
    "seus","quem","nas","me","esse","eles","estao","voce","tinha","foram",
    "essa","num","nem","suas","meu","minha","numa","pelos","elas","qual",
    "nos","lhe","deles","essas","esses","pelas","este","dele","tu","te",
    "voces","vos","lhes","meus","minhas","teu","tua","teus","tuas","nosso",
    "nossa","nossos","nossas","dela","delas","esta","estes","estas","aquele",
    "aquela","aqueles","aquelas","isto","aquilo","estou","esta","estamos",
    "estive","havia","sou","somos","era","eram","fui","tem","temos","tinham",
    "tenho","teve","tive","vai","vou","vamos","ia","the","of","to","and",
}

TOKEN_RE = re.compile(r"[a-zà-ú]+")


def tokenise(text):
    if not isinstance(text, str):
        return []
    words = TOKEN_RE.findall(text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def build_lexicon(neg_docs, pos_docs):
    """log odds ratio per word, negative minus positive.

    smoothing matters here. a word appearing twice, both times in a
    one star review, would otherwise look more negative than a word
    appearing four hundred times with an 80/20 split - which is
    obviously wrong. adding one to every count pulls rare words back
    toward neutral.
    """
    neg_counts = Counter(w for doc in neg_docs for w in doc)
    pos_counts = Counter(w for doc in pos_docs for w in doc)

    neg_total = sum(neg_counts.values())
    pos_total = sum(pos_counts.values())

    scores = {}
    for word in set(neg_counts) | set(pos_counts):
        n, p = neg_counts[word], pos_counts[word]
        if n + p < MIN_COUNT:
            continue
        # probability of the word given each class, smoothed
        pn = (n + 1) / (neg_total + len(neg_counts) + 1)
        pp = (p + 1) / (pos_total + len(pos_counts) + 1)
        scores[word] = math.log(pn / pp)

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    positive = ranked[:LEXICON_SIZE]          # most negative log odds
    negative = ranked[-LEXICON_SIZE:]         # most positive log odds

    lexicon = {}
    for word, s in positive:
        lexicon[word] = s
    for word, s in negative:
        lexicon[word] = s
    return lexicon, scores


def score_comment(tokens, lexicon):
    hits = [lexicon[w] for w in tokens if w in lexicon]
    if not hits:
        return None
    # mean rather than sum, so a long comment isnt automatically more
    # extreme than a short one
    return float(np.mean(hits))


def classify(score, cutoff):
    if score is None:
        return "neutral"
    if score > cutoff:
        return "negative"
    if score < -cutoff:
        return "positive"
    return "neutral"


def star_class(s):
    if s <= 2:
        return "negative"
    if s == 3:
        return "neutral"
    return "positive"


def main():
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    df = reviews.dropna(subset=["review_comment_message"]).copy()
    df = df[df["review_comment_message"].str.strip() != ""]
    print(f"reviews with comment text: {len(df):,}")

    df["tokens"] = df["review_comment_message"].apply(tokenise)
    df = df[df["tokens"].map(len) >= MIN_TOKENS]
    print(f"after tokenising and dropping very short comments: {len(df):,}")

    # held out split - the lexicon never sees the test comments
    rng = np.random.default_rng(RANDOM_STATE)
    mask = rng.random(len(df)) < TRAIN_FRAC
    train, test = df[mask], df[~mask]
    print(f"train {len(train):,} | test {len(test):,}")

    neg_docs = train.loc[train["review_score"] <= 2, "tokens"]
    pos_docs = train.loc[train["review_score"] >= 4, "tokens"]
    print(f"training comments: {len(neg_docs):,} negative, {len(pos_docs):,} positive")

    lexicon, all_scores = build_lexicon(neg_docs, pos_docs)
    print(f"lexicon: {len(lexicon)} words "
          f"(from {len(all_scores):,} appearing at least {MIN_COUNT} times)")

    ranked = sorted(lexicon.items(), key=lambda kv: kv[1])
    print("\nstrongest positive words learned:")
    for w, s in ranked[:12]:
        print(f"  {w:16} {s:+.2f}")
    print("\nstrongest negative words learned:")
    for w, s in ranked[-12:][::-1]:
        print(f"  {w:16} {s:+.2f}")

    # the neutral band cutoff is tuned on the TRAINING split, never the test
    # one. an arbitrary percentile would work but there is no reason to guess
    # when the training labels are right there. optimising macro F1 rather
    # than plain agreement, because the classes are badly imbalanced - most
    # reviews are positive, so a cutoff that calls everything positive would
    # score well on agreement while being useless
    from sklearn.metrics import f1_score

    train = train.copy()
    train["lex_score"] = train["tokens"].apply(lambda t: score_comment(t, lexicon))
    train["star_class"] = train["review_score"].apply(star_class)
    tr = train.dropna(subset=["lex_score"])

    best = (0.2, -1.0)
    for c in np.linspace(0.02, 1.5, 60):
        pred = tr["lex_score"].apply(lambda s: classify(s, c))
        f1 = f1_score(tr["star_class"], pred, average="macro", zero_division=0)
        if f1 > best[1]:
            best = (float(c), f1)
    cutoff, train_f1 = best
    print(f"\nneutral band cutoff tuned on training split: +/-{cutoff:.3f} "
          f"(training macro F1 {train_f1:.3f})")

    test = test.copy()
    test["lex_score"] = test["tokens"].apply(lambda t: score_comment(t, lexicon))
    test["lex_class"] = test["lex_score"].apply(lambda s: classify(s, cutoff))
    test["star_class"] = test["review_score"].apply(star_class)

    coverage = test["lex_score"].notna().mean()
    agree = (test["lex_class"] == test["star_class"]).mean()
    majority = test["star_class"].value_counts(normalize=True).max()

    macro_f1 = f1_score(test["star_class"], test["lex_class"],
                        average="macro", zero_division=0)
    # what always guessing the majority class would score on macro F1 -
    # the honest comparison when the classes are this lopsided
    majority_label = test["star_class"].value_counts().idxmax()
    majority_f1 = f1_score(test["star_class"],
                           [majority_label] * len(test),
                           average="macro", zero_division=0)

    print(f"\n{'='*62}")
    print(f"comments containing at least one lexicon word: {coverage*100:.1f}%")
    print(f"agreement with the star rating:                {agree*100:.1f}%")
    print(f"majority class baseline:                       {majority*100:.1f}%")
    print(f"difference:                                    {(agree-majority)*100:+.1f} points")
    print(f"")
    print(f"macro F1 (the metric that matters here):       {macro_f1:.3f}")
    print(f"macro F1 if you always guess '{majority_label}':    {majority_f1:.3f}")
    print(f"{'='*62}")
    print("(TextBlob managed 14.8% agreement against the same baseline)")

    print("\nactual class vs lexicon class:")
    matrix = pd.crosstab(test["star_class"], test["lex_class"],
                         normalize="index").round(3)
    print(matrix.to_string())

    neg = test[test["star_class"] == "negative"]
    if len(neg):
        caught = (neg["lex_class"] == "negative").mean()
        print(f"\nof 1-2 star reviews, correctly called negative: {caught*100:.1f}%")
        print("(TextBlob caught 1.8%)")

    # per customer feature, in case it earns a place in the model
    scored = df.copy()
    scored["lex_score"] = scored["tokens"].apply(lambda t: score_comment(t, lexicon))
    per_order = scored[["order_id", "lex_score"]].dropna()

    REPORTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"word": w, "polarity": round(s, 4),
          "direction": "negative" if s > 0 else "positive"}
         for w, s in sorted(lexicon.items(), key=lambda kv: -abs(kv[1]))]
    ).to_csv(REPORTS_DIR / "nlp_lexicon.csv", index=False)

    pd.DataFrame([{
        "lexicon_words": len(lexicon),
        "train_comments": len(train),
        "test_comments": len(test),
        "coverage_pct": round(coverage * 100, 1),
        "agreement_pct": round(agree * 100, 1),
        "majority_baseline_pct": round(majority * 100, 1),
        "lift_over_baseline": round((agree - majority) * 100, 1),
        "macro_f1": round(float(macro_f1), 3),
        "majority_macro_f1": round(float(majority_f1), 3),
        "textblob_agreement_pct": 14.8,
        "negative_recall_pct": round(
            (neg["lex_class"] == "negative").mean() * 100, 1) if len(neg) else None,
    }]).to_csv(REPORTS_DIR / "nlp_results.csv", index=False)

    matrix.to_csv(REPORTS_DIR / "nlp_confusion.csv")
    per_order.to_csv(PROCESSED_DIR / "text_sentiment.csv", index=False)

    print(f"\nsaved nlp_lexicon.csv, nlp_results.csv, nlp_confusion.csv")
    print(f"saved per-order text sentiment for {len(per_order):,} orders")


if __name__ == "__main__":
    main()