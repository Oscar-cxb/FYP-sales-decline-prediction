"""
segment_association.py

OBJECTIVE 3
"To investigate the association between RFM-based purchasing patterns and
customer sentiment scores in predicting sales decline risk across different
customer segments."

classifier.py already answers the first half of that by construction. The
label is sentiment-derived (at_risk = average review score below 3) and every
feature the model sees is behavioural - RFM, delivery, payment, category,
segment - because the sentiment columns are dropped immediately after the
label is built, to avoid the leakage documented in classifier.py. So the
model's performance IS a measurement of how strongly RFM purchasing patterns
predict customer sentiment outcomes.

What classifier.py does not do is break that measurement down per segment,
which is the clause this file closes. For every RFM segment it reports:

  - how much of the prediction each feature block drives (mean |SHAP| share)
  - the model's AUC within that segment
  - that segment's own dissatisfaction rate

All of it is measured on the held-out test split only, never on rows the model
was fitted on.

Reading the output: rfm_share is how much of the model's decision inside that
segment comes from recency / frequency / monetary. A segment where RFM drives
a large share is one where purchasing pattern alone carries the sentiment
signal. A segment where delivery dominates is one where service failure, not
purchasing behaviour, is what turns customers negative. That contrast is the
association the objective asks about, per segment.

WRITES
  reports/segment_association.csv

Nothing is retrained and no modelling decision is made here. The model this
reads is the one classifier.py already saved.

run after classifier.py, from the project root:

    python src/segment_association.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import (build_feature_set, train_test_split_data,
                        MODELS_DIR, REPORTS_DIR, RANDOM_STATE)

# SHAP over all 77k customers is slow and adds nothing - the per-segment means
# are stable long before this many rows. sampled WITHIN each segment so a small
# segment is not swamped by a large one.
#
# RandomForest gets a smaller sample because TreeExplainer is far slower on a
# forest of deep sklearn trees than on the same number of boosted stumps -
# measured at roughly 60x here. the group shares still settle well inside 300
# rows per segment, so the smaller sample costs precision in the third decimal
# and saves twenty minutes
PER_SEGMENT = 1200
PER_SEGMENT_RF = 300

# order is fixed so the csv columns are stable between runs
GROUP_ORDER = ["RFM", "Delivery", "Payment", "Category", "Segment", "Other"]


def feature_group(col):
    """which block a feature belongs to.

    sentiment is deliberately absent from this mapping. in this model it is
    the label, not an input - that is the whole point of the leakage decision
    documented in classifier.py, and it is why the association in objective 3
    is measured as 'how well do behavioural blocks predict sentiment' rather
    than 'how do sentiment and RFM compare as inputs'."""
    if col in ("recency", "frequency", "monetary"):
        return "RFM"
    if col == "avg_delivery_delay":
        return "Delivery"
    if col.startswith("payment_") or col in ("avg_installments", "avg_payment_value"):
        return "Payment"
    if col.startswith("category_"):
        return "Category"
    if col.startswith("segment_"):
        return "Segment"
    return "Other"


def positive_class_shap(vals, n_rows, n_features):
    """pull out the at-risk class.

    three shapes turn up depending on model type and shap version:
      - a list [class0, class1]           older shap, RandomForest
      - a 3d array (rows, features, 2)    newer shap, RandomForest
      - a 2d array (rows, features)       XGBoost
    """
    if isinstance(vals, list):
        return np.asarray(vals[1])

    vals = np.asarray(vals)
    if vals.ndim == 3:
        return vals[:, :, 1]
    if vals.shape == (n_rows, n_features):
        return vals

    raise ValueError(f"unexpected shap output shape {vals.shape}, "
                     f"expected ({n_rows}, {n_features})")


def load_best_model():
    """whichever model won on AUC, same rule classifier.py uses."""
    comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    name = str(comp.loc[comp["auc"].idxmax(), "model"])
    filename = "rf_model.pkl" if name == "Random Forest" else "xgb_model.pkl"
    return joblib.load(MODELS_DIR / filename), name


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    df, df_display = build_feature_set()
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")

    model, best_name = load_best_model()
    print(f"\nusing {best_name} (selected on AUC, same rule as classifier.py)")

    # get_dummies produces bool columns in pandas 2+. shap wants numeric, and
    # casting here does not change what the model predicts
    X = df[feature_cols].astype(float)
    y = df["at_risk"].to_numpy()
    seg = df_display["segment_name"].to_numpy()
    probs = model.predict_proba(X)[:, 1]

    # EVERYTHING BELOW IS RESTRICTED TO THE HELD-OUT TEST SPLIT.
    # the model was fitted on the other 80% of these rows, so an AUC measured
    # over all 77k customers would be scoring the model partly on its own
    # training data and would come out optimistically high. same split, same
    # seed, same stratification as classifier.py, so these are exactly the
    # rows that model never saw
    _, X_test, _, _ = train_test_split_data(df)
    is_test = np.zeros(len(df), dtype=bool)
    is_test[df.index.get_indexer(X_test.index)] = True
    print(f"restricted to the {int(is_test.sum())} held-out test customers")

    segments = [str(s) for s in pd.unique(seg)]

    # stratified sample: up to PER_SEGMENT rows from each segment, one
    # explainer pass over the lot
    per_segment = PER_SEGMENT_RF if best_name == "Random Forest" else PER_SEGMENT

    rng = np.random.default_rng(RANDOM_STATE)
    picks = []
    for s in segments:
        pool = np.flatnonzero((seg == s) & is_test)
        take = min(per_segment, len(pool))
        picks.append(rng.choice(pool, take, replace=False))
    idx = np.concatenate(picks)

    print(f"explaining {len(idx)} sampled customers across "
          f"{len(segments)} segments "
          f"(up to {per_segment} per segment)...")
    if best_name == "Random Forest":
        print("  RandomForest selected - this takes a few minutes, "
              "TreeExplainer is slow on forests")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X.iloc[idx])
    sv = np.abs(positive_class_shap(raw, len(idx), len(feature_cols)))

    groups = np.array([feature_group(c) for c in feature_cols])
    masks = {g: (groups == g) for g in GROUP_ORDER}

    rows = []
    for s in segments:
        in_full = (seg == s) & is_test
        in_samp = seg[idx] == s

        row = {
            "segment": s,
            "n_customers": int((seg == s).sum()),
            "n_test": int(in_full.sum()),
            "dissatisfaction_rate": float(y[in_full].mean()),
        }

        # AUC is undefined where a segment holds only one class
        try:
            row["auc"] = float(roc_auc_score(y[in_full], probs[in_full]))
        except ValueError:
            row["auc"] = float("nan")

        block = sv[in_samp]
        totals = {g: (float(block[:, masks[g]].sum(axis=1).mean())
                      if masks[g].any() else 0.0)
                  for g in GROUP_ORDER}
        grand = sum(totals.values()) or 1.0
        for g in GROUP_ORDER:
            row[f"{g.lower()}_share"] = totals[g] / grand

        row["n_explained"] = int(in_samp.sum())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("dissatisfaction_rate",
                                         ascending=False).reset_index(drop=True)

    # every share is printed, including segment, so the columns visibly sum to
    # 1 and nothing looks quietly dropped. note that the segment one-hots are
    # constant inside a segment, so their share is a level effect rather than
    # something separating customers within that segment - the blocks to read
    # against each other are RFM, delivery, payment and category
    show = ["segment", "n_customers", "n_test", "dissatisfaction_rate", "auc",
            "rfm_share", "delivery_share", "payment_share", "category_share",
            "segment_share"]
    print("\nper-segment association "
          "(share of the prediction driven by each feature block):")
    print(out[show].to_string(index=False,
                              float_format=lambda v: f"{v:.3f}"))

    path = REPORTS_DIR / "segment_association.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved {path}  ({len(out)} segments)")


if __name__ == "__main__":
    main()