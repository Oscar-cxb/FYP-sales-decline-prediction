"""
model_curves.py

Saves the per-model evidence the Model Evidence page needs to compare all
three seller models on one page.

WHY THIS EXISTS
seller_model.py writes threshold_curve.csv and capacity_curve.csv for the
winning model only. That is the right default - the operating point of the
model actually in use is what the system runs on. But it means the dashboard
can only ever show one model, and a model comparison is a thing a reader
expects to be able to inspect rather than take on trust.

Everything here comes from the out-of-fold predictions cross_validate()
already produces for all three models. Nothing is retrained and no modelling
decision is made in this file.

WRITES
  reports/roc_curve_by_model.csv        fpr / tpr per model, for overlaid ROC
  reports/threshold_curve_by_model.csv  precision / recall / F1 vs threshold
  reports/capacity_curve_by_model.csv   precision@N per model

This is evidence, not a pipeline stage. Nothing downstream depends on it, and
seller_model.py is untouched.

run after seller_model.py, from the project root:

    python src/model_curves.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seller_model import (
    load_panel, make_models, cross_validate,
    tune_threshold, capacity_analysis, REPORTS_DIR,
)

# a full ROC over ~8k predictions is thousands of points, which is more than a
# chart can show and more than a csv should carry. thinning to this many keeps
# the curve shape exact to the eye and the file small enough to commit
ROC_POINTS = 200


def thin_roc(fpr, tpr, n=ROC_POINTS):
    """keep n evenly spaced points, always including the first and last."""
    if len(fpr) <= n:
        return fpr, tpr
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return fpr[idx], tpr[idx]


def build_roc(oof, model_name):
    d = oof[oof["model"] == model_name]
    fpr, tpr, _ = roc_curve(d["y"], d["p"])
    fpr, tpr = thin_roc(fpr, tpr)
    auc = roc_auc_score(d["y"], d["p"])
    return pd.DataFrame({"model": model_name, "fpr": fpr, "tpr": tpr,
                         "auc": auc})


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    # the cross validation is repeated here because seller_model.py holds the
    # out-of-fold predictions in memory and only writes the curves derived
    # from them. same folds, same seed, so the numbers reproduce exactly
    df, feats = load_panel()
    cv, oof = cross_validate(df, feats, make_models())

    rocs, curves, capacities = [], [], []

    for name in cv["model"]:
        print(f"\n=== {name} ===")

        roc = build_roc(oof, name)
        rocs.append(roc)
        print(f"  ROC: {len(roc)} points, AUC {roc['auc'].iloc[0]:.4f}")

        _, curve = tune_threshold(oof, name)
        curves.append(curve.assign(model=name))

        cap, base_rate = capacity_analysis(oof, name)
        capacities.append(cap.assign(model=name, base_rate=base_rate))

    outputs = [
        ("roc_curve_by_model.csv", pd.concat(rocs, ignore_index=True)),
        ("threshold_curve_by_model.csv", pd.concat(curves, ignore_index=True)),
        ("capacity_curve_by_model.csv", pd.concat(capacities, ignore_index=True)),
    ]

    print()
    for filename, frame in outputs:
        path = REPORTS_DIR / filename
        frame.to_csv(path, index=False)
        print(f"saved {path}  ({len(frame)} rows)")

    print(f"\nmodels covered: {', '.join(cv['model'])}")


if __name__ == "__main__":
    main()