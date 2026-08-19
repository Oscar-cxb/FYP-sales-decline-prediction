"""
sensitivity_check.py

Sweeps each simulator lever across its range and prints what the model
actually does with it.

WHY
The simulator showed risk going DOWN when late deliveries were pushed
up, which is the opposite of what anyone would expect. That is either
a bug in how the change propagates, or a real property of the model
worth reporting. Guessing is not good enough, so this measures it.

It also checks whether the value being tested is inside the range the
model was trained on. Tree models cannot extrapolate: outside the
training range they just return whatever the nearest leaf says, which
is not a prediction so much as an artefact.

Prints a table per lever. Nothing else depends on this - it is a
diagnostic, and its output belongs in chapter 5.

run after seller_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# same levers the simulator exposes
LEVERS = [
    ("revenue", "revenue", None),
    ("avg_review_score", "average review score", (1.0, 5.0)),
    ("pct_negative_reviews", "share of negative reviews", (0.0, 1.0)),
    ("pct_late_deliveries", "share of late deliveries", (0.0, 1.0)),
    ("avg_delivery_delay", "average delivery delay (days)", (-40.0, 40.0)),
    ("rfm_recency", "customer recency (days)", (1.0, 120.0)),
]

STEPS = 9


def apply_change(row, feature, new_value):
    """identical to the simulator's version - level plus its momentum."""
    if feature not in row:
        return row
    delta = new_value - row[feature]
    if delta == 0:
        return row

    row[feature] = new_value
    d_col = f"d_{feature}"
    prev = None
    if d_col in row:
        prev = row[feature] - delta - row[d_col]
        row[d_col] = row[d_col] + delta
    if f"roll3_{feature}" in row:
        row[f"roll3_{feature}"] += delta / 3.0
    if f"trend_{feature}" in row:
        row[f"trend_{feature}"] += delta / 2.0
    if f"pct_{feature}" in row and prev not in (None, 0):
        row[f"pct_{feature}"] = (row[feature] - prev) / prev
    return row


def predict(model, cols, row):
    X = pd.DataFrame([[row.get(c, 0.0) for c in cols]], columns=cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return float(model.predict_proba(X)[0, 1])


def main():
    model = joblib.load(MODELS_DIR / "seller_best_model.pkl")
    cols = joblib.load(MODELS_DIR / "seller_feature_columns.pkl")
    scored = pd.read_csv(PROCESSED_DIR / "seller_risk_table.csv")
    panel = pd.read_csv(PROCESSED_DIR / "seller_panel.csv")

    if "risk_rank" not in scored.columns:
        scored["risk_rank"] = scored["risk_score"].rank(ascending=False,
                                                        method="first").astype(int)

    # a few sellers across the risk range, not just the top one
    picks = scored.sort_values("risk_rank").iloc[[0, len(scored)//4,
                                                  len(scored)//2]].copy()

    rows = []
    for _, seller in picks.iterrows():
        base_row = {k: float(v) for k, v in seller.items()
                    if isinstance(v, (int, float, np.number))}
        base_risk = predict(model, cols, dict(base_row))

        print("\n" + "=" * 74)
        print(f"seller {str(seller['seller_id'])[:12]}  rank #{int(seller['risk_rank'])}  "
              f"base risk {base_risk:.3f}")
        print("=" * 74)

        for feat, label, fixed in LEVERS:
            if feat not in base_row:
                continue

            actual = base_row[feat]
            if fixed:
                lo, hi = fixed
                lo, hi = min(lo, actual), max(hi, actual)
            else:
                lo, hi = actual * 0.3, actual * 2.0

            # what the model actually saw during training
            train_lo = float(panel[feat].quantile(0.01))
            train_hi = float(panel[feat].quantile(0.99))

            print(f"\n  {label}")
            print(f"    actual {actual:,.2f} | training range "
                  f"{train_lo:,.2f} to {train_hi:,.2f}")

            line = []
            risks = []
            for v in np.linspace(lo, hi, STEPS):
                r = predict(model, cols, apply_change(dict(base_row), feat, float(v)))
                risks.append(r)
                flag = "" if train_lo <= v <= train_hi else "*"
                line.append(f"{v:,.1f}{flag}->{r:.3f}")
                rows.append({"seller": str(seller["seller_id"])[:12],
                             "feature": feat, "value": round(float(v), 3),
                             "risk": round(r, 4),
                             "outside_training_range": not (train_lo <= v <= train_hi)})

            print("    " + "  ".join(line))

            spread = max(risks) - min(risks)
            direction = ("rises with the lever" if risks[-1] > risks[0] + 0.005
                         else "falls with the lever" if risks[-1] < risks[0] - 0.005
                         else "barely moves")
            print(f"    -> risk {direction}, total spread {spread:.3f}")

    print("\n" + "=" * 74)
    print("* marks a value outside the 1st-99th percentile of the training data.")
    print("  tree models do not extrapolate - beyond that range the prediction")
    print("  is whatever the nearest leaf holds, not a meaningful estimate.")
    print("=" * 74)

    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "sensitivity_sweep.csv", index=False)
    print("\nsaved reports/sensitivity_sweep.csv")


if __name__ == "__main__":
    main()