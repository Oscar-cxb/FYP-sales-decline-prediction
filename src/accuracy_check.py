"""
accuracy_check.py

Measures accuracy for every seller model - at the operating threshold, at each
model's own best-accuracy threshold, and against the majority-class baseline.

The baseline is the point. At a 38.3% decline rate a model that predicts "no
decline" for every seller already scores 61.7%, so an accuracy figure quoted
on its own says nothing about whether a model separates the two groups. This
is why the seller model is evaluated on AUC and precision@N and read as a
ranked worklist rather than a hard flag.

WRITES
  reports/accuracy_by_model.csv

Nothing is retrained. Same folds and seed as seller_model.py, so the numbers
reproduce.

    python src/accuracy_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seller_model import load_panel, make_models, cross_validate, REPORTS_DIR

DEFAULT_T = 0.35

df, feats = load_panel()
cv, oof = cross_validate(df, feats, make_models())

rows = []
for name in cv["model"]:
    d = oof[oof["model"] == name]
    y, p = d["y"].to_numpy(), d["p"].to_numpy()

    best_acc, best_t = 0.0, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        acc = float(((p >= t).astype(int) == y).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t

    base_rate = float(y.mean())
    rows.append({
        "model": name,
        "accuracy_at_0.35": float(((p >= DEFAULT_T).astype(int) == y).mean()),
        "best_accuracy": best_acc,
        "at_threshold": round(float(best_t), 2),
        "majority_baseline": max(base_rate, 1.0 - base_rate),
        "base_rate": base_rate,
    })

out = pd.DataFrame(rows).sort_values("best_accuracy", ascending=False)
print(out.to_string(index=False))

path = REPORTS_DIR / "accuracy_by_model.csv"
out.to_csv(path, index=False)
print(f"\nsaved {path}")