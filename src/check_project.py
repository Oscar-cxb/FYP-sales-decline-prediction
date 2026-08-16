"""
check_project.py

Sanity check over the whole project. Doesnt train anything, just looks
at what exists and whether the pieces agree with each other.

Run it before a supervisor meeting or a demo - it catches the things
that are embarrassing to find live, like a page reading a column that
isnt in the csv any more.

    python src/check_project.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SRC = ROOT / "src"
PAGES = ROOT / "dashboard" / "pages"

ok, warn, bad = [], [], []


def check(cond, good, issue, fatal=True):
    if cond:
        ok.append(good)
    else:
        (bad if fatal else warn).append(issue)


# ---------------------------------------------------------- raw data

raw_needed = ["olist_orders_dataset.csv", "olist_order_items_dataset.csv",
              "olist_customers_dataset.csv", "olist_order_reviews_dataset.csv",
              "olist_products_dataset.csv", "olist_order_payments_dataset.csv"]
for f in raw_needed:
    check((RAW / f).exists(), f"raw/{f}", f"MISSING raw/{f} - download from kaggle")

# ------------------------------------------------------ processed data

expected = {
    "merged_dataset.csv": ["order_id", "customer_unique_id", "price",
                           "order_purchase_timestamp"],
    "rfm_features.csv": ["customer_unique_id", "recency", "frequency",
                         "monetary", "avg_delivery_delay"],
    "sentiment_features.csv": ["customer_unique_id", "avg_review_score",
                               "pct_negative_reviews"],
    "extra_features.csv": ["customer_unique_id", "dominant_category",
                           "dominant_payment_type"],
    "customer_segments.csv": ["customer_unique_id", "segment_name"],
    "customer_risk_table.csv": ["customer_unique_id", "segment_name", "monetary",
                                "recency", "avg_delivery_delay", "risk_score",
                                "at_risk"],
    "seller_panel.csv": ["seller_id", "month", "revenue", "decline"],
    "seller_risk_table.csv": ["seller_id", "month", "revenue", "risk_score",
                              "avg_review_score", "pct_late_deliveries",
                              "d_revenue"],
}

frames = {}
for name, cols in expected.items():
    p = DATA / name
    if not p.exists():
        bad.append(f"MISSING data/processed/{name}")
        continue
    df = pd.read_csv(p, nrows=5000)
    frames[name] = df
    missing = [c for c in cols if c not in df.columns]
    check(not missing, f"{name} has the columns the pages expect",
          f"{name} is missing columns {missing}")

# risk_rank is written by seller_model.py; pages fall back if absent
if "seller_risk_table.csv" in frames:
    has_rank = "risk_rank" in frames["seller_risk_table.csv"].columns
    check(has_rank, "seller_risk_table has risk_rank",
          "seller_risk_table has no risk_rank - pages derive it, but rerun "
          "seller_model.py to have it stored", fatal=False)

# ------------------------------------------------------------ models

for f in ["seller_best_model.pkl", "seller_feature_columns.pkl",
          "seller_model_meta.pkl"]:
    check((MODELS / f).exists(), f"models/{f}", f"MISSING models/{f}")

# the feature list and the panel have to line up or the simulator breaks
if (MODELS / "seller_feature_columns.pkl").exists() and "seller_panel.csv" in frames:
    import joblib
    cols = joblib.load(MODELS / "seller_feature_columns.pkl")
    panel_cols = set(pd.read_csv(DATA / "seller_panel.csv", nrows=5).columns)
    missing = [c for c in cols if c not in panel_cols]
    check(not missing, f"model's {len(cols)} features all exist in the panel",
          f"model expects features not in seller_panel.csv: {missing[:6]}")

# ----------------------------------------------------------- reports

report_files = ["seller_cv_results.csv", "ablation_results.csv",
                "seller_shap_importance.csv", "seller_shap_groups.csv",
                "capacity_curve.csv", "threshold_curve.csv"]
for f in report_files:
    check((REPORTS / f).exists(), f"reports/{f}", f"MISSING reports/{f}", fatal=False)

# the headline numbers should agree across files
if (REPORTS / "seller_cv_results.csv").exists() and (REPORTS / "ablation_results.csv").exists():
    cv = pd.read_csv(REPORTS / "seller_cv_results.csv")
    ab = pd.read_csv(REPORTS / "ablation_results.csv")
    best_cv = cv["mean_auc"].max()
    all_feat = ab.loc[ab["feature_set"] == "All features", "mean_auc"]
    if len(all_feat):
        agree = abs(best_cv - all_feat.iloc[0]) < 0.005
        check(agree,
              f"CV best ({best_cv:.4f}) matches ablation 'All features' "
              f"({all_feat.iloc[0]:.4f})",
              f"MISMATCH: CV best {best_cv:.4f} but ablation All features "
              f"{all_feat.iloc[0]:.4f} - one of them is from an older run, "
              f"rerun seller_model.py")

if (REPORTS / "seller_shap_groups.csv").exists():
    g = pd.read_csv(REPORTS / "seller_shap_groups.csv")
    check(abs(g["share_pct"].sum() - 100) < 0.1,
          "SHAP group shares sum to 100%",
          f"SHAP group shares sum to {g['share_pct'].sum():.1f}%, not 100")

# --------------------------------------------------------- dead code

for f in ["evaluation.py", "what_if_simulator.py"]:
    p = SRC / f
    if p.exists() and p.stat().st_size < 600:
        warn.append(f"src/{f} is an empty stub - delete it or fill it in, "
                    f"an empty file in the submission looks unfinished")

for f in ["seller_classifier.py", "ablation.py"]:
    if (SRC / f).exists():
        warn.append(f"src/{f} was replaced by seller_model.py - delete it")

# ------------------------------------------------------------- pages

for f in ["overview.py", "customers.py", "sellers.py", "risk_analysis.py",
          "segments.py", "simulator.py", "interventions.py"]:
    p = PAGES / f
    if not p.exists():
        bad.append(f"MISSING dashboard/pages/{f}")
    elif p.stat().st_size < 400:
        warn.append(f"dashboard/pages/{f} is still a stub")

# a page that reads a column the csv doesnt have will only fail when opened
if "customer_risk_table.csv" in frames:
    seg = frames["customer_risk_table.csv"]
    check("segment_name" in seg.columns,
          "segments page will find segment_name",
          "customer_risk_table.csv has no segment_name - segments page will "
          "show its fallback message; rerun clustering.py then classifier.py")

# ------------------------------------------------------------ output

print("=" * 66)
print(f"{len(ok)} checks passed")
print("=" * 66)

if warn:
    print(f"\n{len(warn)} things worth tidying:")
    for w in warn:
        print(f"  - {w}")

if bad:
    print(f"\n{len(bad)} PROBLEMS:")
    for b in bad:
        print(f"  ! {b}")
    print("\nfix these before demoing")
    sys.exit(1)

print("\nno blocking problems found")