"""
seller_model.py  (replaces seller_classifier.py)

Trains and evaluates the seller decline model properly.

WHAT CHANGED FROM v1
1. rolling origin cross validation instead of one train/test split.
   with a single cutoff the score depends on which month you happened
   to cut at - if that month was unusual the number is a fluke. this
   trains on months 1..k, tests on k+1, slides forward, repeats. you
   get a mean and a spread, which is a much harder thing to argue with.

2. logistic regression added as a baseline. if the tree models cant
   beat a linear model theres no case for using them, and its better i
   find that out than the examiner.

3. the decision threshold is chosen from the data instead of just
   defaulting to 0.5. 0.5 is only correct if false positives and false
   negatives cost the same, which they dont - missing a declining
   seller is worse than checking on a healthy one.

4. the ablation runs on the rebuilt feature set, so every block now has
   momentum features and the comparison is finally fair. v1 gave
   momentum to Trading only and then concluded Trading dominates,
   which was rigged without me meaning it to be.

run after seller_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
MIN_TRAIN_MONTHS = 8      # need a reasonable history before the first test fold
DROP_COLS = ["seller_id", "month", "decline"]


def feature_blocks(cols):
    """Sort columns into blocks by what they're derived from.

    doing it by string matching on the base name so the momentum
    versions land in the same block as their parent - d_avg_review_score
    is a Sentiment feature, not a separate thing
    """
    rfm_base = ["rfm_recency", "rfm_frequency", "rfm_monetary"]
    sent_base = ["avg_review_score", "pct_negative_reviews"]
    deliv_base = ["avg_delivery_delay", "pct_late_deliveries"]

    def matches(col, bases):
        return any(col == b or col.endswith(f"_{b}") for b in bases)

    blocks = {"RFM": [], "Sentiment": [], "Delivery": [], "Trading": []}
    for c in cols:
        if matches(c, rfm_base):
            blocks["RFM"].append(c)
        elif matches(c, sent_base):
            blocks["Sentiment"].append(c)
        elif matches(c, deliv_base):
            blocks["Delivery"].append(c)
        else:
            blocks["Trading"].append(c)
    return blocks


def load_panel():
    df = pd.read_csv(PROCESSED_DIR / "seller_panel.csv")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df = df.sort_values(["month", "seller_id"]).reset_index(drop=True)
    feats = [c for c in df.columns if c not in DROP_COLS]
    print(f"panel: {len(df):,} rows, {df['seller_id'].nunique():,} sellers, "
          f"{len(feats)} features, decline rate {df['decline'].mean()*100:.1f}%")
    return df, feats


def make_models():
    return {
        # baseline. scaled because logistic regression cares about that
        # and the trees dont
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.5,
            random_state=RANDOM_STATE, eval_metric="logloss"),
    }


def rolling_origin_folds(months):
    """train on everything up to month k, test on k+1, slide forward."""
    folds = []
    for i in range(MIN_TRAIN_MONTHS, len(months) - 1):
        folds.append((months[:i + 1], months[i + 1]))
    return folds


def fit_predict(model, X_tr, y_tr, X_te):
    # SMOTE on the training fold only, every time - never the test fold
    if y_tr.nunique() < 2 or y_tr.value_counts().min() < 6:
        model.fit(X_tr, y_tr)
    else:
        X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
        model.fit(X_res, y_res)
    return model.predict_proba(X_te)[:, 1]


def cross_validate(df, feats, models):
    months = sorted(df["month"].unique())
    folds = rolling_origin_folds(months)
    print(f"\nrolling origin CV: {len(folds)} folds "
          f"(first test month {folds[0][1]}, last {folds[-1][1]})")

    rows, oof = [], []
    for name, proto in models.items():
        aucs = []
        for train_months, test_month in folds:
            tr = df[df["month"].isin(train_months)]
            te = df[df["month"] == test_month]
            if len(te) < 20 or te["decline"].nunique() < 2:
                continue

            model = make_models()[name]
            probs = fit_predict(model, tr[feats], tr["decline"], te[feats])
            aucs.append(roc_auc_score(te["decline"], probs))

            oof.append(pd.DataFrame({
                "model": name, "month": str(test_month),
                "seller_id": te["seller_id"].values,
                "y": te["decline"].values, "p": probs,
            }))

        rows.append({"model": name, "mean_auc": np.mean(aucs),
                     "std_auc": np.std(aucs), "min_auc": np.min(aucs),
                     "max_auc": np.max(aucs), "n_folds": len(aucs)})
        print(f"  {name:22} AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
              f"(range {np.min(aucs):.3f}-{np.max(aucs):.3f})")

    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def capacity_analysis(oof, model_name, capacities=(10, 25, 50, 100)):
    """precision@N - if an account team can only chase N sellers a month,
    how many of those N actually decline?

    WHY THIS INSTEAD OF A THRESHOLD
    tuning the threshold for F1 gave 0.33, which flagged 75% of sellers and
    95% of revenue. technically the best F1, operationally meaningless - you
    cant tell a marketplace that three quarters of its sellers are at risk
    and call that an early warning system.

    ranking is what these systems actually do in practice: sort by risk,
    work down the list until you run out of capacity. and precision@N is
    honest in a way F1 isnt here, because it answers a question someone
    would really ask.

    computed per month then averaged, because a team works a month at a
    time - not across the whole test period at once.
    """
    d = oof[oof["model"] == model_name]
    base_rate = d["y"].mean()

    rows = []
    for n in capacities:
        precs, recs, covered = [], [], []
        for month, grp in d.groupby("month"):
            if len(grp) < n:
                continue
            top = grp.nlargest(n, "p")
            precs.append(top["y"].mean())
            total_pos = grp["y"].sum()
            recs.append(top["y"].sum() / total_pos if total_pos else np.nan)
            covered.append(n / len(grp))

        if not precs:
            continue
        rows.append({
            "capacity": n,
            "precision_at_n": np.mean(precs),
            "recall_at_n": np.nanmean(recs),
            "pct_of_sellers": np.mean(covered),
            "lift_over_base": np.mean(precs) / base_rate,
            "n_months": len(precs),
        })

    out = pd.DataFrame(rows)
    print(f"\ncapacity analysis (base decline rate {base_rate:.3f}):")
    for _, r in out.iterrows():
        print(f"  top {int(r['capacity']):3} sellers/month -> "
              f"precision {r['precision_at_n']:.3f} "
              f"({r['lift_over_base']:.2f}x base), "
              f"recall {r['recall_at_n']:.3f}, "
              f"covers {r['pct_of_sellers']*100:.0f}% of sellers")
    return out, base_rate


def tune_threshold(oof, model_name):
    """pick the cutoff that maximises F1 on the out-of-fold predictions.

    0.5 assumes a false alarm costs the same as a missed decline. it
    doesnt - chasing a healthy seller wastes a phone call, missing a
    declining one loses revenue. so the threshold should move.
    """
    d = oof[oof["model"] == model_name]
    best = (0.5, 0.0)
    rows = []
    for t in np.arange(0.20, 0.75, 0.01):
        pred = (d["p"] >= t).astype(int)
        f1 = f1_score(d["y"], pred)
        rows.append({"threshold": round(t, 2), "f1": f1,
                     "precision": precision_score(d["y"], pred, zero_division=0),
                     "recall": recall_score(d["y"], pred)})
        if f1 > best[1]:
            best = (round(t, 2), f1)

    curve = pd.DataFrame(rows)
    t, f1 = best
    at = curve[curve["threshold"] == t].iloc[0]
    print(f"\nbest threshold for {model_name}: {t:.2f} "
          f"(F1 {f1:.4f}, precision {at['precision']:.3f}, recall {at['recall']:.3f})")
    default = curve[curve["threshold"] == 0.50].iloc[0]
    print(f"  vs default 0.50: F1 {default['f1']:.4f}, "
          f"precision {default['precision']:.3f}, recall {default['recall']:.3f}")
    return t, curve


def run_ablation(df, feats, best_name):
    blocks = feature_blocks(feats)
    print("\nfeature blocks (momentum versions included in their parent block):")
    for k, v in blocks.items():
        print(f"  {k:10} {len(v):2} features")

    sets = {
        "Trading only": blocks["Trading"],
        "RFM only": blocks["RFM"],
        "Sentiment only": blocks["Sentiment"],
        "Delivery only": blocks["Delivery"],
        "RFM + Sentiment": blocks["RFM"] + blocks["Sentiment"],
        "Customer-derived (RFM+Sent+Deliv)": blocks["RFM"] + blocks["Sentiment"] + blocks["Delivery"],
        "All features": feats,
    }

    months = sorted(df["month"].unique())
    folds = rolling_origin_folds(months)

    rows = []
    for label, cols in sets.items():
        aucs = []
        for train_months, test_month in folds:
            tr = df[df["month"].isin(train_months)]
            te = df[df["month"] == test_month]
            if len(te) < 20 or te["decline"].nunique() < 2:
                continue
            model = make_models()[best_name]
            probs = fit_predict(model, tr[cols], tr["decline"], te[cols])
            aucs.append(roc_auc_score(te["decline"], probs))
        rows.append({"feature_set": label, "n_features": len(cols),
                     "mean_auc": np.mean(aucs), "std_auc": np.std(aucs)})
        print(f"  {label:36} AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)


def final_model_and_shap(df, feats, best_name, threshold):
    """refit on everything except the last month, score that last month."""
    months = sorted(df["month"].unique())
    tr = df[df["month"] < months[-1]]
    te = df[df["month"] == months[-1]]

    model = make_models()[best_name]
    probs = fit_predict(model, tr[feats], tr["decline"], te[feats])
    preds = (probs >= threshold).astype(int)

    print(f"\nfinal holdout ({months[-1]}, {len(te)} sellers) at threshold {threshold}:")
    print(classification_report(te["decline"], preds,
                                target_names=["stable", "decline"], zero_division=0))

    fitted = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    summary = None
    try:
        values = shap.TreeExplainer(fitted).shap_values(te[feats])
        if isinstance(values, list):
            values = values[1]
        elif getattr(values, "ndim", 2) == 3:
            values = values[:, :, 1]
        summary = pd.DataFrame({
            "feature": list(feats),
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception as e:
        # logistic regression isnt a tree so TreeExplainer wont take it
        print(f"  (SHAP skipped: {e})")

    scored = te.copy()
    scored["risk_score"] = probs
    scored["predicted_decline"] = preds

    # rank within the month so the dashboard can show a worklist rather than
    # a blanket flag. rank 1 = highest risk
    scored["risk_rank"] = scored["risk_score"].rank(ascending=False, method="first").astype(int)
    return model, summary, scored


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df, feats = load_panel()
    cv, oof = cross_validate(df, feats, make_models())
    cv.to_csv(REPORTS_DIR / "seller_cv_results.csv", index=False)

    best_name = cv.loc[cv["mean_auc"].idxmax(), "model"]
    print(f"\nbest model: {best_name}")

    threshold, curve = tune_threshold(oof, best_name)
    curve.to_csv(REPORTS_DIR / "threshold_curve.csv", index=False)

    capacity, base_rate = capacity_analysis(oof, best_name)
    capacity.to_csv(REPORTS_DIR / "capacity_curve.csv", index=False)

    # 50 a month is the default worklist size - roughly 13% of the sellers in
    # a month, which is a plausible amount of follow-up for one team. its a
    # business assumption not a statistical one, so its stated plainly rather
    # than dressed up as optimal
    recommended_capacity = 50

    ablation = run_ablation(df, feats, best_name)
    ablation.to_csv(REPORTS_DIR / "ablation_results.csv", index=False)

    model, shap_summary, scored = final_model_and_shap(df, feats, best_name, threshold)

    if shap_summary is not None:
        blocks = feature_blocks(feats)
        groups = []
        for name, cols in blocks.items():
            total = shap_summary[shap_summary["feature"].isin(cols)]["mean_abs_shap"].sum()
            groups.append({"group": name, "total_shap": total, "n_features": len(cols)})
        groups = pd.DataFrame(groups).sort_values("total_shap", ascending=False)
        groups["share_pct"] = groups["total_shap"] / groups["total_shap"].sum() * 100

        print("\nSHAP by group:")
        print(groups.to_string(index=False))
        shap_summary.to_csv(REPORTS_DIR / "seller_shap_importance.csv", index=False)
        groups.to_csv(REPORTS_DIR / "seller_shap_groups.csv", index=False)

    joblib.dump(model, MODELS_DIR / "seller_best_model.pkl")
    joblib.dump(list(feats), MODELS_DIR / "seller_feature_columns.pkl")
    joblib.dump({"model": best_name, "threshold": threshold,
                 "capacity": recommended_capacity, "base_rate": float(base_rate)},
                MODELS_DIR / "seller_model_meta.pkl")

    scored["month"] = scored["month"].astype(str)
    scored.to_csv(PROCESSED_DIR / "seller_risk_table.csv", index=False)



    print(f"\nsaved model, threshold ({threshold}), and {len(scored)} scored sellers")


if __name__ == "__main__":
    main()