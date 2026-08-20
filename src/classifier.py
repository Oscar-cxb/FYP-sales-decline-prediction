"""
classifier.py

Combines RFM + segments + sentiment into one feature table, trains
Random Forest (main model) and XGBoost (comparison model) to predict
at_risk, applies SMOTE to the training set only, and runs SHAP for
explainability.

at_risk = customer was dissatisfied (avg review score < 3), built
from sentiment data. features are recency/frequency/monetary/delivery
delay/segment ONLY - sentiment columns are dropped after building the
label so they cant also be inputs. see build_feature_set() and the
big comment in rfm_features.py for why it took 3 tries to get this
label right (recency threshold and RFM-quintile-score versions both
leaked, "no repeat purchase" version was 99%+ positive and useless).

this is the big one - everything else (dashboard, simulator,
interventions) depends on the model this produces

run after rfm_features.py, clustering.py, sentiment.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42

FEATURE_COLS = [
    "recency", "frequency", "monetary", "avg_delivery_delay", "segment_name",
    "dominant_category", "dominant_payment_type", "avg_installments", "avg_payment_value",
]  # note: this list is just documentation, actual feature_cols in train_test_split_data
   # is computed dynamically from whatever columns exist after one-hot encoding

# review_score <= 2 is "negative" per my own IR's sentiment definition
# (section 2.2.2). using average review score < 3 as the dissatisfaction
# cutoff for the at_risk label
DISSATISFACTION_THRESHOLD = 3.0


def build_feature_set():
    # at_risk went through 3 broken definitions before this one (see the big
    # comment block in rfm_features.py for the full story). short version:
    # cant define it from recency/frequency/monetary since those ARE the
    # model's features (leakage), and "no repeat purchase" is useless here
    # since 99%+ of customers dont come back regardless of happiness (one-shot
    # marketplace). landed on: at_risk = customer was actually dissatisfied,
    # based on their real average review rating. this is genuine outcome data,
    # not something derivable from RFM/delivery numbers.
    rfm_segments = pd.read_csv(PROCESSED_DIR / "customer_segments.csv")
    sentiment = pd.read_csv(PROCESSED_DIR / "sentiment_features.csv")
    extra = pd.read_csv(PROCESSED_DIR / "extra_features.csv")

    df = rfm_segments.merge(sentiment, on="customer_unique_id", how="left")
    df = df.merge(extra, on="customer_unique_id", how="left")

    # a handful of customers might not have matched an order in extra_features
    # (shouldnt be many, but just in case) - fill numeric with mean, categorical with "unknown"
    for col in ["avg_installments", "avg_payment_value"]:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"filling {missing} missing {col} with column mean")
            df[col] = df[col].fillna(df[col].mean())
    for col in ["dominant_category", "dominant_payment_type"]:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"filling {missing} missing {col} with 'unknown'")
            df[col] = df[col].fillna("unknown")

    # ~0.8% of customers had no review in the window (checked this in
    # sentiment.py), fill with dataset averages instead of dropping them
    for col in ["avg_review_score", "pct_negative_reviews", "avg_text_polarity"]:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"filling {missing} missing {col} with column mean")
            df[col] = df[col].fillna(df[col].mean())

    df["at_risk"] = (df["avg_review_score"] < DISSATISFACTION_THRESHOLD).astype(int)
    print(f"at-risk (dissatisfied) customers: {df['at_risk'].sum()} ({df['at_risk'].mean()*100:.1f}%)")

    # IMPORTANT: dropping the sentiment columns from the feature set now that
    # theyve been used to build the label. they cant also be inputs, thats
    # exactly the same leakage mistake all over again just with new columns.
    # model only gets to use recency/frequency/monetary/delay/segment - stuff
    # that doesnt already know the answer
    df = df.drop(columns=["avg_review_score", "pct_negative_reviews", "avg_text_polarity"])

    # keeping a copy with readable category/segment names BEFORE one-hot
    # encoding - dashboard needs "Champions" not a bunch of 0/1 segment_
    # columns, so saving this separately for the customer_risk_table export
    df_display = df.copy()

    # segment_name, dominant_category, dominant_payment_type are all categorical,
    # one-hot encode all of them so both models can use them
    df = pd.get_dummies(df, columns=["segment_name", "dominant_category", "dominant_payment_type"], prefix=["segment", "category", "payment"])

    print(f"final feature table shape: {df.shape}")
    return df, df_display


def train_test_split_data(df):
    feature_cols = [c for c in df.columns if c not in ["customer_unique_id", "cluster", "at_risk"]]
    X = df[feature_cols]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote_train_only(X_train, y_train):
    # important - only balancing the TRAINING set. if we did this before the
    # split, or on the test set too, we'd be testing against synthetic data
    # which makes the accuracy numbers meaningless
    print(f"before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"after SMOTE: {y_train_res.value_counts().to_dict()}")
    return X_train_res, y_train_res


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds))
    print(f"F1: {f1:.4f}  AUC-ROC: {auc:.4f}")

    return {"model": name, "f1": f1, "auc": auc}


def run_shap_explainability(model, X_train, X_test):
    # using TreeExplainer since both RF and XGBoost are tree based,
    # way faster than the generic KernelExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def summarize_shap(shap_values, feature_names):
    # RF's TreeExplainer returns a list [class0_values, class1_values] for
    # binary classification, XGBoost just returns one array directly.
    # we only care about the "at risk" class (1) either way
    if isinstance(shap_values, list):
        values = shap_values[1]
    else:
        values = shap_values

    importance = np.abs(values).mean(axis=0)
    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return summary


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    df, df_display = build_feature_set()
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    X_train_res, y_train_res = apply_smote_train_only(X_train, y_train)

    rf = train_random_forest(X_train_res, y_train_res)
    xgb_model = train_xgboost(X_train_res, y_train_res)

    rf_results = evaluate(rf, X_test, y_test, "Random Forest")
    xgb_results = evaluate(xgb_model, X_test, y_test, "XGBoost")

    comparison = pd.DataFrame([rf_results, xgb_results])
    print("\nmodel comparison:")
    print(comparison)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # running SHAP on whichever model actually performed better instead of
    # just always assuming its RF
    best_model = rf if rf_results["auc"] >= xgb_results["auc"] else xgb_model
    best_name = "Random Forest" if best_model is rf else "XGBoost"
    print(f"\nbest model by AUC: {best_name}")

    explainer, shap_values = run_shap_explainability(best_model, X_train_res, X_test)

    shap_summary = summarize_shap(shap_values, X_test.columns.tolist())
    print("\nSHAP feature importance (which features actually drive at-risk predictions):")
    print(shap_summary)
    shap_summary.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_columns.pkl")

    print("saved rf_model.pkl, xgb_model.pkl, shap_explainer.pkl, feature_columns.pkl")

    # dashboard needs a plain customer-level table (readable segment/category
    # names, not one-hot columns) with a risk SCORE per customer, not just
    # 0/1 - predicting on the FULL feature set (X = same columns as training,
    # in the same order) rather than just the test split, so every customer
    # gets a score, not only the ~20% that ended up in X_test
    feature_cols = list(X_train.columns)
    X_full = df[feature_cols]
    risk_scores = best_model.predict_proba(X_full)[:, 1]

    risk_table = df_display.drop(columns=["cluster"]).copy()
    risk_table["risk_score"] = risk_scores

    out_path = PROCESSED_DIR / "customer_risk_table.csv"
    risk_table.to_csv(out_path, index=False)
    print(f"saved dashboard-ready table to {out_path} ({len(risk_table)} customers)")


if __name__ == "__main__":
    main()