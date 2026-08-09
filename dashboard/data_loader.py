"""
data_loader.py

Single place that loads all the data/models the dashboard needs.
Every page imports from here instead of each one reading its own
csvs - keeps things consistent and means we only load stuff once.
"""

import pandas as pd
import joblib
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def load_risk_table():
    return pd.read_csv(PROCESSED_DIR / "customer_risk_table.csv")


def load_merged_dataset():
    # for the monthly order trend chart on the overview page
    df = pd.read_csv(PROCESSED_DIR / "merged_dataset.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


def load_shap_importance():
    return pd.read_csv(REPORTS_DIR / "shap_feature_importance.csv")


def load_model_comparison():
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


def load_model():
    # loads whichever model classifier.py decided was best (rf or xgb) -
    # just try xgb first since thats what won last run, fall back to rf
    try:
        return joblib.load(MODELS_DIR / "xgb_model.pkl")
    except FileNotFoundError:
        return joblib.load(MODELS_DIR / "rf_model.pkl")


def load_feature_columns():
    return joblib.load(MODELS_DIR / "feature_columns.pkl")


def load_rfm_scaler():
    return joblib.load(MODELS_DIR / "rfm_scaler.pkl")


def load_kmeans_model():
    return joblib.load(MODELS_DIR / "kmeans_model.pkl")

# ---------------- seller level (the primary model) ----------------

def load_seller_risk_table():
    # scored seller-months from the test period, written by seller_classifier.py
    df = pd.read_csv(PROCESSED_DIR / "seller_risk_table.csv")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def load_seller_panel():
    df = pd.read_csv(PROCESSED_DIR / "seller_panel.csv")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def load_seller_cv_results():
    # written by seller_model.py - one row per model with mean/std AUC
    # across the rolling origin folds
    return pd.read_csv(REPORTS_DIR / "seller_cv_results.csv")


def load_seller_model_meta():
    # which model won and the tuned decision threshold
    return joblib.load(MODELS_DIR / "seller_model_meta.pkl")


def load_threshold_curve():
    return pd.read_csv(REPORTS_DIR / "threshold_curve.csv")


def load_capacity_curve():
    # precision@N for a few worklist sizes - see capacity_analysis()
    # in seller_model.py
    return pd.read_csv(REPORTS_DIR / "capacity_curve.csv")


def load_seller_shap_importance():
    return pd.read_csv(REPORTS_DIR / "seller_shap_importance.csv")


def load_seller_shap_groups():
    return pd.read_csv(REPORTS_DIR / "seller_shap_groups.csv")


def load_ablation_results():
    return pd.read_csv(REPORTS_DIR / "ablation_results.csv")