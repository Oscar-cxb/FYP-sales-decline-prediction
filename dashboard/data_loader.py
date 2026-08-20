"""
data_loader.py

Single place that loads everything the dashboard needs. Every page
imports from here rather than reading its own csvs, so paths live in
one file and a rename only has to happen once.

Nothing here computes anything - the pipeline scripts do all the work
and write their output to data/processed and reports. If a number shows
up on a page that isnt in one of these files, something has gone wrong.
"""

import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


# ---------------------------------------------------------- customer level

def load_risk_table():
    """scored customers - written by classifier.py"""
    return pd.read_csv(PROCESSED_DIR / "customer_risk_table.csv")


def load_merged_dataset():
    """the cleaned order-item table, for the monthly trend chart"""
    df = pd.read_csv(PROCESSED_DIR / "merged_dataset.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df


def load_shap_importance():
    return pd.read_csv(REPORTS_DIR / "shap_feature_importance.csv")


def load_model_comparison():
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


def load_model():
    # whichever classifier.py picked. xgboost won last run, so try that
    # first and fall back rather than hardcoding a winner
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



# ------------------------------------------- seller level (primary model)

def load_seller_risk_table():
    """scored seller-months from the final holdout - seller_model.py"""
    df = pd.read_csv(PROCESSED_DIR / "seller_risk_table.csv")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def load_seller_panel():
    df = pd.read_csv(PROCESSED_DIR / "seller_panel.csv")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def load_seller_cv_results():
    """one row per model, with mean and std AUC across the rolling folds"""
    return pd.read_csv(REPORTS_DIR / "seller_cv_results.csv")


def load_seller_model_meta():
    """which model won, the tuned threshold, worklist size, base rate"""
    return joblib.load(MODELS_DIR / "seller_model_meta.pkl")


def load_seller_shap_importance():
    return pd.read_csv(REPORTS_DIR / "seller_shap_importance.csv")


def load_seller_shap_groups():
    return pd.read_csv(REPORTS_DIR / "seller_shap_groups.csv")


def load_ablation_results():
    """feature-group ablation - this is the objective 3 evidence"""
    return pd.read_csv(REPORTS_DIR / "ablation_results.csv")


def load_threshold_curve():
    return pd.read_csv(REPORTS_DIR / "threshold_curve.csv")


def load_capacity_curve():
    """precision@N for a few worklist sizes"""
    return pd.read_csv(REPORTS_DIR / "capacity_curve.csv")


# ------------------------------------------------------- sentiment and nlp

def load_sentiment_validation():
    """how TextBlob actually did on the portuguese text -
    written by src/sentiment_validation.py"""
    return pd.read_csv(REPORTS_DIR / "sentiment_validation.csv")


def load_nlp_results():
    """how the induced lexicon did - src/nlp_lexicon.py"""
    return pd.read_csv(REPORTS_DIR / "nlp_results.csv")


def load_nlp_lexicon():
    """the portuguese sentiment words the lexicon learned, and their weights"""
    return pd.read_csv(REPORTS_DIR / "nlp_lexicon.csv")


def load_text_sentiment():
    """per-order sentiment scored from the comment text. not currently a
    model feature - see the note in nlp_lexicon.py"""
    return pd.read_csv(PROCESSED_DIR / "text_sentiment.csv")

def load_review_examples():
    """individual reviews with both sentiment scores attached, for the raw
    review card - src/review_examples.py"""
    return pd.read_csv(REPORTS_DIR / "review_examples.csv")
