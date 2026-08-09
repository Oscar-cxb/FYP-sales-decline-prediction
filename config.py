"""
config.py
Central paths and constants used across all pipeline scripts.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_STATE = 42
INACTIVITY_THRESHOLD_DAYS = 180  # placeholder - tune during rfm_features.py
