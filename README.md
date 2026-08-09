# Sales Decline Risk Prediction in Retail
RFM Analysis, Machine Learning and Sentiment Analysis — FYP (TP068359)

## Pipeline order
1. `src/data_pipeline.py`   — merge & clean Olist CSVs
2. `src/rfm_features.py`    — RFM feature engineering + at-risk label
3. `src/sentiment.py`       — sentiment scoring from reviews
4. `src/clustering.py`      — MiniBatchKMeans customer segmentation
5. `src/classifier.py`      — Random Forest + XGBoost training, SMOTE, SHAP
6. `src/evaluation.py`      — metrics, model comparison table
7. `dashboard/app.py`       — Plotly Dash dashboard (run last)

## Setup
```
pip install -r requirements.txt
```

## Data
Place the 9 Olist CSVs (from Kaggle: olistbr/brazilian-ecommerce) in `data/raw/`.

## Run dashboard
```
python dashboard/app.py
```
