# Sales Decline Risk Prediction in Retail

RFM Analysis, Machine Learning and Sentiment Analysis
Chen Xiang Bin — TP068359 — APU 

Predicts which Olist sellers will see their revenue fall by more than
20% the following month, and shows the result through a dashboard that
ranks sellers into a worklist and recommends what to do about each one.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
```

Download the Olist Brazilian E-Commerce dataset from Kaggle
(`olistbr/brazilian-ecommerce`) and put all the CSVs in `data/raw/`.
They are not in this repo — about 125 MB, and `.gitignore` excludes them.

Then run the pipeline **in this order**. Nothing is cached in git, so a
fresh clone needs all eight scripts before the dashboard has anything to
show.

```bash
# customer level - segmentation and dissatisfaction
python src/data_pipeline.py       # merge and clean the raw csvs
python src/rfm_features.py        # recency / frequency / monetary + delivery
python src/sentiment.py           # sentiment from review scores
python src/extra_features.py      # product category + payment behaviour
python src/clustering.py          # MiniBatchKMeans customer segments
python src/classifier.py          # dissatisfaction model + SHAP

# seller level - THE PRIMARY MODEL
python src/seller_features.py     # seller-month panel, 49 features
python src/seller_model.py        # rolling-origin CV, ablation, SHAP

python dashboard/app.py           # http://127.0.0.1:8050/
```

`src/check_project.py` verifies everything is present and consistent —
worth running before a demo.

---

## What the model does

**Target.** For each seller-month: does next month's revenue fall more
than 20%? Roughly 37% of seller-months do, so it is a genuinely balanced
target.

Customer-level decline could not be modelled directly. About 99.4% of
Olist customers never place a second order regardless of how satisfied
they were, so "did they come back" measures the marketplace's repeat
rate rather than any individual risk. Sellers have real monthly revenue
that rises and falls, which is what "sales decline" actually means here.

**Features.** 49, in four blocks — Trading, RFM, Sentiment, Delivery.
Every block gets the same treatment: the level, the month-on-month
change, a 3-month rolling average, and a 3-month trend. An earlier
version gave momentum features to Trading only and then concluded
Trading dominated, which was not a fair comparison.

**Validation.** Rolling-origin cross-validation over 11 folds — train on
months 1..k, test on k+1, slide forward. A random split would let the
model see future months while predicting past ones.

## Results

| Model                          | Mean AUC  |  Std  |
|--------------------------------|---------- |-------|
| Random Forest                  | **0.707** | 0.028 |
| XGBoost                        | 0.695     | 0.027 |
| Logistic Regression (baseline) | 0.644     | 0.024 |

**Ablation** — same model and folds, retrained on each feature block:

| Features         | AUC   |
|------------------|-------|
| All              | 0.707 |
| Trading only     | 0.691 |
| RFM + Sentiment  | 0.625 |
| RFM only         | 0.611 |
| Sentiment only   | 0.526 |
| Delivery only    | 0.516 |

RFM and sentiment together beat either alone, so they carry
complementary signal — which is what objective 3 asked. But they add
little once revenue momentum is known, and that has to be reported too.

**In use.** Ranking beats a fixed threshold. Of the 50 highest-risk
sellers each month, 60.4% actually declined, against a 38.3% base rate —
1.57x. Tuning a threshold for F1 instead flagged 75% of sellers and 95%
of revenue, which is not something a team can act on.

## Known limitations

- **Trading momentum dominates** (67% of SHAP importance). Much of that
  is mean reversion — a revenue spike tends to fall back — rather than
  genuine deterioration. The ablation quantifies it rather than hiding it.
- **Frequency is nearly flat.** ~90% of Olist customers order exactly
  once, so the F in RFM separates almost nobody.
- **TextBlob does not work on this data.** The reviews are Portuguese and
  TextBlob's lexicon is English-only; it returned a polarity of 0 for
  most of them. Sentiment therefore comes from the numeric review score,
  which covers 99.2% of customers.
- **Recall.** At the operating point, a short worklist catches a
  fraction of all declining sellers. Precision is traded for a list
  short enough to act on.

## Layout

```
src/         pipeline scripts, in the run order above
dashboard/   dash app - app.py, theme.py, data_loader.py, pages/
data/raw/    olist csvs (not in git)
data/processed/  generated (not in git)
models/      trained models (not in git)
reports/     metrics and SHAP output
```

## Dashboard

**Sellers** — Seller risk (worklist and model performance) · Interventions
(rule-based actions per seller) · Risk analysis (SHAP, per-seller
comparison) · Simulator (what-if)

**Customers** — Overview (dissatisfaction model) · Segments (KMeans
clusters) · Customer list (filterable)