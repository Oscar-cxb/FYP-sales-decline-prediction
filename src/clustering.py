"""
clustering.py

Runs MiniBatchKMeans on the RFM(+delivery delay) features to segment
customers, matching the RFM diagram from my IR (Champions, At Risk,
Lost, etc). This is the "before modelling" step - classifier.py will
use these segments as an extra feature.

run after rfm_features.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import joblib

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

FEATURES = ["recency", "frequency", "monetary", "avg_delivery_delay"]

# testing k from 3 to 6 - my proposal said "k>=3" so starting there,
# 6 felt like a reasonable upper bound before segments get too granular to be useful
K_RANGE = range(3, 7)


def scale_features(df):
    # monetary is heavily right skewed (median ~90 but max ~13k) so log-transforming
    # it first, otherwise it completely dominates the euclidean distance and
    # kmeans just ends up clustering on monetary alone basically
    df = df.copy()
    df["monetary_log"] = np.log1p(df["monetary"])

    cols_to_scale = ["recency", "frequency", "monetary_log", "avg_delivery_delay"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols_to_scale])

    return scaled, scaler


def find_best_k(scaled_features):
    # using silhouette score to pick k instead of just eyeballing an elbow plot,
    # easier to justify a specific number this way for the report
    scores = {}
    for k in K_RANGE:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        scores[k] = score
        print(f"k={k}: silhouette score = {score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"best k = {best_k} (silhouette = {scores[best_k]:.4f})")
    return best_k


def fit_kmeans(scaled_features, k):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_features)
    return km, labels


def label_segments(df, scaled_features):
    # was ranking clusters using raw monetary - recency before, but that let
    # recency's bigger numeric range quietly dominate the ranking. e.g. "Loyal
    # Customers" is the only segment with real repeat purchases (freq 2.11 vs
    # everyone else stuck at 1.0) but still got ranked below "Champions" just
    # because of a small recency difference. thats backwards - frequency
    # actually meaning something should count for a lot here.
    # fix: rank using the already-standardized features (same units, mean 0 std 1)
    # instead of raw values, so no single column's scale can dominate

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=["recency_z", "frequency_z", "monetary_z", "delay_z"],
    )
    scaled_df["cluster"] = df["cluster"].values

    cluster_avg = scaled_df.groupby("cluster").mean()

    # higher frequency/monetary = better, lower recency/delay = better
    # (recency_z and delay_z get flipped with a minus sign)
    cluster_avg["rank_score"] = (
        cluster_avg["frequency_z"]
        + cluster_avg["monetary_z"]
        - cluster_avg["recency_z"]
        - cluster_avg["delay_z"]
    )
    cluster_avg = cluster_avg.sort_values("rank_score", ascending=False)

    print("cluster ranking (best to worst):")
    print(cluster_avg[["recency_z", "frequency_z", "monetary_z", "delay_z", "rank_score"]])

    # renamed a couple of these after actually looking at what defines them:
    # - "Loyal Customers" -> "Promising": this segment has freq=1.0 same as
    #   most others, its recent+high spend that puts it here, not repeat
    #   purchases. calling it "loyal" overstates it - only the Champions
    #   cluster has real repeat-purchase behaviour (freq=2.11)
    # - "Lost" -> "Delayed Delivery": this cluster's defining trait isnt low
    #   recency/monetary really, its the only segment with consistently LATE
    #   deliveries (positive avg_delivery_delay while every other segment is
    #   early). (note: the at_risk % comparison that originally motivated this
    #   rename was from an earlier, since-replaced label definition - keeping
    #   the rename anyway since the delivery-delay observation itself still holds)
    segment_names_pool = ["Champions", "Promising", "At Risk", "Needs Attention",
                          "Delayed Delivery", "Hibernating"]
    ranked_clusters = cluster_avg.index.tolist()

    name_map = {}
    for i, cluster_id in enumerate(ranked_clusters):
        if i < len(segment_names_pool):
            name_map[cluster_id] = segment_names_pool[i]
        else:
            name_map[cluster_id] = f"Segment {cluster_id}"

    df["segment_name"] = df["cluster"].map(name_map)
    return df


def main():
    rfm = pd.read_csv(PROCESSED_DIR / "rfm_features.csv")

    scaled_features, scaler = scale_features(rfm)
    best_k = find_best_k(scaled_features)

    km, labels = fit_kmeans(scaled_features, best_k)
    rfm["cluster"] = labels

    rfm = label_segments(rfm, scaled_features)

    print(rfm.groupby("segment_name")[FEATURES].mean())
    print(rfm["segment_name"].value_counts())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(km, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "rfm_scaler.pkl")

    out_path = PROCESSED_DIR / "customer_segments.csv"
    rfm.to_csv(out_path, index=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()