"""Builds the large-catalog artifact for the Streamlit demo only.

Does NOT touch src/data_loading.py's load_spotify_tracks() or the 114k-track dataset
the notebooks/case study/API are built on — this is a separate, demo-only catalog.

Source: rodolfofigueroa/spotify-12m-songs on Kaggle (1.2M tracks, real audio features,
no genre column). Clusters tracks by their own standardized audio features into
mood/style groups as a substitute for genre, auto-labeled from each cluster's
distinguishing traits.

Usage:
    python scripts/build_large_catalog.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import join_key, normalize_name

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature",
]
N_CLUSTERS = 9

# Which side of the global mean each feature needs to land on, and the word used to
# describe that side, for the small set of features expressive enough to name a
# cluster after (key/mode/time_signature are categorical/structural, not moods).
DESCRIPTIVE_FEATURES = {
    "energy": ("Energetic", "Mellow"),
    "loudness": ("Loud", "Quiet"),
    "danceability": ("Danceable", "Free-Form"),
    "acousticness": ("Acoustic", "Electronic"),
    "valence": ("Upbeat", "Somber"),
    "tempo": ("Fast", "Slow"),
    "instrumentalness": ("Instrumental", "Vocal"),
}


def parse_artists(raw: str) -> str:
    """The Kaggle dataset stores `artists` as a Python-list repr string, e.g.
    "['Rage Against The Machine']" — parse it into a plain display string."""
    try:
        artists = ast.literal_eval(raw)
        if isinstance(artists, list) and artists:
            return ", ".join(artists)
    except (ValueError, SyntaxError):
        pass
    return str(raw)


def label_cluster(centroid: pd.Series, global_mean: pd.Series, global_std: pd.Series, n_words: int = 2) -> str:
    """Labels a cluster from the n_words features whose centroid deviates most (in
    standard deviations) from the global mean, among the human-describable feature set."""
    z_scores = {
        feat: (centroid[feat] - global_mean[feat]) / global_std[feat]
        for feat in DESCRIPTIVE_FEATURES
    }
    ranked = sorted(z_scores.items(), key=lambda kv: -abs(kv[1]))
    words = []
    for feat, z in ranked[:n_words]:
        high_word, low_word = DESCRIPTIVE_FEATURES[feat]
        words.append(high_word if z > 0 else low_word)
    return " & ".join(words)


def main() -> None:
    import kagglehub

    print("Downloading source dataset from Kaggle...")
    dataset_path = Path(kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs"))
    csv_path = dataset_path / "tracks_features.csv"

    print("Loading raw CSV...")
    df = pd.read_csv(
        csv_path,
        usecols=["id", "name", "artists"] + FEATURES,
    )
    df = df.rename(columns={"id": "track_id", "name": "track_name"})
    df["artists"] = df["artists"].map(parse_artists)
    df = df.dropna(subset=["track_name", "artists"])

    print(f"Loaded {len(df):,} raw rows. Deduplicating...")
    df["join_key"] = [
        join_key(a, t) for a, t in zip(df["artists"], df["track_name"])
    ]
    df = df.drop_duplicates("join_key").drop(columns=["join_key"]).reset_index(drop=True)
    print(f"{len(df):,} rows after dedup.")

    print("Clustering by audio features...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURES])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(scaled)

    global_mean = df[FEATURES].mean()
    global_std = df[FEATURES].std()
    centroids = {
        cid: df.loc[df["cluster_id"] == cid, FEATURES].mean() for cid in range(N_CLUSTERS)
    }
    cluster_labels = {
        cid: label_cluster(centroid, global_mean, global_std) for cid, centroid in centroids.items()
    }

    # Disambiguate collisions (different clusters sharing a top-2 pair) by adding a
    # 3rd distinguishing word instead of a numeric suffix.
    label_counts: dict[str, int] = {}
    for label in cluster_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    for cid, label in list(cluster_labels.items()):
        if label_counts[label] > 1:
            cluster_labels[cid] = label_cluster(centroids[cid], global_mean, global_std, n_words=3)

    df["mood"] = df["cluster_id"].map(cluster_labels)

    print("Cluster sizes and labels:")
    print(df["mood"].value_counts())

    out_path = PROJECT_ROOT / "data" / "raw" / "interactions" / "large_catalog.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["cluster_id"]).to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
