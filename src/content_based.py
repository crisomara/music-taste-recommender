"""Content-based recommender: cosine similarity over standardized audio features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .data_loading import AUDIO_FEATURES


class ContentBasedRecommender:
    def __init__(self, tracks: pd.DataFrame, feature_cols: list[str] = AUDIO_FEATURES):
        self.tracks = tracks.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.tracks[feature_cols])
        self._id_to_idx = {tid: i for i, tid in enumerate(self.tracks["track_id"])}

    def similar_to(self, track_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to k (track_id, similarity) pairs most similar to the given track."""
        idx = self._id_to_idx.get(track_id)
        if idx is None:
            return []
        sims = cosine_similarity(
            self.feature_matrix[idx : idx + 1], self.feature_matrix
        )[0]
        order = np.argsort(-sims)
        results = []
        for i in order:
            if i == idx:
                continue
            results.append((self.tracks.iloc[i]["track_id"], float(sims[i])))
            if len(results) >= k:
                break
        return results

    def explain(self, track_id: str, similar_id: str) -> pd.Series:
        """Per-feature (standardized) contribution to the similarity of two tracks —
        the features with the smallest absolute difference drove the match most."""
        idx_a = self._id_to_idx[track_id]
        idx_b = self._id_to_idx[similar_id]
        diff = np.abs(self.feature_matrix[idx_a] - self.feature_matrix[idx_b])
        return pd.Series(diff, index=self.feature_cols).sort_values()

    def feature_lookup(self) -> dict:
        """track_id -> standardized feature vector, for diversity metrics."""
        return {
            tid: self.feature_matrix[i]
            for i, tid in enumerate(self.tracks["track_id"])
        }
