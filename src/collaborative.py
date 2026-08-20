"""Collaborative filtering via ALS (implicit feedback) over the LastFM-1K play data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class ALSRecommender:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        random_state: int = 42,
    ):
        # Fixed seed: ALS factor init is otherwise random, so reported precision/recall
        # would shift slightly on every re-run — not reproducible for a case study.
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.user_to_idx: dict = {}
        self.idx_to_user: dict = {}
        self.item_to_idx: dict = {}
        self.idx_to_item: dict = {}
        self.user_items: csr_matrix | None = None

    def fit(self, events: pd.DataFrame, user_col: str, item_col: str):
        """events: one row per (user, item) play; play counts are aggregated here."""
        counts = events.groupby([user_col, item_col]).size().reset_index(name="plays")

        users = counts[user_col].unique()
        items = counts[item_col].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        rows = counts[user_col].map(self.user_to_idx)
        cols = counts[item_col].map(self.item_to_idx)
        self.user_items = csr_matrix(
            (counts["plays"].astype(np.float32), (rows, cols)),
            shape=(len(users), len(items)),
        )
        self.model.fit(self.user_items)
        return self

    def recommend(self, user_id, k: int = 10) -> list:
        """Ordered list of item ids recommended for the given user."""
        uidx = self.user_to_idx.get(user_id)
        if uidx is None:
            return []
        ids, _scores = self.model.recommend(
            uidx, self.user_items[uidx], N=k, filter_already_liked_items=True
        )
        return [self.idx_to_item[i] for i in ids]
