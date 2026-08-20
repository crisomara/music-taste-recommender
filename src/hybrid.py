"""Hybrid recommender: collaborative-first cascade with content-based cold-start fallback.

Rationale (see reports/case_study.md): collaborative filtering is expected to win for
users/items with enough interaction history; content-based similarity is the fallback
when ALS has too little signal (a new/rarely-played track, or a user below the min
interaction threshold) — this is the standard cold-start cascade pattern, not a novel
technique, and is evaluated against both single-method baselines to prove it's worth
the added complexity.
"""
from __future__ import annotations

from .collaborative import ALSRecommender
from .content_based import ContentBasedRecommender


class HybridRecommender:
    """Bridges two identifier spaces: the collaborative layer is trained on `join_key`
    (the only id shared by both datasets, covering the full 740k-track catalog), while
    the content-based layer operates on Spotify `track_id` (only ~1.3% of that catalog
    has audio features). `track_id_to_join_key` translates content-based output into
    the collaborative layer's space so the two can be combined and compared fairly.
    """

    def __init__(
        self,
        collaborative: ALSRecommender,
        content: ContentBasedRecommender,
        track_id_to_join_key: dict,
        min_interactions_for_cf: int = 5,
    ):
        self.collaborative = collaborative
        self.content = content
        self.track_id_to_join_key = track_id_to_join_key
        self.min_interactions_for_cf = min_interactions_for_cf

    def _content_as_join_keys(self, seed_track_id: str, k: int) -> list[str]:
        out = []
        for tid, _ in self.content.similar_to(seed_track_id, k=k * 3):
            jk = self.track_id_to_join_key.get(tid)
            if jk is not None:
                out.append(jk)
            if len(out) >= k:
                break
        return out

    def recommend_for_user(
        self, user_id, seed_track_id: str | None, k: int = 10
    ) -> list[str]:
        """seed_track_id is a Spotify track_id (e.g. user's most recent matched play),
        used for the content-based fallback when collaborative filtering has
        insufficient signal. Returns join_keys throughout, matching the collaborative
        layer's catalog and the evaluation harness's held-out sets."""
        uidx = self.collaborative.user_to_idx.get(user_id)
        n_interactions = (
            self.collaborative.user_items[uidx].nnz if uidx is not None else 0
        )

        if n_interactions >= self.min_interactions_for_cf:
            recs = self.collaborative.recommend(user_id, k=k)
            if len(recs) >= k or seed_track_id is None:
                return recs
            fallback = [
                jk for jk in self._content_as_join_keys(seed_track_id, k * 2)
                if jk not in recs
            ]
            return (recs + fallback)[:k]

        if seed_track_id is None:
            return []
        return self._content_as_join_keys(seed_track_id, k)
