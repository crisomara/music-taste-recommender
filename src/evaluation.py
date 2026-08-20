"""Standard implicit-feedback recommender evaluation metrics.

All functions take `recommended` (ordered list of item ids, top-k already sliced) and
`relevant` (set of item ids the user actually interacted with in the held-out split).
"""
from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def catalog_coverage(all_recommendations: list[list], catalog_size: int) -> float:
    """Fraction of the catalog that appears in at least one user's recommendation list."""
    recommended_items = set()
    for rec_list in all_recommendations:
        recommended_items.update(rec_list)
    return len(recommended_items) / catalog_size


def intra_list_diversity(recommended: list, feature_lookup: dict) -> float:
    """Mean pairwise cosine distance between recommended items' audio-feature vectors.
    Higher = more diverse recommendation list. feature_lookup maps item_id -> np.array.
    """
    vecs = [feature_lookup[i] for i in recommended if i in feature_lookup]
    if len(vecs) < 2:
        return 0.0
    vecs = np.array(vecs)
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sim_matrix = norm @ norm.T
    n = len(vecs)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(1 - upper_tri.mean())


def evaluate_model(
    recommend_fn, users: list, held_out: dict[str, set], k: int = 10
) -> dict:
    """Run precision@k / recall@k across a list of users.

    recommend_fn(user) -> ordered list of recommended item ids.
    held_out: user_id -> set of item ids relevant in the held-out split.
    """
    precisions, recalls = [], []
    all_recs = []
    for user in users:
        relevant = held_out.get(user, set())
        if not relevant:
            continue
        recs = recommend_fn(user)
        all_recs.append(recs[:k])
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "n_users_evaluated": len(precisions),
        "all_recommendations": all_recs,
    }
