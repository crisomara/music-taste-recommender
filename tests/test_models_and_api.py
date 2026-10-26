"""ALS, hybrid cascade and FastAPI endpoints on a tiny synthetic dataset.

The real ALS model and joined_train.parquet live on Hugging Face Hub; these tests never
download them. The API is exercised by filling its state with a small model instead of
running the startup hook.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from src.collaborative import ALSRecommender
from src.content_based import ContentBasedRecommender
from src.data_loading import AUDIO_FEATURES
from src.hybrid import HybridRecommender
from src.normalize import join_key


@pytest.fixture(scope="module")
def tracks() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 12
    df = pd.DataFrame(rng.random((n, len(AUDIO_FEATURES))), columns=AUDIO_FEATURES)
    df["track_id"] = [f"t{i}" for i in range(n)]
    df["track_name"] = [f"song {i}" for i in range(n)]
    df["artists"] = [f"artist {i}" for i in range(n)]
    df["join_key"] = [join_key(a, t) for a, t in zip(df["artists"], df["track_name"])]
    return df


@pytest.fixture(scope="module")
def als(tracks) -> ALSRecommender:
    rng = np.random.default_rng(2)
    keys = list(tracks["join_key"])
    rows = []
    for u in range(8):
        for item in rng.choice(keys, size=6, replace=False):
            rows.extend([(f"user{u}", item)] * int(rng.integers(1, 4)))
    # A user with too few distinct plays for the collaborative layer.
    rows.append(("sparse_user", keys[0]))
    events = pd.DataFrame(rows, columns=["user_id", "join_key"])
    return ALSRecommender(factors=4, iterations=3).fit(events, "user_id", "join_key")


@pytest.fixture(scope="module")
def hybrid(als, tracks) -> HybridRecommender:
    cbr = ContentBasedRecommender(tracks)
    mapping = dict(zip(tracks["track_id"], tracks["join_key"]))
    return HybridRecommender(als, cbr, mapping, min_interactions_for_cf=5)


def test_als_recommends_unseen_known_items(als):
    recs = als.recommend("user0", k=3)
    assert 0 < len(recs) <= 3
    seen = set(als.user_items[als.user_to_idx["user0"]].indices)
    assert all(als.item_to_idx[r] not in seen for r in recs)
    assert als.recommend("nobody") == []


def test_hybrid_uses_collaborative_for_active_users(hybrid, als):
    assert hybrid.recommend_for_user("user0", None, k=3) == als.recommend("user0", k=3)


def test_hybrid_falls_back_to_content_for_sparse_users(hybrid, tracks):
    recs = hybrid.recommend_for_user("sparse_user", "t0", k=3)
    assert len(recs) == 3
    assert tracks.loc[0, "join_key"] not in recs
    assert set(recs) <= set(tracks["join_key"])
    assert hybrid.recommend_for_user("sparse_user", None, k=3) == []


@pytest.fixture()
def client(monkeypatch, als, hybrid, tracks):
    state = {
        "als": als,
        "cbr": hybrid.content,
        "tracks": tracks,
        "track_lookup": tracks.set_index("track_id")[["track_name", "artists"]],
        "hybrid": hybrid,
        "last_track_per_user": {"user0": "t0", "sparse_user": "t1"},
    }
    monkeypatch.setattr(api_main, "_state", state)
    # Not used as a context manager, so the startup hook (which downloads artifacts) does not run.
    return TestClient(api_main.app)


def test_health(client, als, tracks):
    body = client.get("/health").json()
    assert body == {"status": "ok", "known_users": len(als.user_to_idx), "known_tracks": len(tracks)}


def test_health_before_startup(monkeypatch):
    monkeypatch.setattr(api_main, "_state", {})
    assert TestClient(api_main.app).get("/health").status_code == 503


def test_samples(client, als):
    users = client.get("/users/sample", params={"n": 3}).json()
    assert len(users) == 3 and all(u["user_id"] in als.user_to_idx for u in users)
    assert len(client.get("/tracks/sample", params={"n": 4}).json()) == 4


def test_recommend(client):
    resp = client.get("/recommend/sparse_user", params={"k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["seed_track_id"] == "t1"
    assert len(body["recommendations"]) == 3
    assert all(r["artist"].startswith("Artist") for r in body["recommendations"])
    assert client.get("/recommend/unknown").status_code == 404


def test_similar(client):
    resp = client.get("/similar/t2", params={"k": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["seed_track_name"] == "song 2"
    assert len(body["similar"]) == 4
    assert all(-1.0 <= s["similarity"] <= 1.0 for s in body["similar"])
    assert client.get("/similar/nope").status_code == 404
