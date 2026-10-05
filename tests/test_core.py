"""Unit tests for the pure-Python pieces of src/ (no network, no trained artifacts)."""
import numpy as np
import pandas as pd
import pytest

from src.content_based import ContentBasedRecommender
from src.data_loading import AUDIO_FEATURES, build_joined_interactions
from src.evaluation import (
    catalog_coverage, evaluate_model, intra_list_diversity, precision_at_k, recall_at_k,
)
from src.normalize import join_key, normalize_name, primary_artist
from src.splitting import held_out_as_dict, temporal_split


def make_tracks(n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(AUDIO_FEATURES))), columns=AUDIO_FEATURES)
    df["track_id"] = [f"t{i}" for i in range(n)]
    df["track_name"] = [f"Song {i}" for i in range(n)]
    df["artists"] = [f"Artist {i}" for i in range(n)]
    df["join_key"] = [join_key(a, t) for a, t in zip(df["artists"], df["track_name"])]
    df["track_genre"] = "pop"
    df["popularity"] = range(n)
    return df


# --- normalize -------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Beyoncé", "beyonce"),
        ("Song Title (Live at Wembley)", "song title"),
        ("Track feat. Somebody Else", "track"),
        ("Hello, World!", "hello world"),
        ("  Many   spaces  ", "many spaces"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_primary_artist_and_join_key():
    assert primary_artist("Burna Boy;Ed Sheeran") == "Burna Boy"
    assert primary_artist(float("nan")) == ""
    assert join_key("Burna Boy", "Last Last (Remix)") == "burna boy|||last last"


# --- splitting -------------------------------------------------------------------

def test_temporal_split_holds_out_latest_and_drops_sparse_users():
    events = pd.DataFrame({
        "user": ["a"] * 10 + ["b"] * 2,
        "item": [f"i{i}" for i in range(10)] + ["x", "y"],
        "ts": list(range(10)) + [0, 1],
        "extra": 0,
    })
    train, test = temporal_split(events, "user", "item", "ts", test_frac=0.2, min_events_per_user=5)
    assert set(train["user"]) == {"a"} and set(test["user"]) == {"a"}
    assert list(test["item"]) == ["i8", "i9"]
    assert train["ts"].max() < test["ts"].min()
    assert "extra" not in train.columns
    assert held_out_as_dict(test, "user", "item") == {"a": {"i8", "i9"}}


# --- evaluation ------------------------------------------------------------------

def test_precision_recall_coverage():
    recs, relevant = ["a", "b", "c", "d"], {"b", "d", "z"}
    assert precision_at_k(recs, relevant, 2) == 0.5
    assert precision_at_k(recs, relevant, 0) == 0.0
    assert recall_at_k(recs, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(recs, set(), 4) == 0.0
    assert catalog_coverage([["a", "b"], ["b", "c"]], 6) == 0.5


def test_intra_list_diversity():
    lookup = {"x": np.array([1.0, 0.0]), "y": np.array([1.0, 0.0]), "z": np.array([0.0, 1.0])}
    assert intra_list_diversity(["x", "y"], lookup) == pytest.approx(0.0, abs=1e-6)
    assert intra_list_diversity(["x", "z"], lookup) == pytest.approx(1.0)
    assert intra_list_diversity(["x"], lookup) == 0.0


def test_evaluate_model_skips_users_without_held_out_items():
    result = evaluate_model(lambda u: ["a", "b"], ["u1", "u2"], {"u1": {"a"}}, k=2)
    assert result["n_users_evaluated"] == 1
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 1.0


# --- content-based ---------------------------------------------------------------

def test_content_based_similar_to_excludes_seed_and_is_sorted():
    tracks = make_tracks()
    cbr = ContentBasedRecommender(tracks)
    results = cbr.similar_to("t0", k=3)
    assert len(results) == 3
    assert "t0" not in [tid for tid, _ in results]
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)
    assert cbr.similar_to("unknown") == []
    assert list(cbr.explain("t0", results[0][0]).index) != []
    assert set(cbr.feature_lookup()) == set(tracks["track_id"])


def test_content_based_identical_track_is_most_similar():
    tracks = make_tracks()
    tracks.loc[5, AUDIO_FEATURES] = tracks.loc[0, AUDIO_FEATURES].values
    cbr = ContentBasedRecommender(tracks)
    top_id, top_sim = cbr.similar_to("t0", k=1)[0]
    assert top_id == "t5"
    assert top_sim == pytest.approx(1.0)


# --- join ------------------------------------------------------------------------

def test_build_joined_interactions_reports_coverage():
    tracks = make_tracks(3)
    interactions = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "join_key": [tracks.loc[0, "join_key"], "nobody|||nothing", tracks.loc[1, "join_key"], tracks.loc[0, "join_key"]],
    })
    joined, stats = build_joined_interactions(interactions, tracks)
    assert stats["n_events"] == 4
    assert stats["n_matched_events"] == 3
    assert stats["event_coverage_pct"] == 75.0
    assert stats["unique_track_coverage_pct"] == pytest.approx(66.67)
    assert len(joined) == 3
    assert {"track_id", "danceability", "track_genre"} <= set(joined.columns)
