"""FastAPI serving layer for the hybrid music recommender.

Loads the artifacts already trained in notebooks/ (see README) and reuses the same
src/ classes the notebooks use — no modeling logic is duplicated here.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse, Recommendation, RecommendResponse, SampleTrack,
    SampleUser, SimilarResponse, SimilarTrack,
)
from src.artifacts import ensure_artifacts
from src.content_based import ContentBasedRecommender
from src.data_loading import load_spotify_tracks
from src.hybrid import HybridRecommender

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Music Taste Recommender API",
    description="Hybrid (content-based + collaborative) music recommendation service.",
    version="1.0.0",
)

_state: dict = {}


def _split_join_key(join_key: str) -> tuple[str, str]:
    artist, track = join_key.split("|||", 1)
    return track.title(), artist.title()


@app.on_event("startup")
def load_artifacts() -> None:
    interactions_dir = ensure_artifacts()
    with open(interactions_dir / "als_model.pkl", "rb") as f:
        als = pickle.load(f)

    tracks = load_spotify_tracks()
    cbr = ContentBasedRecommender(tracks)
    track_id_to_join_key = dict(zip(tracks["track_id"], tracks["join_key"]))
    track_lookup = tracks.set_index("track_id")[["track_name", "artists"]]

    joined = pd.read_parquet(interactions_dir / "joined_train.parquet")
    last_track_per_user = (
        joined.sort_values("timestamp").groupby("user_id")["track_id"].last().to_dict()
    )

    hybrid = HybridRecommender(als, cbr, track_id_to_join_key, min_interactions_for_cf=5)

    _state.update(
        als=als,
        cbr=cbr,
        tracks=tracks,
        track_lookup=track_lookup,
        hybrid=hybrid,
        last_track_per_user=last_track_per_user,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not _state:
        raise HTTPException(503, "Artifacts not loaded yet")
    return HealthResponse(
        status="ok",
        known_users=len(_state["als"].user_to_idx),
        known_tracks=len(_state["tracks"]),
    )


@app.get("/users/sample", response_model=list[SampleUser])
def sample_users(n: int = 20) -> list[SampleUser]:
    """Returns real user IDs the ALS model was trained on — use one of these to call
    /recommend/{user_id}, since the model has no signal for unknown users."""
    users = list(_state["als"].user_to_idx.keys())
    picked = random.sample(users, min(n, len(users)))
    return [SampleUser(user_id=u) for u in picked]


@app.get("/tracks/sample", response_model=list[SampleTrack])
def sample_tracks(n: int = 20) -> list[SampleTrack]:
    """Returns real Spotify track IDs with names — use one to call /similar/{track_id}."""
    tracks = _state["tracks"]
    sample = tracks.sample(min(n, len(tracks)))
    return [
        SampleTrack(track_id=row.track_id, track_name=row.track_name, artist=row.artists)
        for row in sample.itertuples()
    ]


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: str, k: int = 10) -> RecommendResponse:
    if user_id not in _state["als"].user_to_idx:
        raise HTTPException(404, f"Unknown user_id '{user_id}'. See /users/sample.")
    seed_track_id = _state["last_track_per_user"].get(user_id)
    join_keys = _state["hybrid"].recommend_for_user(user_id, seed_track_id, k=k)
    recs = []
    for jk in join_keys:
        track_name, artist = _split_join_key(jk)
        recs.append(Recommendation(join_key=jk, track_name=track_name, artist=artist))
    return RecommendResponse(user_id=user_id, seed_track_id=seed_track_id, recommendations=recs)


@app.get("/similar/{track_id}", response_model=SimilarResponse)
def similar(track_id: str, k: int = 10) -> SimilarResponse:
    track_lookup = _state["track_lookup"]
    if track_id not in track_lookup.index:
        raise HTTPException(404, f"Unknown track_id '{track_id}'. See /tracks/sample.")
    seed_row = track_lookup.loc[track_id]
    results = _state["cbr"].similar_to(track_id, k=k)
    similar_tracks = []
    for tid, sim in results:
        row = track_lookup.loc[tid]
        similar_tracks.append(
            SimilarTrack(track_id=tid, track_name=row.track_name, artist=row.artists, similarity=round(sim, 4))
        )
    return SimilarResponse(
        seed_track_id=track_id,
        seed_track_name=seed_row.track_name,
        seed_artist=seed_row.artists,
        similar=similar_tracks,
    )
