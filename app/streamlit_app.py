"""Streamlit demo for the hybrid music recommender.

Imports src/ directly rather than calling api/ over HTTP — see README's Architecture
section for why (Streamlit Community Cloud reliably hosts one process; chaining to a
separately-hosted API adds a second point of failure for no portfolio benefit, and both
artifacts are independently real).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import ensure_artifacts
from src.content_based import ContentBasedRecommender
from src.data_loading import load_spotify_tracks
from src.hybrid import HybridRecommender

st.set_page_config(page_title="Music Taste Recommender", page_icon="🎧", layout="centered")


@st.cache_resource
def load_state():
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
    return als, cbr, tracks, track_lookup, hybrid, last_track_per_user


def split_join_key(join_key: str) -> tuple[str, str]:
    artist, track = join_key.split("|||", 1)
    return track.title(), artist.title()


als, cbr, tracks, track_lookup, hybrid, last_track_per_user = load_state()

st.title("🎧 Music Taste Recommender")
st.caption(
    "Hybrid content-based + collaborative filtering, trained on 13.4M real LastFM "
    "plays. Full technical writeup in `reports/case_study.md`."
)

tab_user, tab_track = st.tabs(["Recommend for a listener", "Find similar tracks"])

with tab_user:
    st.write("Pick a real listener from the training data to see their recommendations.")
    known_users = sorted(als.user_to_idx.keys())
    user_id = st.selectbox("Listener", known_users, key="user_select")
    k = st.slider("Number of recommendations", 5, 20, 10, key="user_k")

    if st.button("Recommend", key="user_button"):
        seed_track_id = last_track_per_user.get(user_id)
        join_keys = hybrid.recommend_for_user(user_id, seed_track_id, k=k)
        if not join_keys:
            st.warning("No recommendations available for this listener.")
        else:
            rows = [dict(zip(("Track", "Artist"), split_join_key(jk))) for jk in join_keys]
            st.table(pd.DataFrame(rows))

with tab_track:
    st.write("Pick a seed track to find audio-feature-similar tracks.")
    sample_tracks = tracks.sample(min(300, len(tracks)), random_state=42)
    options = {
        f"{row.track_name} — {row.artists}": row.track_id
        for row in sample_tracks.itertuples()
    }
    label = st.selectbox("Seed track", sorted(options.keys()), key="track_select")
    k2 = st.slider("Number of similar tracks", 5, 20, 10, key="track_k")

    if st.button("Find similar", key="track_button"):
        track_id = options[label]
        results = cbr.similar_to(track_id, k=k2)
        if not results:
            st.warning("No similar tracks found.")
        else:
            rows = []
            for tid, sim in results:
                row = track_lookup.loc[tid]
                rows.append({
                    "Track": row.track_name,
                    "Artist": row.artists,
                    "Similarity": round(sim, 3),
                })
            st.table(pd.DataFrame(rows))

st.divider()
st.caption(
    "Source: [github.com/crisomara/music-taste-recommender]"
    "(https://github.com/crisomara/music-taste-recommender)"
)
