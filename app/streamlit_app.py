"""Streamlit demo: find tracks that sound like a given song.

Single-purpose by design — pick a mood/style group to narrow the search, then type a
song or artist name to find audio-feature-similar tracks, searched across a 1.2M-track
catalog (see scripts/build_large_catalog.py for how it was built). Genre labels aren't
available at this scale, so tracks are grouped by their own audio-feature clusters
instead — arguably more informative than a genre tag anyway, since it reflects what a
song actually sounds like rather than how it's marketed.

Imports src/ directly rather than calling api/ over HTTP — see README's Architecture
section for why (Streamlit Community Cloud reliably hosts one process; chaining it to
a separately-hosted API adds a second point of failure for no portfolio benefit, and
both artifacts are independently real). Note: this app uses its own larger catalog,
separate from src/data_loading.py's 114k-track dataset that the notebooks, case study,
and FastAPI service are built on — see README's Architecture section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import LARGE_CATALOG_FILE, ensure_artifacts

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature",
]

st.set_page_config(page_title="Music Taste Recommender", page_icon="🎧", layout="centered")

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

  h1, h2, h3, .app-title { font-family: 'Fraunces', Georgia, serif !important; }

  .app-title {
    font-size: 2.4rem;
    font-weight: 600;
    color: #1B1D1B;
    margin-bottom: 0.1rem;
    letter-spacing: -0.01em;
  }
  .app-subtitle {
    color: #5B5F58;
    font-size: 1.02rem;
    margin-bottom: 1.8rem;
    max-width: 60ch;
  }

  .seed-banner {
    background: #E7EAE3;
    border-left: 3px solid #3E7C74;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 1.2rem 0 1.6rem;
    font-size: 0.95rem;
    color: #1B1D1B;
  }
  .seed-banner b { color: #B5652A; }

  .track-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #E7EAE3;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
  }
  .track-card .rank {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.1rem;
    color: #B5652A;
    width: 2rem;
    flex-shrink: 0;
  }
  .track-card .info { flex-grow: 1; min-width: 0; }
  .track-card .track-name {
    font-weight: 600;
    color: #1B1D1B;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .track-card .artist {
    color: #5B5F58;
    font-size: 0.9rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .track-card .similarity {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: #3E7C74;
    font-size: 1rem;
    flex-shrink: 0;
  }

  .footer-note {
    color: #767B72;
    font-size: 0.82rem;
    margin-top: 2.5rem;
  }
  .footer-note a { color: #3E7C74; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_state():
    interactions_dir = ensure_artifacts([LARGE_CATALOG_FILE])
    tracks = pd.read_parquet(interactions_dir / LARGE_CATALOG_FILE)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(tracks[FEATURES])
    idx_lookup = {tid: i for i, tid in enumerate(tracks["track_id"])}
    return tracks, feature_matrix, idx_lookup


def similar_to(track_id: str, tracks: pd.DataFrame, feature_matrix: np.ndarray, idx_lookup: dict, k: int = 10):
    idx = idx_lookup.get(track_id)
    if idx is None:
        return []
    sims = cosine_similarity(feature_matrix[idx : idx + 1], feature_matrix)[0]
    order = np.argsort(-sims)
    results = []
    for i in order:
        if i == idx:
            continue
        results.append((tracks.iloc[i]["track_id"], float(sims[i])))
        if len(results) >= k:
            break
    return results


tracks, feature_matrix, idx_lookup = load_state()

st.markdown('<div class="app-title">🎧 Find similar tracks</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Pick a mood or style, then type a song or artist to find '
    'tracks that sound alike — matched on real audio features (tempo, energy, mood, and '
    f'more) across {len(tracks):,} songs. No genre tags here — groups are built from '
    'what each song actually sounds like, not how it\'s marketed.</div>',
    unsafe_allow_html=True,
)

moods = ["All moods"] + sorted(tracks["mood"].unique())
mood = st.selectbox("Mood / style", moods)

mood_pool = tracks if mood == "All moods" else tracks[tracks["mood"] == mood]

artists = ["All artists"] + sorted(mood_pool["artists"].unique())
artist = st.selectbox(
    "Artist", artists,
    help="Start typing to search — stays selected while you pick a song below.",
)

pool = mood_pool if artist == "All artists" else mood_pool[mood_pool["artists"] == artist]
options = {
    row.track_name: row.track_id for row in pool.sort_values("track_name").itertuples()
} if artist != "All artists" else {
    f"{row.track_name} — {row.artists}": row.track_id
    for row in pool.sort_values("track_name").itertuples()
}

if not options:
    st.warning("No tracks found for that combination.")
else:
    label = st.selectbox(
        "Song", sorted(options.keys()),
        help="Start typing to search.",
    )
    seed_track_id = options[label]
    seed_row = tracks.set_index("track_id").loc[seed_track_id]

    st.markdown(
        f'<div class="seed-banner">Finding songs that sound like '
        f'<b>{seed_row.track_name}</b> by {seed_row.artists}</div>',
        unsafe_allow_html=True,
    )

    results = similar_to(seed_track_id, tracks, feature_matrix, idx_lookup, k=10)
    track_lookup = tracks.set_index("track_id")

    if not results:
        st.info("No similar tracks found for this song.")
    else:
        cards = []
        for i, (tid, sim) in enumerate(results, start=1):
            row = track_lookup.loc[tid]
            cards.append(
                f'<div class="track-card">'
                f'<div class="rank">{i:02d}</div>'
                f'<div class="info">'
                f'<div class="track-name">{row.track_name}</div>'
                f'<div class="artist">{row.artists}</div>'
                f'</div>'
                f'<div class="similarity">{sim:.0%}</div>'
                f'</div>'
            )
        st.markdown("".join(cards), unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Source: '
    '<a href="https://github.com/crisomara/music-taste-recommender">'
    "github.com/crisomara/music-taste-recommender</a></div>",
    unsafe_allow_html=True,
)
