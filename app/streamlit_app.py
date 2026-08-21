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

# key/mode/time_signature are structural/categorical (a shared value doesn't mean two
# songs "feel" similar the way a shared tempo or energy does), so they're excluded from
# the human-facing "why this match" explanation, same reasoning as
# scripts/build_large_catalog.py's DESCRIPTIVE_FEATURES.
EXPLAINABLE_FEATURES = [
    "tempo", "energy", "loudness", "danceability", "acousticness",
    "valence", "instrumentalness", "speechiness", "liveness",
]
FEATURE_LABELS = {
    "tempo": "Tempo", "energy": "Energy", "loudness": "Loudness",
    "danceability": "Danceability", "acousticness": "Acoustic quality",
    "valence": "Mood", "instrumentalness": "Instrumental balance",
    "speechiness": "Vocal presence", "liveness": "Live-performance feel",
}

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
  .track-card .why-match {
    color: #3E7C74;
    font-size: 0.78rem;
    margin-top: 0.2rem;
    letter-spacing: 0.01em;
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


FEATURE_COL_INDEX = {name: i for i, name in enumerate(FEATURES)}


def explain_match(seed_idx: int, other_idx: int, feature_matrix: np.ndarray, n: int = 2) -> list[str]:
    """Returns up to n human-readable feature names that most drove the similarity
    (smallest standardized difference) — same rationale as
    src/content_based.py's ContentBasedRecommender.explain(), adapted for this app's
    own feature matrix."""
    diffs = {
        feat: abs(feature_matrix[seed_idx, FEATURE_COL_INDEX[feat]] - feature_matrix[other_idx, FEATURE_COL_INDEX[feat]])
        for feat in EXPLAINABLE_FEATURES
    }
    ranked = sorted(diffs.items(), key=lambda kv: kv[1])
    return [FEATURE_LABELS[feat] for feat, _ in ranked[:n]]


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

artists = ["All artists"] + sorted(mood_pool["primary_artist"].unique(), key=str.casefold)
artist = st.selectbox(
    "Artist", artists,
    help="Start typing to search — stays selected while you pick a song below.",
)

pool = mood_pool if artist == "All artists" else mood_pool[mood_pool["primary_artist"] == artist]

if pool.empty:
    st.warning("No tracks found for that combination.")
else:
    if "seed_track_id" not in st.session_state:
        st.session_state.seed_track_id = None

    # A clicked suggestion is applied here, *before* the text_input widget below is
    # instantiated — session_state for a widget's key can only be set before that
    # widget is created in a given run, not after.
    pending = st.session_state.pop("_pending_pick", None)
    if pending is not None:
        st.session_state.seed_track_id = pending[0]
        st.session_state.song_query = pending[1]

    query = st.text_input(
        "Song",
        key="song_query",
        placeholder="Start typing a song or artist name…",
    )

    def pick(track_id: str, display: str) -> None:
        st.session_state._pending_pick = (track_id, display)

    if query.strip():
        q = query.strip().lower()
        name_match = pool["track_name"].str.lower().str.contains(q, na=False, regex=False)
        artist_match = pool["artists"].str.lower().str.contains(q, na=False, regex=False)
        suggestions = pool[name_match | artist_match].head(8)

        if not suggestions.empty:
            st.caption("Suggestions — click one, or keep typing and press the button below")
            for row in suggestions.itertuples():
                display = f"{row.track_name} — {row.artists}"
                if st.button(display, key=f"suggest_{row.track_id}", use_container_width=True):
                    pick(row.track_id, display)
                    st.rerun()

    if st.button("🔍 Find similar tracks", type="primary"):
        resolved_id = st.session_state.seed_track_id
        # If the text was edited after a suggestion was picked (or never picked at
        # all), resolve the current free-typed text to the closest match instead.
        current_label = None
        if resolved_id is not None:
            row = tracks.set_index("track_id").loc[resolved_id]
            current_label = f"{row.track_name} — {row.artists}"
        if query.strip() and query.strip() != current_label:
            q = query.strip().lower()
            exact = pool[pool["track_name"].str.lower() == q]
            starts = pool[pool["track_name"].str.lower().str.startswith(q, na=False)]
            contains = pool[
                pool["track_name"].str.lower().str.contains(q, na=False, regex=False)
                | pool["artists"].str.lower().str.contains(q, na=False, regex=False)
            ]
            best = next((m for m in (exact, starts, contains) if not m.empty), None)
            resolved_id = best.iloc[0]["track_id"] if best is not None else None

        if resolved_id is None:
            st.warning("Couldn't find a matching song — try a suggestion or a different search.")
        else:
            st.session_state.seed_track_id = resolved_id

    seed_track_id = st.session_state.seed_track_id
    if seed_track_id is not None:
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
            seed_idx = idx_lookup[seed_track_id]
            cards = []
            for i, (tid, sim) in enumerate(results, start=1):
                row = track_lookup.loc[tid]
                why = explain_match(seed_idx, idx_lookup[tid], feature_matrix)
                cards.append(
                    f'<div class="track-card">'
                    f'<div class="rank">{i:02d}</div>'
                    f'<div class="info">'
                    f'<div class="track-name">{row.track_name}</div>'
                    f'<div class="artist">{row.artists}</div>'
                    f'<div class="why-match">Matched on {" &amp; ".join(why)}</div>'
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
