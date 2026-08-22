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

import hashlib
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
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
  html, body, [class*="css"] { font-family: 'Manrope', -apple-system, sans-serif; }

  /* App chrome: pin Streamlit's own frame to the same near-black surface so there's
     no light-gray gutter around the themed content. */
  .stApp, [data-testid="stHeader"] { background: #121212; }
  .block-container { padding-top: 2.5rem; max-width: 780px; }

  h1, h2, h3, .app-title { font-family: 'Poppins', 'Manrope', sans-serif !important; }

  .app-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.15rem;
    letter-spacing: -0.02em;
  }
  .app-subtitle {
    color: #B3B3B3;
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 2rem;
    max-width: 62ch;
  }

  /* Widget labels ("Mood / style", "Artist") as small caps section headers. */
  [data-testid="stWidgetLabel"] p {
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #B3B3B3;
  }

  /* Selects, text input: rounded dark surfaces instead of Streamlit's default boxes. */
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
  [data-testid="stTextInput"] input {
    background-color: #1F1F1F !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
  }
  [data-testid="stTextInput"] input::placeholder { color: #6A6A6A; }

  /* Primary submit button ("Find similar tracks") as a Spotify-style pill. */
  [data-testid="stFormSubmitButton"] button,
  [data-testid="baseButton-primary"] {
    background-color: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.6rem !important;
    transition: transform 0.1s ease, background-color 0.15s ease;
  }
  [data-testid="stFormSubmitButton"] button:hover,
  [data-testid="baseButton-primary"]:hover {
    background-color: #1ED760 !important;
    transform: scale(1.02);
  }

  /* Suggestion buttons: quiet dark rows that light up on hover. */
  [data-testid="baseButton-secondary"] {
    background-color: #181818 !important;
    color: #E0E0E0 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 6px !important;
    text-align: left !important;
    justify-content: flex-start !important;
  }
  [data-testid="baseButton-secondary"]:hover {
    background-color: #282828 !important;
    border-color: #3A3A3A !important;
    color: #FFFFFF !important;
  }

  .seed-banner {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    background: linear-gradient(90deg, #1a2e22, #181818 65%);
    border-left: 3px solid #1DB954;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1.3rem 0 1.8rem;
    font-size: 0.92rem;
    color: #E0E0E0;
  }
  .seed-banner .eyebrow {
    display: block;
    text-transform: uppercase;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #1DB954;
    margin-bottom: 0.2rem;
  }
  .seed-banner b { color: #FFFFFF; }

  .track-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #181818;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.75rem 1.1rem;
    margin-bottom: 0.55rem;
    transition: background-color 0.15s ease, border-color 0.15s ease;
  }
  .track-card:hover {
    background: #232323;
    border-color: #2E2E2E;
  }
  .track-card .art {
    width: 2.75rem;
    height: 2.75rem;
    border-radius: 6px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
  }
  .track-card .rank {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #6A6A6A;
    width: 1.6rem;
    flex-shrink: 0;
    text-align: center;
  }
  .track-card .info { flex-grow: 1; min-width: 0; }
  .track-card .track-name {
    font-weight: 600;
    color: #FFFFFF;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .track-card .artist {
    color: #B3B3B3;
    font-size: 0.88rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .track-card .why-match {
    display: inline-block;
    color: #1DB954;
    background: rgba(29, 185, 84, 0.12);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-top: 0.35rem;
  }
  .track-card .similarity {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    color: #1DB954;
    font-size: 0.95rem;
    flex-shrink: 0;
    background: rgba(29, 185, 84, 0.1);
    padding: 0.25rem 0.6rem;
    border-radius: 500px;
  }

  .footer-note {
    color: #6A6A6A;
    font-size: 0.82rem;
    margin-top: 2.5rem;
  }
  .footer-note a { color: #1DB954; text-decoration: none; }
  .footer-note a:hover { text-decoration: underline; }
</style>
""",
    unsafe_allow_html=True,
)


REQUIRED_COLUMNS = {"track_id", "track_name", "artists", "primary_artist", "mood", *FEATURES}


@st.cache_resource
def load_state():
    interactions_dir = ensure_artifacts([LARGE_CATALOG_FILE])
    catalog_path = interactions_dir / LARGE_CATALOG_FILE
    tracks = pd.read_parquet(catalog_path)

    if not REQUIRED_COLUMNS.issubset(tracks.columns):
        # A previous deploy's copy of this file can survive a redeploy (Streamlit
        # Cloud's filesystem persists outside git-tracked paths), so a schema change
        # here silently serves stale data instead of the current build's version.
        # Delete it and force a fresh download from Hugging Face Hub.
        catalog_path.unlink()
        interactions_dir = ensure_artifacts([LARGE_CATALOG_FILE])
        tracks = pd.read_parquet(interactions_dir / LARGE_CATALOG_FILE)

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(tracks[FEATURES])
    idx_lookup = {tid: i for i, tid in enumerate(tracks["track_id"])}
    return tracks, feature_matrix, idx_lookup


FEATURE_COL_INDEX = {name: i for i, name in enumerate(FEATURES)}


def art_gradient(track_id: str) -> str:
    """Deterministic gradient per track (Spotify-style placeholder art), so the same
    song always gets the same "cover" instead of a random color on every rerun."""
    h = int(hashlib.md5(track_id.encode()).hexdigest(), 16)
    hue1 = h % 360
    hue2 = (hue1 + 45) % 360
    return f"linear-gradient(135deg, hsl({hue1},55%,42%), hsl({hue2},60%,22%))"


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
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    # A clicked suggestion is applied here, *before* the form below is instantiated —
    # session_state for a widget's key can only be set before that widget is created
    # in a given run, not after.
    pending = st.session_state.pop("_pending_pick", None)
    if pending is not None:
        st.session_state.seed_track_id = pending[0]
        st.session_state.song_query = pending[1]

    # Plain (non-form) st.text_input only syncs its typed value to the backend on
    # blur/Enter — clicking a separate button right after typing can race and submit
    # a stale value, which is exactly the "couldn't find a match" bug reported after
    # real typing (a plain script-level test of the same resolution logic passed
    # cleanly, which pointed at this browser-side sync race rather than the matching
    # code itself). st.form guarantees the input and the submit action are captured
    # together atomically, which a bare text_input + sibling button cannot.
    with st.form("search_form"):
        query = st.text_input(
            "Song",
            key="song_query",
            placeholder="Start typing a song or artist name…",
        )
        submitted = st.form_submit_button("🔍 Find similar tracks", type="primary")

    def pick(track_id: str, display: str) -> None:
        st.session_state._pending_pick = (track_id, display)

    def find_matches(df: pd.DataFrame, q: str) -> pd.DataFrame:
        exact = df[df["track_name"].str.lower() == q]
        starts = df[df["track_name"].str.lower().str.startswith(q, na=False)]
        contains = df[
            df["track_name"].str.lower().str.contains(q, na=False, regex=False)
            | df["artists"].str.lower().str.contains(q, na=False, regex=False)
        ]
        return next((m for m in (exact, starts, contains) if not m.empty), df.iloc[0:0])

    if submitted:
        st.session_state.last_query = query
        resolved_id = st.session_state.seed_track_id
        # If the text was edited after a suggestion was picked (or never picked at
        # all), resolve the current free-typed text to the closest match instead.
        current_label = None
        if resolved_id is not None:
            row = tracks.set_index("track_id").loc[resolved_id]
            current_label = f"{row.track_name} by {row.artists}"
        if query.strip() and query.strip() != current_label:
            q = query.strip().lower()
            # Search the mood/artist-filtered pool first, but a song title typed here
            # isn't guaranteed to belong to the currently selected filter (e.g. a
            # collab credited to a different primary artist, or a mood cluster that
            # happens to not include it) — falling back to the full catalog instead of
            # giving up is what makes this behave like an actual search box rather
            # than a filter that silently swallows valid queries.
            matches = find_matches(pool, q)
            if matches.empty:
                matches = find_matches(tracks, q)
            resolved_id = matches.iloc[0]["track_id"] if not matches.empty else None

        if resolved_id is None:
            st.warning("Couldn't find a matching song — try a suggestion below, or a different search.")
        else:
            st.session_state.seed_track_id = resolved_id

    # Suggestions are shown from the last *submitted* query (not live keystrokes,
    # since the query is now only reliably known at submission time inside the form
    # above) — click one to switch the seed track. Same pool-then-full-catalog
    # fallback as the resolution above, so suggestions don't go empty just because
    # the match lives outside the current mood/artist filter.
    if st.session_state.last_query.strip():
        q = st.session_state.last_query.strip().lower()
        name_match = pool["track_name"].str.lower().str.contains(q, na=False, regex=False)
        artist_match = pool["artists"].str.lower().str.contains(q, na=False, regex=False)
        suggestions = pool[name_match | artist_match]
        suggestion_scope = pool
        if suggestions.empty:
            name_match = tracks["track_name"].str.lower().str.contains(q, na=False, regex=False)
            artist_match = tracks["artists"].str.lower().str.contains(q, na=False, regex=False)
            suggestions = tracks[name_match | artist_match]
            suggestion_scope = tracks
        suggestions = suggestions.head(8)

        if not suggestions.empty:
            if suggestion_scope is not pool:
                st.caption("No matches in your current mood/artist filter — showing results from the full catalog:")
            else:
                st.caption("Other matches for your last search:")
            for row in suggestions.itertuples():
                display = f"{row.track_name} by {row.artists}"
                if st.button(display, key=f"suggest_{row.track_id}", use_container_width=True):
                    pick(row.track_id, display)
                    st.rerun()

    seed_track_id = st.session_state.seed_track_id
    if seed_track_id is not None:
        seed_row = tracks.set_index("track_id").loc[seed_track_id]

        seed_art = art_gradient(seed_track_id)
        st.markdown(
            f'<div class="seed-banner">'
            f'<div class="art" style="background:{seed_art}">♪</div>'
            f'<div><span class="eyebrow">Finding tracks like</span>'
            f'<b>{seed_row.track_name}</b> by {seed_row.artists}</div>'
            f'</div>',
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
                art_bg = art_gradient(tid)
                cards.append(
                    f'<div class="track-card">'
                    f'<div class="rank">{i:02d}</div>'
                    f'<div class="art" style="background:{art_bg}">♪</div>'
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
