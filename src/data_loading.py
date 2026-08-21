"""Loaders for the two source datasets and the name-based join between them."""
from pathlib import Path

import pandas as pd

from .normalize import join_key, primary_artist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPOTIFY_TRACKS_CSV = PROJECT_ROOT / "data" / "raw" / "spotify_tracks" / "dataset.csv"

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature",
]


def load_spotify_tracks() -> pd.DataFrame:
    """Load the 114k-track Spotify Tracks Dataset, deduplicated on (artist, track).

    A handful of rows in the raw CSV have stray Latin-1 bytes mixed into an otherwise
    UTF-8 file (e.g. a raw 0xE1 for "á" instead of the proper 2-byte UTF-8 sequence) —
    a real quirk in the source data, not a parsing bug. encoding_errors="replace" keeps
    those few rows loadable (as "�") rather than crashing the whole 114k-row read.
    """
    df = pd.read_csv(SPOTIFY_TRACKS_CSV, index_col=0, encoding_errors="replace")
    df["primary_artist"] = df["artists"].map(primary_artist)
    df["join_key"] = [
        join_key(a, t) for a, t in zip(df["primary_artist"], df["track_name"])
    ]
    # Multiple entries can share a join_key (e.g. re-releases); keep the most popular.
    df = df.sort_values("popularity", ascending=False).drop_duplicates("join_key")
    return df.reset_index(drop=True)


def load_lastfm_1k(split: str = "train") -> pd.DataFrame:
    """Load LastFM-1K listening events via the HuggingFace `datasets` library.

    Requires network access on first call (cached locally by `datasets` after that).
    Returns columns: user_id, artist_name, track_name, timestamp (+ raw indices).
    """
    from datasets import load_dataset

    ds = load_dataset("matthewfranglen/lastfm-1k", split=split)
    df = ds.to_pandas()
    df["join_key"] = [
        join_key(a, t) for a, t in zip(df["artist_name"], df["track_name"])
    ]
    return df


def build_joined_interactions(
    interactions: pd.DataFrame, tracks: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Attach Spotify audio features to LastFM interactions via the name-based join.

    Returns (joined_df, coverage_stats). coverage_stats is meant to be reported
    verbatim in notebooks/01_data_audit.ipynb and reports/case_study.md.
    """
    matched_keys = set(tracks["join_key"])
    interactions = interactions.copy()
    interactions["matched"] = interactions["join_key"].isin(matched_keys)

    n_events = len(interactions)
    n_matched_events = int(interactions["matched"].sum())
    n_unique_tracks = interactions["join_key"].nunique()
    n_matched_unique_tracks = interactions.loc[interactions["matched"], "join_key"].nunique()

    coverage_stats = {
        "n_events": n_events,
        "n_matched_events": n_matched_events,
        "event_coverage_pct": round(100 * n_matched_events / n_events, 2),
        "n_unique_tracks_in_interactions": n_unique_tracks,
        "n_matched_unique_tracks": n_matched_unique_tracks,
        "unique_track_coverage_pct": round(
            100 * n_matched_unique_tracks / n_unique_tracks, 2
        ),
    }

    joined = interactions.loc[interactions["matched"]].merge(
        tracks[["join_key", "track_id"] + AUDIO_FEATURES + ["track_genre", "popularity"]],
        on="join_key",
        how="left",
    )
    return joined, coverage_stats
