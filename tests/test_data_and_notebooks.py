"""Checks against files committed to the repo: the Spotify tracks CSV and the notebooks."""
from pathlib import Path

import nbformat
import pytest

from src.data_loading import AUDIO_FEATURES, SPOTIFY_TRACKS_CSV, load_spotify_tracks

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not SPOTIFY_TRACKS_CSV.exists(), reason="Spotify tracks CSV not present")
def test_load_spotify_tracks():
    df = load_spotify_tracks()
    assert len(df) > 50_000
    assert df["join_key"].is_unique
    assert set(AUDIO_FEATURES) <= set(df.columns)
    assert df[AUDIO_FEATURES].notna().all().all()


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_is_valid(path):
    # The notebooks download LastFM-1K and train on 13M events, so they are validated, not executed.
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert any(cell.cell_type == "code" for cell in nb.cells)
