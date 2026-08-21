"""Fetches the trained model artifacts, downloading from Hugging Face Hub on first
run if they aren't already present locally (e.g. a fresh clone, or a fresh deploy on
Streamlit Community Cloud, where data/raw/interactions/ is gitignored and never
committed — see README's Architecture section)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERACTIONS_DIR = PROJECT_ROOT / "data" / "raw" / "interactions"

HF_REPO_ID = "ChrisOmara/music-taste-recommender-artifacts"
ARTIFACT_FILES = ["als_model.pkl", "joined_train.parquet"]
LARGE_CATALOG_FILE = "large_catalog.parquet"


def ensure_artifacts(filenames: list[str] | None = None) -> Path:
    """Returns INTERACTIONS_DIR with the given files guaranteed present (default:
    als_model.pkl + joined_train.parquet), downloading any missing ones from Hugging
    Face Hub. Pass filenames=[LARGE_CATALOG_FILE] for the demo app's larger catalog."""
    filenames = filenames if filenames is not None else ARTIFACT_FILES
    INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in filenames if not (INTERACTIONS_DIR / f).exists()]
    if not missing:
        return INTERACTIONS_DIR

    from huggingface_hub import hf_hub_download

    for filename in missing:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=filename, repo_type="model"
        )
        target = INTERACTIONS_DIR / filename
        if not target.exists():
            try:
                target.symlink_to(downloaded_path)
            except OSError:
                # Windows often disallows symlinks without dev mode/admin rights.
                import shutil
                shutil.copy(downloaded_path, target)

    return INTERACTIONS_DIR
