"""ensure_artifacts: no download when files exist, and a mocked Hub download otherwise."""
import sys
import types

from src import artifacts


def test_existing_artifacts_are_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "INTERACTIONS_DIR", tmp_path)
    for name in artifacts.ARTIFACT_FILES:
        (tmp_path / name).write_bytes(b"x")

    def fail(**_kwargs):
        raise AssertionError("should not download")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fail))
    assert artifacts.ensure_artifacts() == tmp_path


def test_missing_artifacts_are_fetched_from_hub(tmp_path, monkeypatch):
    target_dir = tmp_path / "interactions"
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    monkeypatch.setattr(artifacts, "INTERACTIONS_DIR", target_dir)
    calls = []

    def fake_download(repo_id, filename, repo_type):
        calls.append((repo_id, filename, repo_type))
        path = cache / filename
        path.write_bytes(b"artifact")
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    out = artifacts.ensure_artifacts([artifacts.LARGE_CATALOG_FILE])
    assert out == target_dir
    assert calls == [(artifacts.HF_REPO_ID, artifacts.LARGE_CATALOG_FILE, "model")]
    assert (target_dir / artifacts.LARGE_CATALOG_FILE).read_bytes() == b"artifact"
