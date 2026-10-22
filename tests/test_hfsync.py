from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from sentiment import config, hfsync


def test_pull_downloads_into_dest(monkeypatch, tmp_path):
    calls = {}

    def fake(repo_id, **kw):
        calls.update(kw, repo_id=repo_id)
        return str(kw["local_dir"])

    monkeypatch.setattr(hfsync, "snapshot_download", fake)
    assert hfsync.pull("me/ds", "hf_fake", tmp_path / "w") == tmp_path / "w"
    assert calls == {
        "repo_id": "me/ds",
        "repo_type": "dataset",
        "token": "hf_fake",
        "local_dir": tmp_path / "w",
    }


def test_pull_models_uses_the_hub_cache(monkeypatch):
    seen = {}

    def fake(repo_id, **kw):
        seen.update(kw)
        return "/cache/models--me--m/snapshots/abc"

    monkeypatch.setattr(hfsync, "snapshot_download", fake)
    rev = "c" * 40
    assert hfsync.pull_models("me/m", "hf_fake", rev) == Path("/cache/models--me--m/snapshots/abc")
    assert seen == {
        "repo_type": "model",
        "token": "hf_fake",
        "revision": rev,
        "allow_patterns": list(hfsync.MODEL_FILES),
    }


def test_pull_models_without_a_pinned_revision_downloads_nothing(monkeypatch):
    monkeypatch.setattr(hfsync, "snapshot_download", lambda *a, **k: pytest.fail("no Hub call"))
    assert hfsync.pull_models("me/m", "hf_fake", None) is None


def test_pull_models_missing_repo_is_none(monkeypatch):
    def fake(repo_id, **kw):
        request = httpx.Request("GET", "https://huggingface.co/api/models/me/m")
        raise RepositoryNotFoundError("nope", response=httpx.Response(404, request=request))

    monkeypatch.setattr(hfsync, "snapshot_download", fake)
    assert hfsync.pull_models("me/m", None, "c" * 40) is None


class FakeApi:
    def __init__(self, token=None):
        self.token = token
        FakeApi.last = self
        self.commits = []
        self.folders = []
        self.repos = []

    def create_commit(self, repo_id, operations, *, commit_message, repo_type):
        self.commits.append(
            (
                repo_id,
                [(o.path_in_repo, o.path_or_fileobj) for o in operations],
                commit_message,
                repo_type,
            )
        )

    def upload_folder(self, *, repo_id, folder_path, repo_type, commit_message):
        self.folders.append((repo_id, str(folder_path), repo_type, commit_message))
        return type("CommitInfo", (), {"oid": "d" * 40})()

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.repos.append((repo_id, repo_type, private, exist_ok))


def test_push_one_commit_with_all_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(hfsync, "HfApi", FakeApi)
    (tmp_path / "headlines").mkdir()
    (tmp_path / "headlines" / "2026-10.parquet").write_bytes(b"x")
    (tmp_path / "status.json").write_text("{}")
    hfsync.push(
        "me/ds", "hf_fake", tmp_path, ["headlines/2026-10.parquet", "status.json"], "collect"
    )
    [(repo, ops, msg, kind)] = FakeApi.last.commits
    assert (repo, msg, kind) == ("me/ds", "collect", "dataset")
    assert ops == [
        ("headlines/2026-10.parquet", str(tmp_path / "headlines" / "2026-10.parquet")),
        ("status.json", str(tmp_path / "status.json")),
    ]
    assert FakeApi.last.token == "hf_fake"


def test_push_nothing_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.setattr(hfsync, "HfApi", lambda token=None: pytest.fail("no API call expected"))
    hfsync.push("me/ds", "hf_fake", tmp_path, [], "collect")


def test_push_folder_and_init_repos(monkeypatch, tmp_path):
    monkeypatch.setattr(hfsync, "HfApi", FakeApi)
    assert hfsync.push_folder("me/m", "hf_fake", tmp_path, "models") == "d" * 40
    assert FakeApi.last.folders == [("me/m", str(tmp_path), "model", "models")]
    hfsync.init_repos(
        config.load({"HF_TOKEN": "hf_fake", "HF_DATASET_REPO": "me/ds", "HF_MODEL_REPO": "me/m"})
    )
    assert FakeApi.last.repos == [("me/ds", "dataset", True, True), ("me/m", "model", True, True)]


def test_data_root_local_never_touches_hub(monkeypatch, tmp_path):
    monkeypatch.setattr(hfsync, "snapshot_download", lambda *a, **k: pytest.fail("no Hub call"))
    s = config.load({"SENTIMENT_DATA_DIR": str(tmp_path)})
    assert hfsync.data_root(s) == tmp_path


def test_data_root_hub_pulls_into_a_persistent_cache(monkeypatch, tmp_path):
    # Not /tmp: unpushed hand labels live here until the owner pushes them.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(hfsync, "snapshot_download", lambda repo_id, **kw: str(kw["local_dir"]))
    root = hfsync.data_root(config.load({"HF_DATASET_REPO": "me/ds"}))
    assert root == tmp_path / "cache" / "sentiment-data"
