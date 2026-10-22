"""Moving the dataset and the models between the Hugging Face Hub and local folders.
Both repos are private; the token comes from the environment and is never logged."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

from .config import Settings

log = logging.getLogger(__name__)


def pull(repo: str, token: str | None, dest: Path) -> Path:
    """The dataset repo's files as real, writable files in dest."""
    snapshot_download(repo, repo_type="dataset", token=token, local_dir=dest)
    return dest


# What the pipeline loads from the model repo: skops files and the DistilBERT folder.
MODEL_FILES = ("tfidf-*.skops", "distilbert/*")


def pull_models(repo: str, token: str | None, revision: str | None) -> Path | None:
    """The pinned commit of the model repo in the Hub cache (~/.cache/huggingface, which
    Actions caches), or None when nothing is pinned yet or the repo does not exist."""
    if revision is None:
        log.warning("no models.lock: headlines stay unscored")
        return None
    try:
        return Path(
            snapshot_download(
                repo,
                repo_type="model",
                token=token,
                revision=revision,
                allow_patterns=list(MODEL_FILES),
            )
        )
    except RepositoryNotFoundError:
        log.warning("model repo %s not found: headlines stay unscored", repo)
        return None


def push(repo: str, token: str | None, root: Path, paths: list[str], message: str) -> None:
    """One commit with every changed file."""
    if not paths:
        return
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(root / p)) for p in paths]
    HfApi(token=token).create_commit(repo, ops, commit_message=message, repo_type="dataset")


def push_folder(repo: str, token: str | None, folder: Path, message: str) -> str:
    """Upload folder; returns the new commit's hash, for models.lock."""
    info = HfApi(token=token).upload_folder(
        repo_id=repo, folder_path=folder, repo_type="model", commit_message=message
    )
    return info.oid


def init_repos(settings: Settings) -> None:
    """Create both repos, private, if they do not exist."""
    api = HfApi(token=settings.hf_token)
    api.create_repo(settings.dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(settings.model_repo, repo_type="model", private=True, exist_ok=True)


def data_root(settings: Settings) -> Path:
    """Where the dataset is: the local folder in local mode, else a pull from the Hub into a
    persistent cache (not /tmp: unpushed hand labels live there until they are pushed)."""
    if settings.data_dir is not None:
        return settings.data_dir
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return pull(settings.dataset_repo, settings.hf_token, cache / "sentiment-data")
