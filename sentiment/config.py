"""Settings from the environment. Every value has a working default except the token."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    hf_token: str | None = field(repr=False)  # never in a repr, a log line or an error
    dataset_repo: str
    model_repo: str
    data_dir: Path | None  # set: work on this local folder only, never touch the Hub
    model_dir: Path | None  # set: load models from here instead of the Hub
    metrics_path: Path
    label_mode: bool


def load(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    def get(key: str) -> str:
        return env.get(key, "").strip()

    def path(key: str) -> Path | None:
        return Path(get(key)) if get(key) else None

    return Settings(
        hf_token=get("HF_TOKEN") or None,
        dataset_repo=get("HF_DATASET_REPO") or "rationislabs/sentiment-headlines",
        model_repo=get("HF_MODEL_REPO") or "rationislabs/sentiment-models",
        data_dir=path("SENTIMENT_DATA_DIR"),
        model_dir=path("SENTIMENT_MODEL_DIR"),
        metrics_path=path("SENTIMENT_METRICS_PATH") or ROOT / "metrics.json",
        label_mode=get("LABEL_MODE") == "1",
    )
