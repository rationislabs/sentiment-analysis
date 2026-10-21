"""The dataset on disk: one Parquet file per month of fetch time (headlines/YYYY-MM.parquet),
the gold labels (gold.parquet) and the last run's report (status.json). The Hub repo has the
same layout; hfsync moves it."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .rows import COLUMNS, DTYPES

HEADLINES = "headlines"
GOLD = "gold.parquet"
STATUS = "status.json"
GOLD_DTYPES = {
    "url_hash": "string",
    "title": "string",
    "source": "string",
    "lang": "string",
    "label": "string",
    "labelled_at": "datetime64[us, UTC]",
    "split": "string",  # dev | test
    "label2": "string",  # the blind relabel, a week later
    "relabelled_at": "datetime64[us, UTC]",
}


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(COLUMNS)).astype(DTYPES)


def merge(existing: pd.DataFrame, rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(everything, the fresh rows): a URL already stored, or seen earlier in this batch, is
    dropped — the first source to bring it keeps it."""
    new = frame(rows).drop_duplicates("url_hash")
    fresh = new[~new["url_hash"].isin(existing["url_hash"])].reset_index(drop=True)
    return pd.concat([existing, fresh], ignore_index=True), fresh


def months(df: pd.DataFrame) -> list[str]:
    return sorted(df["fetched_at"].dt.strftime("%Y-%m").unique())


def read_headlines(root: Path) -> pd.DataFrame:
    files = sorted((root / HEADLINES).glob("*.parquet"))
    if not files:
        return frame([])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).astype(DTYPES)


def write_headlines(root: Path, df: pd.DataFrame, which: set[str]) -> list[str]:
    """Rewrite the given months only; returns their paths relative to root."""
    (root / HEADLINES).mkdir(parents=True, exist_ok=True)
    key = df["fetched_at"].dt.strftime("%Y-%m")
    out = []
    for month in sorted(which):
        rel = f"{HEADLINES}/{month}.parquet"
        df[key == month].sort_values(["fetched_at", "url_hash"]).to_parquet(root / rel, index=False)
        out.append(rel)
    return out


def read_gold(root: Path) -> pd.DataFrame:
    path = root / GOLD
    if not path.exists():
        return pd.DataFrame(columns=list(GOLD_DTYPES)).astype(GOLD_DTYPES)
    return pd.read_parquet(path).astype(GOLD_DTYPES)


def write_gold(root: Path, gold: pd.DataFrame) -> str:
    gold.to_parquet(root / GOLD, index=False)
    return GOLD


def split_of(url_hash: str) -> str:
    """A third of the gold set is dev (model choice), the rest test (reported once)."""
    return "dev" if int(url_hash, 16) % 3 == 0 else "test"


def read_status(root: Path) -> dict | None:
    path = root / STATUS
    return json.loads(path.read_text()) if path.exists() else None


def write_status(root: Path, status: dict) -> str:
    (root / STATUS).write_text(json.dumps(status, indent=1) + "\n")
    return STATUS
