"""The dataset on disk: one Parquet file per month of fetch time (headlines/YYYY-MM.parquet),
the gold labels (gold.parquet) and the last run's report (status.json). The Hub repo has the
same layout; hfsync moves it."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from .rows import COLUMNS, DTYPES

HEADLINES = "headlines"
GOLD = "gold.parquet"
# Labels not pushed yet. Never on the Hub, so a pull (which rewrites gold.parquet) leaves it
# alone.
LOCAL_GOLD = "gold.local.parquet"
STATUS = "status.json"
# Every Streamlit session of one app process saves through here: one at a time, so two tabs
# saving in the same instant never drop a label.
_LOCAL = threading.Lock()
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


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(GOLD_DTYPES)).astype(GOLD_DTYPES)
    return pd.read_parquet(path).astype(GOLD_DTYPES)


def read_gold(root: Path) -> pd.DataFrame:
    """The Hub's gold set with the local, unpushed labels on top (a local label wins)."""
    hub, local = _read(root / GOLD), _read(root / LOCAL_GOLD)
    if local.empty:
        return hub
    if hub.empty:
        return local
    both = pd.concat([hub, local], ignore_index=True)
    return both.drop_duplicates("url_hash", keep="last").reset_index(drop=True)


def write_gold(root: Path, gold: pd.DataFrame) -> str:
    """Add or replace these labels in the local file; publish_gold puts them in gold.parquet
    for a push. The file is read again first: another tab may have saved labels since."""
    with _LOCAL:
        both = pd.concat([_read(root / LOCAL_GOLD), gold.astype(GOLD_DTYPES)], ignore_index=True)
        both.drop_duplicates("url_hash", keep="last").to_parquet(root / LOCAL_GOLD, index=False)
    return LOCAL_GOLD


def publish_gold(root: Path) -> pd.DataFrame:
    """Write the merged set to gold.parquet, the file a push sends; returns what it wrote."""
    gold = read_gold(root)
    gold.to_parquet(root / GOLD, index=False)
    return gold


def clear_local_gold(root: Path, published: pd.DataFrame) -> None:
    """After a successful push, drop the local labels the push carried. A label saved (or
    changed) in another tab while the push ran stays for the next one."""
    with _LOCAL:
        local = _read(root / LOCAL_GOLD)
        key = list(GOLD_DTYPES)
        sent = local.merge(published.astype(GOLD_DTYPES), on=key, how="left", indicator=True)
        left = local[(sent["_merge"] == "left_only").to_numpy()]
        if left.empty:
            (root / LOCAL_GOLD).unlink(missing_ok=True)
        else:
            left.to_parquet(root / LOCAL_GOLD, index=False)


def split_of(url_hash: str) -> str:
    """A third of the gold set is dev (model choice), the rest test (reported once)."""
    return "dev" if int(url_hash, 16) % 3 == 0 else "test"


def read_status(root: Path) -> dict | None:
    path = root / STATUS
    return json.loads(path.read_text()) if path.exists() else None


def write_status(root: Path, status: dict) -> str:
    (root / STATUS).write_text(json.dumps(status, indent=1) + "\n")
    return STATUS
