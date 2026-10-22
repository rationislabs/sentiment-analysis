"""The data behind the dashboard's tabs, as plain functions (app/main.py only draws)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .rows import LABELS
from .store import GOLD_DTYPES, split_of

_MARKDOWN = re.compile(r"([!-/:-@\[-`{-~])")  # every ASCII punctuation character


def plain_markdown(text: str) -> str:
    """text that Streamlit's markdown shows as written: no image, link, colour or maths."""
    return _MARKDOWN.sub(r"\\\1", text)


GOLD_TARGET = 600
RELABEL_N = 50
RELABEL_AFTER = timedelta(days=7)
GUIDELINE = (
    "**Label the event as a reader experiences it, not the writer's tone.** "
    '"Flood kills 12" is negative; "ECB holds rates" is neutral; '
    '"Vaccine approved for children" is positive.'
)
NO_GOLD = (
    "No gold set yet. Label headlines locally (`uv run inv app`, the Label tab), then run "
    "`uv run inv evaluate`."
)


def trends(df: pd.DataFrame, lang: str | None = None) -> pd.DataFrame:
    """Daily mean of P(positive) − P(negative) per source: −1 all negative, +1 all positive."""
    d = df[df["label"].notna()]
    if lang:
        d = d[d["lang"] == lang]
    if d.empty:
        return pd.DataFrame()
    polarity = (d["p_positive"] - d["p_negative"]).astype("float64")
    day = d["fetched_at"].dt.floor("D")
    return polarity.groupby([day, d["source"]]).mean().unstack("source").sort_index()


def status_table(status: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Source": s["source"],
                "Fetched": s["fetched"],
                "New": s["new"],
                "Refused": ", ".join(f"{k}: {v}" for k, v in sorted(s["refused"].items())),
                "Alarms": "; ".join(s["alarms"]),
                "Errors": len(s["errors"]),
            }
            for s in status["sources"]
        ]
    )


def headline_table(df: pd.DataFrame, query: str = "") -> pd.DataFrame:
    d = df
    if query.strip():
        d = d[d["title"].str.contains(query.strip(), case=False, regex=False, na=False)]
    cols = ["title", "source", "lang", "label", "published_at", "rank", "url"]
    return d.sort_values("fetched_at", ascending=False)[cols].head(500).reset_index(drop=True)


def read_metrics(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def scores_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": r["model"],
                "n": r["n"],
                "Macro-F1": round(r["macro_f1"], 3),
                "95 % CI": f"{r['ci'][0]:.3f}–{r['ci'][1]:.3f}",
            }
            for r in results
        ]
    )


def next_to_label(headlines: pd.DataFrame, gold: pd.DataFrame) -> pd.Series | None:
    """An unlabelled headline from the language, then the source, with the fewest labels."""
    pool = headlines[~headlines["url_hash"].isin(gold["url_hash"])]
    if pool.empty:
        return None
    by_lang, by_source = gold["lang"].value_counts(), gold["source"].value_counts()
    lang = min(pool["lang"].unique(), key=lambda x: (by_lang.get(x, 0), x))
    pool = pool[pool["lang"] == lang]
    source = min(pool["source"].unique(), key=lambda x: (by_source.get(x, 0), x))
    return pool[pool["source"] == source].sample(1, random_state=len(gold)).iloc[0]


def add_label(gold: pd.DataFrame, row: pd.Series, label: str, now: datetime) -> pd.DataFrame:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    new = pd.DataFrame(
        [
            {
                "url_hash": row["url_hash"],
                "title": row["title"],
                "source": row["source"],
                "lang": row["lang"],
                "label": label,
                "labelled_at": now,
                "split": split_of(row["url_hash"]),
                "label2": None,
                "relabelled_at": None,
            }
        ]
    ).astype(GOLD_DTYPES)
    return pd.concat([gold, new], ignore_index=True)


def relabel_queue(gold: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """The oldest labels at least a week old, blind-relabelled until RELABEL_N are done."""
    left = RELABEL_N - int(gold["label2"].notna().sum())
    if left <= 0:
        return gold.iloc[0:0]
    due = gold[gold["label2"].isna() & (gold["labelled_at"] <= now - RELABEL_AFTER)]
    return due.sort_values("labelled_at").head(left)


def add_relabel(gold: pd.DataFrame, url_hash: str, label: str, now: datetime) -> pd.DataFrame:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    gold = gold.copy()
    hit = gold["url_hash"] == url_hash
    gold.loc[hit, "label2"] = label
    gold.loc[hit, "relabelled_at"] = now
    return gold
