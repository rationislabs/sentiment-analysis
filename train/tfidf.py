"""One TF-IDF + logistic regression pipeline per language."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import skops.io
from sklearn.pipeline import Pipeline

from sentiment.score import TFIDF_FILE, Tfidf


def fit(df: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        lang: Tfidf.build().fit(g["text"].tolist(), g["label"].tolist())
        for lang, g in df.groupby("lang")
    }


def save(pipes: dict[str, Pipeline], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, p in pipes.items():
        skops.io.dump(p, out_dir / TFIDF_FILE.format(lang=lang))
