"""The public training sets as one frame (text, label, lang, dataset). Downloaded on use from
the Hugging Face Hub, never committed. Licences: README and model card."""

from __future__ import annotations

import pandas as pd

from sentiment.rows import LABELS

# name: (lang, Hub id, config, text column, label column)
SETS = {
    "tweet_eval": ("en", "cardiffnlp/tweet_eval", "sentiment", "text", "label"),
    "sb10k": ("de", "Alienmaster/SB10k", None, "Text", "Sentiment"),
    "germeval2017": ("de", "uhhlt/GermEval2017", None, "Text", "Sentiment"),
}
LIMIT_PER_LANG = 20_000


def to_frame(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    lang, _, _, text, label = SETS[name]
    df = raw[[text, label]].rename(columns={text: "text", label: "label"})
    if pd.api.types.is_integer_dtype(df["label"]):
        df["label"] = df["label"].map(dict(enumerate(LABELS)))  # 0 negative, 1 neutral, 2 positive
    df = df[df["label"].isin(LABELS) & df["text"].notna()]
    return df.assign(lang=lang, dataset=name).reset_index(drop=True)


def cap(df: pd.DataFrame, limit: int, seed: int = 0) -> pd.DataFrame:
    """At most `limit` rows per language, so neither language drowns the other."""
    return pd.concat(
        [g.sample(min(len(g), limit), random_state=seed) for _, g in df.groupby("lang")],
        ignore_index=True,
    )


def load_public(limit: int = LIMIT_PER_LANG) -> pd.DataFrame:
    from datasets import load_dataset

    frames = [
        to_frame(name, load_dataset(hub_id, cfg, split="train").to_pandas())
        for name, (_, hub_id, cfg, _, _) in SETS.items()
    ]
    return cap(pd.concat(frames, ignore_index=True), limit)
