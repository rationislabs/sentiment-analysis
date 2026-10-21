"""The honest-metrics part: macro-F1 with a bootstrap confidence interval, the winner rule,
self-agreement, and the examples worth looking at. Pure functions; no model, no I/O."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .rows import LABELS
from .score import NAMES, Prediction


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0))


def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of macro-F1 over n resamples of the rows (with replacement)."""
    t, p = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    rng = np.random.default_rng(seed)
    scores = [macro_f1(t[i], p[i]) for i in (rng.integers(0, len(t), len(t)) for _ in range(n))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


@dataclass
class Result:
    model: str
    lang: str
    split: str
    n: int
    macro_f1: float
    ci: tuple[float, float]
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]  # rows: gold, columns: predicted, both in LABELS order


def result(model: str, lang: str, split: str, y_true, y_pred) -> Result:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    return Result(
        model=model,
        lang=lang,
        split=split,
        n=len(y_true),
        macro_f1=macro_f1(y_true, y_pred),
        ci=bootstrap_ci(y_true, y_pred),
        per_class={
            lab: {"precision": float(precision[i]), "recall": float(recall[i])}
            for i, lab in enumerate(LABELS)
        },
        confusion=confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    )


def winner(dev: list[Result]) -> str | None:
    """Highest dev macro-F1; among the models whose interval overlaps the best one's, the
    cheapest wins. The majority-class floor never wins."""
    candidates = [r for r in dev if r.model in NAMES]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r.macro_f1)
    close = [r for r in candidates if r.ci[0] <= best.ci[1] and best.ci[0] <= r.ci[1]]
    return min(close, key=lambda r: NAMES.index(r.model)).model


def majority(labels) -> str:
    counts = Counter(labels)
    return max(LABELS, key=lambda lab: (counts[lab], -LABELS.index(lab)))


def kappa(a, b) -> float | None:
    """Cohen's kappa of the owner's first and blind second labels; None below 2 pairs."""
    if len(a) < 2:
        return None
    return float(cohen_kappa_score(a, b, labels=list(LABELS)))


def worst_errors(titles, y_true, preds: list[Prediction], k: int = 10) -> list[dict]:
    wrong = [
        {"title": t, "gold": g, "predicted": p.label, "confidence": round(max(p.probs), 3)}
        for t, g, p in zip(titles, y_true, preds, strict=True)
        if p.label != g
    ]
    return sorted(wrong, key=lambda w: -w["confidence"])[:k]


def disagreements(titles, y_true, labels_by_model: dict[str, list[str]], k: int = 20) -> list[dict]:
    names = list(labels_by_model)
    rows = []
    for i, (t, g) in enumerate(zip(titles, y_true, strict=True)):
        preds = {n: labels_by_model[n][i] for n in names}
        if len(set(preds.values())) > 1:
            rows.append({"title": t, "gold": g, "predictions": preds})
    return sorted(rows, key=lambda r: (-len(set(r["predictions"].values())), r["title"]))[:k]
