"""One collection run: pull the dataset, collect every source through the gate, score what
has no label yet (new rows, and old rows from before a model existed), write the changed
months, push them, and report. Returns 1 when a selector alarm fired — after the data is
pushed — so GitHub's failure email is the alarm."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import hfsync, polite, store
from .collect import Report, collect, load_sources
from .config import ROOT, Settings
from .score import LANGS, Scorer, read_choice

log = logging.getLogger(__name__)


def run(
    settings: Settings,
    *,
    sources_path: Path = ROOT / "sources.yaml",
    choice_path: Path = ROOT / "model.yaml",
    now: datetime | None = None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> int:
    now = now or datetime.now(UTC)
    local = settings.data_dir is not None
    work = settings.data_dir if local else Path(tempfile.mkdtemp(prefix="sentiment-"))
    work.mkdir(parents=True, exist_ok=True)
    if not local:
        hfsync.pull(settings.dataset_repo, settings.hf_token, work)
    model_dir = settings.model_dir or (
        None if local else hfsync.pull_models(settings.model_repo, settings.hf_token)
    )
    scorer = Scorer(read_choice(choice_path), model_dir)

    existing = store.read_headlines(work)
    seen = set(existing["url_hash"])
    rows: list[dict] = []
    reports: list[Report] = []
    with polite.client() as client:
        gate = polite.Gate(client, clock=clock, sleep=sleep)
        for source in load_sources(sources_path):
            try:
                got, report = collect(
                    source, gate, now, scorer.label if source.body else None, seen
                )
            except Exception as e:  # noqa: BLE001 — one broken site never stops the others
                log.exception("source %s failed", source.name)
                got, report = [], Report(source.name, errors=[repr(e)])
            rows += got
            reports.append(report)

    df, fresh = store.merge(existing, rows)
    new_by_source = fresh["source"].value_counts().to_dict()
    for report in reports:
        report.new = int(new_by_source.get(report.source, 0))
    df, scored_months = score_missing(df, scorer)
    paths = store.write_headlines(work, df, set(store.months(fresh)) | scored_months)
    paths.append(
        store.write_status(
            work, {"run_at": now.isoformat(), "sources": [r.as_dict() for r in reports]}
        )
    )
    if not local:
        hfsync.push(
            settings.dataset_repo,
            settings.hf_token,
            work,
            paths,
            f"collect {now:%Y-%m-%d %H:%M} UTC: {len(fresh)} new",
        )
    _write_summary(summary(reports, int(df["label"].isna().sum()), scorer.missing()))
    return 1 if any(r.alarms for r in reports) else 0


def score_missing(df: pd.DataFrame, scorer: Scorer) -> tuple[pd.DataFrame, set[str]]:
    """Score every row without a label; returns the frame and the months that changed."""
    df = df.copy()
    changed: set[str] = set()
    for lang in LANGS:
        mask = df["label"].isna() & (df["lang"] == lang)
        if not mask.any():
            continue
        name, preds = scorer.score(df.loc[mask, "title"].tolist(), lang)
        got = [(i, p) for i, p in zip(df.index[mask], preds, strict=True) if p is not None]
        if name is None or not got:
            continue
        ids = [i for i, _ in got]
        df.loc[ids, "model"] = name
        df.loc[ids, "label"] = [p.label for _, p in got]
        df.loc[ids, ["p_negative", "p_neutral", "p_positive"]] = [list(p.probs) for _, p in got]
        changed |= set(store.months(df.loc[ids]))
    return df, changed


def summary(reports: list[Report], unscored: int, missing: list[str]) -> str:
    def cell(items) -> str:
        return "; ".join(items) or "—"

    lines = [
        "| Source | Fetched | New | Refused | Alarms | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        refused = [f"{reason}: {n}" for reason, n in sorted(r.refused.items())]
        lines.append(
            f"| {r.source} | {r.fetched} | {r.new} | {cell(refused)} | {cell(r.alarms)} "
            f"| {cell(r.errors)} |"
        )
    if unscored:
        why = f" (no model yet for: {', '.join(missing)})" if missing else ""
        lines += ["", f"Unscored headlines: {unscored}{why}"]
    return "\n".join(lines) + "\n"


def _write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as f:
            f.write(text)
