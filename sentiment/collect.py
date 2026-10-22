"""Collect one source: its RSS feed, then (if switched on and the gate allows it) rank,
section and lead story from its front page, then (if switched on) the sentiment of each new
article's text. Article text is scored in memory and dropped; it never reaches a row, a
report or a log line."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import defusedxml.ElementTree as DefusedET
import feedparser
import httpx
import yaml
from bs4 import BeautifulSoup
from defusedxml import DefusedXmlException

from .polite import Gate, Refused
from .rows import new_row, url_hash

MIN_BODY = 200  # shorter extracted text: a paywall or consent wall, not an article
MAX_BODIES = 30  # per source per run

ScoreBody = Callable[[str, str], str | None]

__all__ = ["Refused", "Report", "Source", "collect", "load_sources"]


@dataclass(frozen=True)
class Source:
    name: str
    lang: str
    rss: str
    frontpage_url: str | None = None
    selectors: dict | None = None  # item, link, optional section (CSS)
    body_selector: str | None = None
    frontpage: bool = True
    body: bool = False


@dataclass
class Report:
    source: str
    fetched: int = 0
    new: int = 0
    refused: Counter = field(default_factory=Counter)
    alarms: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "fetched": self.fetched,
            "new": self.new,
            "refused": dict(self.refused),
            "alarms": list(self.alarms),
            "errors": list(self.errors),
        }


def load_sources(path: Path) -> list[Source]:
    sources = [Source(**entry) for entry in yaml.safe_load(path.read_text()) or []]
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate source names in {path}")
    for s in sources:
        if s.lang not in ("de", "en"):
            raise ValueError(f"{s.name}: lang must be de or en, not {s.lang!r}")
    return sources


def collect(
    source: Source,
    gate: Gate,
    now: datetime,
    score_body: ScoreBody | None = None,
    seen: set[str] | frozenset[str] = frozenset(),
    *,
    deadline: float = float("inf"),
) -> tuple[list[dict], Report]:
    """`deadline` is a gate.now() value shared by the whole run: past it, nothing more is
    fetched, so the run still pushes what it has before the job's time limit."""
    report = Report(source.name)
    if gate.now() > deadline:
        report.errors.append("skipped: run deadline reached")
        return [], report
    try:
        rows = read_feed(source, gate, now)
    except Refused as e:
        report.refused[e.reason] += 1
        return [], report
    except httpx.HTTPError as e:
        report.errors.append(f"feed: {e}")
        return [], report
    report.fetched = len(rows)
    if source.frontpage and source.frontpage_url:
        try:
            _front_page(source, gate, rows, report)
        except Exception as e:  # noqa: BLE001 — a broken selector must not cost the feed rows
            report.errors.append(f"front page: {e!r}")
    if source.body and score_body is not None:
        fresh = [r for r in rows if r["url_hash"] not in seen][:MAX_BODIES]
        _bodies(source, gate, fresh, score_body, report, deadline)
    return rows, report


def read_feed(source: Source, gate: Gate, now: datetime) -> list[dict]:
    r = gate.fetch(source.rss, "feed")
    r.raise_for_status()
    _refuse_entity_expansion(source.rss, r.content)
    rows: dict[str, dict] = {}
    for e in feedparser.parse(r.content).entries:
        title, link = (e.get("title") or "").strip(), e.get("link")
        if not title or not link or urlsplit(link).scheme not in ("http", "https"):
            continue  # a javascript: or file: link would be clickable in the dashboard
        t = e.get("published_parsed") or e.get("updated_parsed")
        row = new_row(
            url=link,
            title=title,
            source=source.name,
            lang=source.lang,
            published_at=datetime(*t[:6], tzinfo=UTC) if t else None,
            fetched_at=now,
        )
        rows.setdefault(row["url_hash"], row)
    return list(rows.values())


def _refuse_entity_expansion(url: str, content: bytes) -> None:
    """A feed that declares entities ("billion laughs") is refused before feedparser expands
    them. Malformed XML is left to feedparser, which copes with most of it."""
    try:
        DefusedET.fromstring(content)
    except (DefusedXmlException, ValueError) as e:  # ValueError: an encoding expat cannot read
        raise Refused(url, "unsafe-xml") from e
    except ET.ParseError:
        pass


def front_page_ranks(
    html: str, base_url: str, selectors: dict
) -> dict[str, tuple[int, str | None]]:
    """url_hash → (rank, section), rank counting only items that have a link."""
    out: dict[str, tuple[int, str | None]] = {}
    rank = 0
    for item in BeautifulSoup(html, "lxml").select(selectors["item"]):
        a = item.select_one(selectors["link"]) if selectors.get("link") else item
        href = a.get("href") if a is not None else None
        if not href:
            continue
        rank += 1
        sec = item.select_one(selectors["section"]) if selectors.get("section") else None
        out.setdefault(
            url_hash(urljoin(base_url, href)), (rank, sec.get_text(strip=True) if sec else None)
        )
    return out


def _front_page(source: Source, gate: Gate, rows: list[dict], report: Report) -> None:
    try:
        r = gate.fetch(source.frontpage_url, "scrape")
        r.raise_for_status()
    except Refused as e:
        report.refused[e.reason] += 1
        return
    except httpx.HTTPError as e:
        report.errors.append(f"front page: {e}")
        return
    if not source.selectors:
        report.alarms.append("front page allowed but no selectors configured")
        return
    ranks = front_page_ranks(r.text, source.frontpage_url, source.selectors)
    if not ranks:
        report.alarms.append(f"selector {source.selectors['item']!r} matched nothing")
        return
    for row in rows:
        rank, section = ranks.get(row["url_hash"], (None, None))
        row.update(rank=rank, section=section, is_lead=rank == 1)


def body_text(html: str, selector: str | None) -> str:
    soup = BeautifulSoup(html, "lxml")
    for css in filter(None, (selector, "article p", "p")):
        text = " ".join(t for el in soup.select(css) if (t := el.get_text(" ", strip=True)))
        if text:
            return text
    return ""


def _bodies(
    source: Source,
    gate: Gate,
    rows: list[dict],
    score_body: ScoreBody,
    report: Report,
    deadline: float,
) -> None:
    for i, row in enumerate(rows):
        if gate.now() > deadline:
            report.errors.append(f"body: run deadline reached, {len(rows) - i} skipped")
            return
        try:
            r = gate.fetch(row["url"], "scrape")
        except Refused as e:
            report.refused[e.reason] += 1
            continue
        except httpx.HTTPError as e:
            report.errors.append(f"body: {e}")
            continue
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            continue
        try:
            text = body_text(r.text, source.body_selector)
            if len(text) >= MIN_BODY:
                row["body_label"] = score_body(text, source.lang)
        except Exception as e:  # noqa: BLE001 — one odd article never costs the source
            # The type only: a message could quote the article, which is never stored.
            report.errors.append(f"body: {type(e).__name__}")
