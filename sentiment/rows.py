"""One stored row per headline: its columns and dtypes, and the URL key that deduplicates.
Article text is not a column, by design."""

from __future__ import annotations

import hashlib
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

LABELS = ("negative", "neutral", "positive")

DTYPES = {
    "url": "string",
    "url_hash": "string",
    "title": "string",
    "source": "string",
    "lang": "string",
    "published_at": "datetime64[us, UTC]",
    "fetched_at": "datetime64[us, UTC]",
    "rank": "Int64",
    "section": "string",
    "is_lead": "boolean",
    "model": "string",
    "label": "string",
    "p_negative": "Float64",
    "p_neutral": "Float64",
    "p_positive": "Float64",
    "body_label": "string",
}
COLUMNS = tuple(DTYPES)

_TRACKING = ("utm_", "at_")  # at_*: the BBC's feed tracking parameters


def normalize_url(url: str) -> str:
    """Same article, same key: lowercase scheme and host, no fragment, no tracking parameters."""
    s = urlsplit(url.strip())
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(s.query, keep_blank_values=True)
            if not k.lower().startswith(_TRACKING)
        ]
    )
    return urlunsplit((s.scheme.lower(), s.netloc.lower(), s.path or "/", query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16]


def new_row(
    *,
    url: str,
    title: str,
    source: str,
    lang: str,
    published_at: datetime | None,
    fetched_at: datetime,
) -> dict:
    row = dict.fromkeys(COLUMNS)
    clean = normalize_url(url)
    row.update(
        url=clean,
        url_hash=url_hash(clean),
        title=" ".join(title.split()),
        source=source,
        lang=lang,
        published_at=published_at,
        fetched_at=fetched_at,
    )
    return row
