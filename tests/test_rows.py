from datetime import UTC, datetime

from sentiment import rows


def test_labels_order():
    assert rows.LABELS == ("negative", "neutral", "positive")


def test_columns_follow_dtypes():
    assert rows.COLUMNS == tuple(rows.DTYPES)
    assert rows.COLUMNS[:2] == ("url", "url_hash")
    assert "body_label" in rows.COLUMNS and "text" not in rows.COLUMNS


def test_normalize_drops_tracking_and_fragment():
    url = "https://WWW.BBC.co.uk/news/articles/c8?at_medium=RSS&at_campaign=rss&id=7#top"
    assert rows.normalize_url(url) == "https://www.bbc.co.uk/news/articles/c8?id=7"


def test_normalize_drops_utm_keeps_other_params():
    url = "https://example.org/a?utm_source=x&page=2&UTM_medium=y"
    assert rows.normalize_url(url) == "https://example.org/a?page=2"


def test_normalize_empty_path_is_slash():
    assert rows.normalize_url("https://example.org") == "https://example.org/"


def test_url_hash_is_stable_and_ignores_tracking():
    a = rows.url_hash("https://example.org/a?utm_source=x")
    assert a == rows.url_hash("https://example.org/a")
    assert len(a) == 16 and int(a, 16) >= 0


def test_new_row_has_every_column():
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)
    r = rows.new_row(
        url="https://example.org/a?utm_source=x",
        title="  A   title\n",
        source="hn",
        lang="en",
        published_at=None,
        fetched_at=now,
    )
    assert tuple(r) == rows.COLUMNS
    assert r["url"] == "https://example.org/a"
    assert r["title"] == "A title"
    assert r["url_hash"] == rows.url_hash("https://example.org/a")
    assert r["label"] is None and r["rank"] is None
