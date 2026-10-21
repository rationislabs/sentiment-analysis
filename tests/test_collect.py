from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from sentiment import collect, polite
from sentiment.collect import Source
from tests.conftest import FIXTURES, robots_txt

NOW = datetime(2026, 10, 8, 14, tzinfo=UTC)
ALLOW_ALL = "User-agent: *\nAllow: /\n"
ROOT = Path(__file__).resolve().parent.parent
HN = Source(
    name="hn",
    lang="en",
    rss="https://news.ycombinator.com/rss",
    frontpage_url="https://news.ycombinator.com/",
    selectors={"item": "tr.athing", "link": "span.titleline > a"},
)
GUARDIAN = Source(
    name="guardian",
    lang="en",
    rss="https://www.theguardian.com/world/rss",
    frontpage_url="https://www.theguardian.com/international",
)


def feed(name: str) -> bytes:
    return (FIXTURES / "feeds" / f"{name}.xml").read_bytes()


def serve_hn(web, frontpage: str | None = None):
    web.get("https://news.ycombinator.com/robots.txt").respond(
        200, text=robots_txt("news.ycombinator.com")
    )
    web.get("https://news.ycombinator.com/.well-known/tdmrep.json").respond(404)
    web.get("https://news.ycombinator.com/rss").respond(200, content=feed("hn"))
    web.get("https://news.ycombinator.com/").respond(
        200,
        text=frontpage or (FIXTURES / "frontpages" / "hn.html").read_text(),
        headers={"content-type": "text/html; charset=utf-8"},
    )


# --- sources.yaml -----------------------------------------------------------------------------


def test_sources_yaml_loads():
    sources = collect.load_sources(ROOT / "sources.yaml")
    assert [s.name for s in sources] == [
        "bbc",
        "guardian",
        "hn",
        "tagesschau",
        "orf",
        "derstandard",
        "heise",
    ]
    assert all(s.frontpage and not s.body for s in sources)
    assert {s.lang for s in sources} == {"de", "en"}


def test_sources_reject_unknown_keys_bad_lang_and_duplicates(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("- {name: a, lang: en, rss: https://a.test/rss, ignore_robots: true}\n")
    with pytest.raises(TypeError):
        collect.load_sources(p)
    p.write_text("- {name: a, lang: fr, rss: https://a.test/rss}\n")
    with pytest.raises(ValueError):
        collect.load_sources(p)
    p.write_text("- {name: a, lang: en, rss: https://a.test/rss}\n" * 2)
    with pytest.raises(ValueError):
        collect.load_sources(p)


# --- feeds ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,host,path,lang",
    [
        ("bbc", "feeds.bbci.co.uk", "/news/world/rss.xml", "en"),
        ("guardian", "www.theguardian.com", "/world/rss", "en"),
        ("hn", "news.ycombinator.com", "/rss", "en"),
        ("tagesschau", "www.tagesschau.de", "/index~rss2.xml", "de"),
        ("orf", "rss.orf.at", "/news.xml", "de"),
        ("derstandard", "www.derstandard.at", "/rss", "de"),
        ("heise", "www.heise.de", "/rss/heise-atom.xml", "de"),
    ],
)
def test_every_recorded_feed_parses(gate, web, name, host, path, lang):
    robots = FIXTURES / "robots" / f"{host}.txt"
    if robots.exists():
        web.get(f"https://{host}/robots.txt").respond(200, text=robots.read_text())
    else:
        web.get(f"https://{host}/robots.txt").respond(404)  # rss.orf.at has none
    web.get(f"https://{host}{path}").respond(200, content=feed(name))
    src = Source(name=name, lang=lang, rss=f"https://{host}{path}", frontpage=False)
    rows, report = collect.collect(src, gate, NOW)
    assert rows and report.fetched == len(rows)
    assert len({r["url_hash"] for r in rows}) == len(rows)
    for r in rows:
        assert r["title"] and r["url"].startswith("http")
        assert r["source"] == name and r["lang"] == lang and r["fetched_at"] == NOW
        assert r["published_at"] is None or r["published_at"].tzinfo is not None
        assert "at_medium" not in r["url"] and "utm_" not in r["url"]


def test_feed_skips_broken_entries(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/rss").respond(
        200,
        text=(
            "<rss><channel>"
            "<item><title>No link</title></item>"
            "<item><link>https://x.test/no-title</link></item>"
            "<item><title>Tom &amp; Jerry\n  return</title><link>https://x.test/a</link></item>"
            "<item><title>Twice</title><link>https://x.test/a#dup</link></item>"
            "</channel></rss>"
        ),
    )
    rows, _ = collect.collect(Source(name="x", lang="en", rss="https://x.test/rss"), gate, NOW)
    assert [r["title"] for r in rows] == ["Tom & Jerry return"]
    assert rows[0]["published_at"] is None


def test_feed_refused_by_robots(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text="User-agent: *\nDisallow: /\n")
    rows, report = collect.collect(Source(name="x", lang="en", rss="https://x.test/rss"), gate, NOW)
    assert rows == [] and report.refused == {"robots": 1}


def test_feed_server_error_is_reported_not_raised(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/rss").respond(500)
    rows, report = collect.collect(Source(name="x", lang="en", rss="https://x.test/rss"), gate, NOW)
    assert rows == [] and len(report.errors) == 1 and report.errors[0].startswith("feed:")


# --- front page -------------------------------------------------------------------------------


def test_hn_front_page_ranks(gate, web):
    serve_hn(web)
    rows, report = collect.collect(HN, gate, NOW)
    assert len(rows) == 30 and not report.alarms
    assert sorted(r["rank"] for r in rows) == list(range(1, 31))
    first = min(rows, key=lambda r: r["rank"])
    assert first["title"] == "Notes on a tiny text editor" and first["is_lead"] is True
    assert sum(r["is_lead"] for r in rows) == 1
    assert all(r["section"] is None for r in rows)


def test_front_page_refused_leaves_ranks_null(gate, web):
    web.get("https://www.theguardian.com/robots.txt").respond(
        200, text=robots_txt("www.theguardian.com")
    )
    web.get("https://www.theguardian.com/world/rss").respond(200, content=feed("guardian"))
    rows, report = collect.collect(GUARDIAN, gate, NOW)
    assert rows and report.refused == {"ai-agent": 1} and not report.alarms
    assert all(r["rank"] is None and r["is_lead"] is None for r in rows)


def test_front_page_switched_off_is_not_fetched(gate, web):
    serve_hn(web)
    rows, report = collect.collect(
        Source(name="hn", lang="en", rss=HN.rss, frontpage_url=HN.frontpage_url, frontpage=False),
        gate,
        NOW,
    )
    assert all(r["rank"] is None for r in rows)
    assert not any(c.request.url == "https://news.ycombinator.com/" for c in web.calls)


def test_selector_alarm_when_nothing_matches(gate, web):
    serve_hn(web, frontpage="<html><body><p>new design</p></body></html>")
    rows, report = collect.collect(HN, gate, NOW)
    assert rows and report.alarms == ["selector 'tr.athing' matched nothing"]


def test_alarm_when_allowed_but_no_selectors(gate, web):
    serve_hn(web)
    src = Source(name="hn", lang="en", rss=HN.rss, frontpage_url=HN.frontpage_url)
    _, report = collect.collect(src, gate, NOW)
    assert report.alarms == ["front page allowed but no selectors configured"]


def test_front_page_ranks_section_and_relative_links():
    html = (
        '<div class="s"><h2>World</h2><a class="h" href="/a">A</a></div>'
        '<div class="s"><a class="h" href="https://x.test/b?utm_source=f">B</a></div>'
        '<div class="s"><span>no link</span></div>'
    )
    ranks = collect.front_page_ranks(
        html, "https://x.test/", {"item": "div.s", "link": "a.h", "section": "h2"}
    )
    from sentiment.rows import url_hash

    assert ranks == {
        url_hash("https://x.test/a"): (1, "World"),
        url_hash("https://x.test/b"): (2, None),
    }


# --- article bodies ---------------------------------------------------------------------------

LONG = "Markets rallied strongly today as investors cheered the news. " * 5
SECRET = "SENTINEL-ARTICLE-TEXT"


def body_source(**kw):
    return Source(name="hn", lang="en", rss=HN.rss, frontpage=False, body=True, **kw)


def serve_articles(web, html: str):
    web.get(url__regex=r"https://(?!news\.ycombinator\.com)[^/]+/robots\.txt").respond(
        200, text=ALLOW_ALL
    )
    web.get(
        url__regex=r"https://(?!news\.ycombinator\.com)[^/]+/\.well-known/tdmrep\.json"
    ).respond(404)
    web.get(url__regex=r"https://(?!news\.ycombinator\.com)[^/]+/(?!robots\.txt).*").respond(
        200, text=html, headers={"content-type": "text/html"}
    )


def test_bodies_are_scored_and_never_stored(gate, web, monkeypatch):
    monkeypatch.setattr(collect, "MAX_BODIES", 2)
    serve_hn(web)
    serve_articles(web, f"<html><article><p>{LONG}</p><p>{SECRET}</p></article></html>")
    seen_text = []

    def score_body(text, lang):
        seen_text.append(text)
        return "positive"

    rows, report = collect.collect(body_source(), gate, NOW, score_body=score_body)
    assert [r["body_label"] for r in rows[:2]] == ["positive", "positive"]
    assert all(r["body_label"] is None for r in rows[2:])
    assert len(seen_text) == 2 and SECRET in seen_text[0]
    assert SECRET not in repr(rows) and SECRET not in repr(report.as_dict())


def test_bodies_only_for_new_rows(gate, web, monkeypatch):
    monkeypatch.setattr(collect, "MAX_BODIES", 2)
    serve_hn(web)
    serve_articles(web, f"<html><p>{LONG}</p></html>")
    first, _ = collect.collect(body_source(), gate, NOW, score_body=lambda t, lang: "neutral")
    seen = {r["url_hash"] for r in first[:2]}
    rows, _ = collect.collect(
        body_source(), gate, NOW, score_body=lambda t, lang: "neutral", seen=seen
    )
    assert [r["body_label"] for r in rows[:4]] == [None, None, "neutral", "neutral"]


def test_short_body_counts_as_paywall(gate, web, monkeypatch):
    monkeypatch.setattr(collect, "MAX_BODIES", 1)
    serve_hn(web)
    serve_articles(web, "<html><p>Subscribe to read.</p></html>")
    rows, _ = collect.collect(body_source(), gate, NOW, score_body=lambda t, lang: "negative")
    assert rows[0]["body_label"] is None


def test_body_odd_responses(gate, web, monkeypatch):
    monkeypatch.setattr(collect, "MAX_BODIES", 3)
    serve_hn(web)
    rows_probe, _ = collect.collect(
        Source(name="hn", lang="en", rss=HN.rss, frontpage=False), gate, NOW
    )
    first3 = [r["url"] for r in rows_probe[:3]]
    for url, response in zip(
        first3,
        [
            httpx.Response(404),
            httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}),
            httpx.Response(301, headers={"location": "https://refuses.test/x"}),
        ],
        strict=True,
    ):
        host = httpx.URL(url).host
        web.get(f"https://{host}/robots.txt").respond(200, text=ALLOW_ALL)
        web.get(f"https://{host}/.well-known/tdmrep.json").respond(404)
        web.get(url).mock(return_value=response)
    web.get("https://refuses.test/robots.txt").respond(200, text="User-agent: CCBot\nDisallow: /\n")
    rows, report = collect.collect(body_source(), gate, NOW, score_body=lambda t, lang: "positive")
    assert [r["body_label"] for r in rows[:3]] == [None, None, None]
    assert report.refused == {"ai-agent": 1}


def test_body_switch_needs_a_scorer(gate, web):
    serve_hn(web)
    rows, _ = collect.collect(body_source(), gate, NOW, score_body=None)
    assert all(r["body_label"] is None for r in rows)
    assert all("ycombinator" in str(c.request.url) for c in web.calls)


def test_body_text_prefers_selector_then_article_then_paragraphs():
    html = "<div class='x'>Selected</div><article><p>In article</p></article><p>Loose</p>"
    assert collect.body_text(html, "div.x") == "Selected"
    assert collect.body_text(html, None) == "In article"
    assert collect.body_text("<p>Loose</p><p>two</p>", None) == "Loose two"
    assert collect.body_text("<div>none</div>", None) == ""


def test_report_as_dict():
    r = collect.Report("hn", fetched=3, new=1)
    r.refused["ai-agent"] += 1
    assert r.as_dict() == {
        "source": "hn",
        "fetched": 3,
        "new": 1,
        "refused": {"ai-agent": 1},
        "alarms": [],
        "errors": [],
    }


def test_gate_refusal_type_is_the_gates():
    assert collect.Refused is polite.Refused


def test_body_time_budget_stops_scraping(gate, web, monkeypatch):
    monkeypatch.setattr(collect, "BODY_BUDGET", 20.0)  # each article costs >= 10 s of waits
    serve_hn(web)
    serve_articles(web, f"<html><p>{LONG}</p></html>")
    rows, report = collect.collect(body_source(), gate, NOW, score_body=lambda t, lang: "neutral")
    scored = sum(r["body_label"] is not None for r in rows)
    assert 1 <= scored < 30
    assert report.errors == [f"body: time budget spent, {30 - scored} skipped"]
