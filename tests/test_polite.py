import httpx
import pytest

from sentiment import polite
from tests.conftest import robots_txt

ALLOW_ALL = "User-agent: *\nAllow: /\n"


def no_tdmrep(web, host):
    web.get(f"https://{host}/.well-known/tdmrep.json").respond(404)


# --- the real robots.txt files of 2026-10-08 -------------------------------------------------

REAL = [
    ("www.bbc.com", "/news"),
    ("www.theguardian.com", "/international"),
    ("www.tagesschau.de", "/"),
    ("orf.at", "/"),
    ("www.derstandard.at", "/"),
    ("www.heise.de", "/"),
]


@pytest.mark.parametrize("host,path", REAL)
def test_real_news_sites_refuse_the_scraper(gate, web, host, path):
    web.get(f"https://{host}/robots.txt").respond(200, text=robots_txt(host))
    with pytest.raises(polite.Refused) as e:
        gate.check(f"https://{host}{path}", "scrape")
    assert e.value.reason == "ai-agent"


def test_real_hacker_news_allows_the_scraper(gate, web):
    web.get("https://news.ycombinator.com/robots.txt").respond(
        200, text=robots_txt("news.ycombinator.com")
    )
    no_tdmrep(web, "news.ycombinator.com")
    gate.check("https://news.ycombinator.com/", "scrape")


@pytest.mark.parametrize(
    "host,feed",
    [
        ("feeds.bbci.co.uk", "/news/world/rss.xml"),
        ("www.theguardian.com", "/world/rss"),
        ("www.tagesschau.de", "/index~rss2.xml"),
        ("www.derstandard.at", "/rss"),
        ("www.heise.de", "/rss/heise-atom.xml"),
        ("news.ycombinator.com", "/rss"),
    ],
)
def test_real_feeds_are_allowed(gate, web, host, feed):
    web.get(f"https://{host}/robots.txt").respond(200, text=robots_txt(host))
    gate.check(f"https://{host}{feed}", "feed")


@pytest.mark.parametrize(
    "host,expected",
    [
        ("www.bbc.com", True),
        ("www.tagesschau.de", True),
        ("www.heise.de", True),
        ("www.theguardian.com", False),
        ("orf.at", False),
        ("www.derstandard.at", False),
        ("news.ycombinator.com", False),
        ("feeds.bbci.co.uk", False),
    ],
)
def test_reservation_text_in_real_files(host, expected):
    assert polite.reservation_text(robots_txt(host)) is expected


# --- rules, one at a time ---------------------------------------------------------------------


def test_robots_disallow_for_our_agent(gate, web):
    web.get("https://x.test/robots.txt").respond(
        200, text="User-agent: RationisLabsSentimentBot\nDisallow: /\n"
    )
    for purpose in ("feed", "scrape"):
        with pytest.raises(polite.Refused) as e:
            gate.check("https://x.test/rss", purpose)
        assert e.value.reason == "robots"


def test_longest_match_wins(gate, web):
    web.get("https://x.test/robots.txt").respond(
        200, text="User-agent: *\nAllow: /\nDisallow: /private\n"
    )
    gate.check("https://x.test/public", "feed")
    with pytest.raises(polite.Refused):
        gate.check("https://x.test/private/a", "feed")


def test_ai_block_refuses_scraper_but_not_feed(gate, web):
    web.get("https://x.test/robots.txt").respond(
        200, text="User-agent: CCBot\nDisallow: /\n\n" + ALLOW_ALL
    )
    gate.check("https://x.test/rss", "feed")
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/", "scrape")
    assert e.value.reason == "ai-agent"


def test_ai_allow_list_is_respected_per_path(gate, web):
    web.get("https://x.test/robots.txt").respond(
        200, text="User-agent: GPTBot\nAllow: /download/\nDisallow: /\n\n" + ALLOW_ALL
    )
    no_tdmrep(web, "x.test")
    gate.check("https://x.test/download/a", "scrape")
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/news/a", "scrape")
    assert e.value.reason == "ai-agent"


@pytest.mark.parametrize("phrase", polite.RESERVATION_PHRASES)
def test_each_reservation_phrase_refuses_scraper(gate, web, phrase):
    web.get("https://x.test/robots.txt").respond(
        200, text=f"# We object to {phrase.upper()} of our content.\n" + ALLOW_ALL
    )
    gate.check("https://x.test/rss", "feed")
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/", "scrape")
    assert e.value.reason == "reservation-text"


def test_phrases_only_count_in_comments_and_as_words():
    assert not polite.reservation_text("User-agent: *\nDisallow: /tdm/\n")
    assert not polite.reservation_text("# see our art. 42 policy\n")
    assert not polite.reservation_text("# tdmx is a tool\n")
    assert polite.reservation_text("Disallow: /a  # TDM reserved\n")


def test_robots_404_allows_everything(gate, web):
    web.get("https://x.test/robots.txt").respond(404)
    no_tdmrep(web, "x.test")
    gate.check("https://x.test/a", "scrape")


@pytest.mark.parametrize("status", [500, 503])
def test_robots_5xx_refuses_everything(gate, web, status):
    web.get("https://x.test/robots.txt").respond(status)
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/rss", "feed")
    assert e.value.reason == "unreachable"


def test_robots_network_error_refuses_everything(gate, web):
    web.get("https://x.test/robots.txt").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/rss", "feed")
    assert e.value.reason == "unreachable"


def test_robots_odd_bodies(gate, web):
    # Latin-1 bytes and an HTML error page served as 200: no crash; the HTML gives no rules.
    web.get("https://a.test/robots.txt").respond(
        200, content="# Gr\xfc\xdfe\nUser-agent: *\nDisallow: /x\n".encode("latin-1")
    )
    web.get("https://b.test/robots.txt").respond(
        200, text="<html><body>Not found</body></html>", headers={"content-type": "text/html"}
    )
    with pytest.raises(polite.Refused):
        gate.check("https://a.test/x/1", "feed")
    gate.check("https://b.test/rss", "feed")


def test_robots_fetched_once_per_host(gate, web):
    route = web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    gate.check("https://x.test/a", "feed")
    gate.check("https://x.test/b", "feed")
    assert route.call_count == 1


# --- TDMRep -----------------------------------------------------------------------------------


def test_tdmrep_file_reservation(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/.well-known/tdmrep.json").respond(
        200, json=[{"location": "/news/*", "tdm-reservation": 1}]
    )
    gate.check("https://x.test/rss", "feed")
    gate.check("https://x.test/about", "scrape")
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/news/a", "scrape")
    assert e.value.reason == "tdmrep"


def test_tdmrep_first_matching_rule_decides():
    rules = [
        {"location": "/open/*", "tdm-reservation": 0},
        {"location": "/*", "tdm-reservation": 1},
    ]
    assert not polite.tdmrep_reserves(rules, "/open/a")
    assert polite.tdmrep_reserves(rules, "/closed/a")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="<html>no</html>"),
        httpx.Response(200, json={"not": "a list"}),
    ],
)
def test_tdmrep_unusable_means_no_rules(gate, web, response):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/.well-known/tdmrep.json").mock(return_value=response)
    gate.check("https://x.test/a", "scrape")


def test_tdm_header_refuses_scrape(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    no_tdmrep(web, "x.test")
    web.get("https://x.test/a").respond(200, text="hi", headers={"tdm-reservation": "1"})
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://x.test/a", "scrape")
    assert e.value.reason == "tdmrep"


def test_tdm_meta_refuses_scrape(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    no_tdmrep(web, "x.test")
    web.get("https://x.test/a").respond(
        200,
        text='<html><head><meta name="tdm-reservation" content="1"></head></html>',
        headers={"content-type": "text/html; charset=utf-8"},
    )
    with pytest.raises(polite.Refused):
        gate.fetch("https://x.test/a", "scrape")


def test_tdm_header_does_not_block_feeds(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/rss").respond(200, text="<rss/>", headers={"tdm-reservation": "1"})
    assert gate.fetch("https://x.test/rss", "feed").status_code == 200


# --- redirects, rate limit, User-Agent --------------------------------------------------------


def test_redirect_is_checked_again(gate, web):
    web.get("https://a.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://b.test/robots.txt").respond(200, text="User-agent: *\nDisallow: /\n")
    web.get("https://a.test/rss").respond(301, headers={"location": "https://b.test/rss"})
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://a.test/rss", "feed")
    assert e.value.url == "https://b.test/rss"


def test_redirect_loop_gives_up(gate, web):
    web.get("https://a.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://a.test/loop").respond(302, headers={"location": "/loop"})
    with pytest.raises(httpx.HTTPError):
        gate.fetch("https://a.test/loop", "feed")


def test_min_interval_per_host(gate, web, clock):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/a").respond(200)
    web.get("https://y.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://y.test/a").respond(200)
    gate.fetch("https://x.test/a", "feed")  # robots, then a: one wait of 5 s
    gate.fetch("https://y.test/a", "feed")  # another host waited 5 s already: no wait for robots
    assert clock.slept == [5.0, 5.0]


def test_crawl_delay_longer_than_minimum_wins(gate, web, clock):
    web.get("https://x.test/robots.txt").respond(200, text="User-agent: *\nCrawl-delay: 30\n")
    web.get("https://x.test/a").respond(200)
    gate.fetch("https://x.test/a", "feed")
    gate.fetch("https://x.test/a", "feed")
    assert clock.slept == [30.0, 30.0]


def test_crawl_delay_shorter_than_minimum_is_ignored(gate, web, clock):
    web.get("https://x.test/robots.txt").respond(200, text="User-agent: *\nCrawl-delay: 1\n")
    web.get("https://x.test/a").respond(200)
    gate.fetch("https://x.test/a", "feed")
    assert clock.slept == [5.0]


def test_user_agent_header(gate, web):
    route = web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    gate.check("https://x.test/a", "feed")
    assert route.calls[0].request.headers["user-agent"] == (
        "RationisLabsSentimentBot/1.0 (+https://github.com/rationislabs/sentiment-analysis)"
    )


def test_gate_has_no_switch():
    # The gate takes no option that could skip a rule.
    import inspect

    params = set(inspect.signature(polite.Gate).parameters)
    assert params == {"client", "clock", "sleep"}


def test_robots_with_utf8_bom_keeps_first_group(gate, web):
    web.get("https://x.test/robots.txt").respond(
        200, content="﻿User-agent: *\nDisallow: /\n".encode()
    )
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/rss", "feed")
    assert e.value.reason == "robots"


def test_crawl_delay_over_cap_refuses_instead_of_sleeping(gate, web, clock):
    web.get("https://x.test/robots.txt").respond(200, text="User-agent: *\nCrawl-delay: 1800\n")
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://x.test/rss", "feed")
    assert e.value.reason == "crawl-delay" and clock.slept == []
