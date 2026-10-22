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


def test_tdmrep_location_is_a_prefix_like_robots_txt():
    assert polite.tdmrep_reserves([{"location": "/", "tdm-reservation": 1}], "/world/article")
    assert polite.tdmrep_reserves([{"location": "/news", "tdm-reservation": 1}], "/news/a")
    assert not polite.tdmrep_reserves([{"location": "/news", "tdm-reservation": 1}], "/sport/a")
    exact = [{"location": "/page$", "tdm-reservation": 1}]
    assert polite.tdmrep_reserves(exact, "/page")
    assert not polite.tdmrep_reserves(exact, "/page/2")


def test_tdmrep_file_behind_a_redirect_still_counts(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/.well-known/tdmrep.json").respond(
        301, headers={"location": "https://x.test/tdm/policy.json"}
    )
    web.get("https://x.test/tdm/policy.json").respond(
        200, json=[{"location": "/", "tdm-reservation": 1}]
    )
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/news/a", "scrape")
    assert e.value.reason == "tdmrep"


@pytest.mark.parametrize("failure", ["503", "connect"])
def test_tdmrep_unreachable_refuses_scraping(gate, web, failure):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    route = web.get("https://x.test/.well-known/tdmrep.json")
    if failure == "503":
        route.respond(503)
    else:
        route.mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/news/a", "scrape")
    assert e.value.reason == "unreachable"


def test_an_unreachable_tdmrep_file_is_asked_once_per_host(gate, web):
    # Every article of a site whose tdmrep.json is down cost one more rate-limited request.
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    route = web.get("https://x.test/.well-known/tdmrep.json").respond(503)
    for path in ("/a", "/b"):
        with pytest.raises(polite.Refused) as e:
            gate.check(f"https://x.test{path}", "scrape")
        assert e.value.reason == "unreachable"
    assert route.call_count == 1


def test_tdmrep_location_matches_the_query_too(gate, web):
    # Like robots.txt, a location is matched against the path and the query.
    rules = [{"location": "/search?q=*", "tdm-reservation": 1}]
    assert polite.tdmrep_reserves(rules, "/search?q=x")
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/.well-known/tdmrep.json").respond(200, json=rules)
    gate.check("https://x.test/search", "scrape")
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/search?q=1", "scrape")
    assert e.value.reason == "tdmrep"


def test_an_empty_tdmrep_location_matches_nothing():
    # Like an empty robots.txt pattern: it must not reserve the whole site.
    assert not polite.tdmrep_reserves([{"location": "", "tdm-reservation": 1}], "/news/a")


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


# --- limits -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://169.254.169.254/latest/meta-data", "http://127.0.0.1:8080/", "ftp://x.test/a"]
)
def test_private_or_non_web_addresses_are_never_fetched(gate, web, url):
    route = web.route().respond(200)
    with pytest.raises(polite.Refused) as e:
        gate.fetch(url, "scrape")
    assert e.value.reason == "private" and not route.called


def test_a_name_that_resolves_to_a_private_address_is_refused(gate, web, monkeypatch):
    monkeypatch.setattr(polite, "_resolve", lambda host: ["10.0.0.5"])
    route = web.route().respond(200)
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://intranet.x.test/a", "feed")
    assert e.value.reason == "private" and not route.called


def test_a_redirect_to_a_private_address_is_refused(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/a").respond(302, headers={"location": "http://127.0.0.1/admin"})
    inner = web.get("http://127.0.0.1/admin").respond(200)
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://x.test/a", "feed")
    assert e.value.reason == "private" and not inner.called


def test_a_huge_response_is_cut_off(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/big").respond(200, content=b"x" * (polite.MAX_BYTES + 1))
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://x.test/big", "feed")
    assert e.value.reason == "too-large"


def test_a_normal_response_still_reads_whole(gate, web):
    web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get("https://x.test/ok").respond(200, text="hello", headers={"x-a": "1"})
    r = gate.fetch("https://x.test/ok", "feed")
    assert r.text == "hello" and r.headers["x-a"] == "1"


@pytest.mark.parametrize("path", ["/robots.txt", "/.well-known/tdmrep.json"])
def test_redirects_httpx_follows_itself_are_address_checked_too(gate, web, path):
    # robots.txt and tdmrep.json follow redirects inside httpx, not in Gate.fetch.
    if path != "/robots.txt":
        web.get("https://x.test/robots.txt").respond(200, text=ALLOW_ALL)
    web.get(f"https://x.test{path}").respond(302, headers={"location": "http://169.254.169.254/x"})
    inner = web.get("http://169.254.169.254/x").respond(200, text=ALLOW_ALL)
    with pytest.raises(polite.Refused) as e:
        gate.check("https://x.test/news/a", "scrape")
    assert e.value.reason in ("private", "unreachable") and not inner.called


def test_waits_that_would_pass_the_run_deadline_are_not_taken(gate, web, clock):
    # A 60 s crawl-delay near the end of the run would push it past the job's limit.
    web.get("https://x.test/robots.txt").respond(200, text="User-agent: *\nCrawl-delay: 60\n")
    web.get("https://x.test/a").respond(200)
    gate.stop_at = clock() + 30
    with pytest.raises(polite.Refused) as e:
        gate.fetch("https://x.test/a", "feed")  # robots now, then a 60 s wait for /a
    assert e.value.reason == "deadline"
    assert clock() <= gate.stop_at


def test_a_tdmrep_rule_without_a_location_matches_nothing():
    rules = [{"tdm-reservation": 0}, {"location": "/", "tdm-reservation": 1}]
    assert polite.tdmrep_reserves(rules, "/news/a")
