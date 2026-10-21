"""The politeness gate. Every HTTP request goes through Gate.fetch, and nothing here can be
switched off.

- Feeds need robots.txt to allow our User-Agent.
- Scrapers (front pages, article bodies) additionally need: no named AI/TDM crawler blocked
  for that URL, no rights-reservation text in robots.txt, and no TDMRep reservation
  (/.well-known/tdmrep.json, the tdm-reservation header or meta tag).
- At most one request per MIN_INTERVAL per host; a longer Crawl-delay wins.

robots.txt follows RFC 9309: 4xx means no rules (allow all); 5xx or a network error means
"unreachable" (refuse all)."""

from __future__ import annotations

import fnmatch
import re
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from protego import Protego

USER_AGENT = "RationisLabsSentimentBot/1.0 (+https://github.com/rationislabs/sentiment-analysis)"
AGENT = "RationisLabsSentimentBot"
AI_AGENTS = (
    "GPTBot",
    "ChatGPT-User",
    "ClaudeBot",
    "anthropic-ai",
    "CCBot",
    "Google-Extended",
    "Applebot-Extended",
    "PerplexityBot",
    "Bytespider",
    "meta-externalagent",
)
RESERVATION_PHRASES = (
    "text and data mining",
    "text-and-data-mining",
    "tdm",
    "article 4",
    "art. 4",
    "reservation of rights",
    "rights reserved",
    "nutzungsvorbehalt",
    "vorbehalt",
    "text- und data-mining",
)
MIN_INTERVAL = 5.0
MAX_CRAWL_DELAY = 60.0  # a site asking for more is refused, not waited for: a run has 10 minutes
MAX_REDIRECTS = 3
Purpose = Literal["feed", "scrape"]

_PHRASE = re.compile("|".join(rf"(?<!\w){re.escape(p)}(?!\w)" for p in RESERVATION_PHRASES))


class Refused(Exception):
    """The site said no, or could not be asked.
    reason: robots | crawl-delay | ai-agent | reservation-text | tdmrep | unreachable"""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{reason}: {url}")
        self.url = url
        self.reason = reason


def reservation_text(robots_txt: str) -> bool:
    """A rights-reservation phrase in any comment of a robots.txt."""
    return any(
        _PHRASE.search(line.split("#", 1)[1].lower())
        for line in robots_txt.splitlines()
        if "#" in line
    )


def tdmrep_reserves(rules: list[dict], path: str) -> bool:
    """The first rule whose location matches decides; `*` matches anything."""
    for rule in rules:
        if fnmatch.fnmatchcase(path, str(rule.get("location", ""))):
            return str(rule.get("tdm-reservation")) == "1"
    return False


def reserved_in_response(r: httpx.Response) -> bool:
    if r.headers.get("tdm-reservation", "").strip() == "1":
        return True
    if "html" not in r.headers.get("content-type", ""):
        return False
    meta = BeautifulSoup(r.text, "lxml").find("meta", attrs={"name": "tdm-reservation"})
    return meta is not None and str(meta.get("content", "")).strip() == "1"


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20)


@dataclass
class _Site:
    robots: Protego | None  # None: robots.txt unreachable, so everything is refused
    reserved_text: bool
    tdmrep: list[dict] | None = None  # fetched on the first scrape request only
    text: str = ""  # the robots.txt as fetched (empty when there was none)


class Gate:
    def __init__(
        self,
        client: httpx.Client,
        *,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self._client = client
        self._clock = clock
        self._sleep = sleep
        self._sites: dict[str, _Site] = {}
        self._last: dict[str, float] = {}

    def fetch(self, url: str, purpose: Purpose) -> httpx.Response:
        """GET url if the site allows it; every redirect target is checked again."""
        for _ in range(MAX_REDIRECTS + 1):
            self.check(url, purpose)
            r = self._get(url)
            if not r.is_redirect:
                if purpose == "scrape" and reserved_in_response(r):
                    raise Refused(url, "tdmrep")
                return r
            url = urljoin(url, r.headers["location"])
        raise httpx.HTTPError(f"more than {MAX_REDIRECTS} redirects: {url}")

    def now(self) -> float:
        return self._clock()

    def robots_text(self, url: str) -> str:
        """The site's robots.txt as the gate fetched it (one request per host, rate-limited)."""
        return self._site(url).text

    def check(self, url: str, purpose: Purpose) -> None:
        site = self._site(url)
        if site.robots is None:
            raise Refused(url, "unreachable")
        if not site.robots.can_fetch(url, AGENT):
            raise Refused(url, "robots")
        if (site.robots.crawl_delay(AGENT) or 0) > MAX_CRAWL_DELAY:
            raise Refused(url, "crawl-delay")
        if purpose == "feed":
            return
        if any(not site.robots.can_fetch(url, agent) for agent in AI_AGENTS):
            raise Refused(url, "ai-agent")
        if site.reserved_text:
            raise Refused(url, "reservation-text")
        if site.tdmrep is None:
            site.tdmrep = self._tdmrep(url)
        if tdmrep_reserves(site.tdmrep, urlsplit(url).path or "/"):
            raise Refused(url, "tdmrep")

    def _site(self, url: str) -> _Site:
        s = urlsplit(url)
        if s.netloc not in self._sites:
            self._sites[s.netloc] = self._load_robots(f"{s.scheme}://{s.netloc}/robots.txt")
        return self._sites[s.netloc]

    def _load_robots(self, robots_url: str) -> _Site:
        try:
            r = self._get(robots_url, follow=True)
        except httpx.HTTPError:
            return _Site(None, False)
        if r.status_code >= 500:
            return _Site(None, False)
        if r.status_code >= 400:
            return _Site(Protego.parse(""), False)
        text = r.content.decode("utf-8-sig", errors="replace")  # a BOM would hide the first group
        return _Site(Protego.parse(text), reservation_text(text), text=text)

    def _tdmrep(self, url: str) -> list[dict]:
        s = urlsplit(url)
        try:
            r = self._get(f"{s.scheme}://{s.netloc}/.well-known/tdmrep.json")
            rules = r.json() if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return []
        return [x for x in rules if isinstance(x, dict)] if isinstance(rules, list) else []

    def _get(self, url: str, follow: bool = False) -> httpx.Response:
        host = urlsplit(url).netloc
        site = self._sites.get(host)
        crawl_delay = (site.robots.crawl_delay(AGENT) or 0) if site and site.robots else 0
        last = self._last.get(host)
        if last is not None and (wait := last + max(MIN_INTERVAL, crawl_delay) - self._clock()) > 0:
            self._sleep(wait)
        try:
            return self._client.get(url, follow_redirects=follow)
        finally:
            self._last[host] = self._clock()
