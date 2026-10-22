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

import ipaddress
import math
import re
import socket
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
MAX_BYTES = 5_000_000  # one response; a front page is a few hundred KB
TOTAL_TIMEOUT = 30.0  # seconds for one whole response (httpx's timeout is per read)
Purpose = Literal["feed", "scrape"]

_PHRASE = re.compile("|".join(rf"(?<!\w){re.escape(p)}(?!\w)" for p in RESERVATION_PHRASES))


class Refused(Exception):
    """The site said no, or could not be asked.
    reason: robots | crawl-delay | ai-agent | reservation-text | tdmrep | unreachable |
    private | too-large | too-slow | deadline"""

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


def _location_matches(pattern: str, path: str) -> bool:
    """robots.txt-style: a prefix, `*` matches any run of characters, `$` ends the path."""
    anchored = pattern.endswith("$")
    body = re.escape(pattern[:-1] if anchored else pattern).replace(r"\*", ".*")
    return re.match(body + ("$" if anchored else ""), path) is not None


def tdmrep_reserves(rules: list[dict], path: str) -> bool:
    """The first rule whose location matches decides. `path` includes the query, as in
    robots.txt."""
    for rule in rules:
        # an empty location matches nothing, like an empty robots.txt pattern
        if rule.get("location") and _location_matches(str(rule["location"]), path):
            return str(rule.get("tdm-reservation")) == "1"
    return False


def reserved_in_response(r: httpx.Response) -> bool:
    if r.headers.get("tdm-reservation", "").strip() == "1":
        return True
    if "html" not in r.headers.get("content-type", ""):
        return False
    meta = BeautifulSoup(r.text, "lxml").find("meta", attrs={"name": "tdm-reservation"})
    return meta is not None and str(meta.get("content", "")).strip() == "1"


def _resolve(host: str) -> list[str]:
    return [a[4][0] for a in socket.getaddrinfo(host, None)]


def public_http_url(url: str) -> bool:
    """http(s), and every address the host resolves to is a public one: a feed link to
    169.254.169.254 or a redirect to 127.0.0.1 is never fetched."""
    s = urlsplit(url)
    if s.scheme not in ("http", "https") or not s.hostname:
        return False
    try:
        addrs = [s.hostname] if _is_ip(s.hostname) else _resolve(s.hostname)
    except OSError:
        return True  # unresolvable: the request itself fails; nothing private is reached
    return bool(addrs) and all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _public_only(request: httpx.Request) -> None:
    """Every request, redirect hops httpx follows itself included (robots.txt, tdmrep.json)."""
    if not public_http_url(str(request.url)):
        raise Refused(str(request.url), "private")


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=20, event_hooks={"request": [_public_only]}
    )


@dataclass
class _Site:
    robots: Protego | None  # None: robots.txt unreachable, so everything is refused
    reserved_text: bool
    tdmrep: list[dict] | None = None  # fetched on the first scrape request only
    tdmrep_refused: str = ""  # why the TDMRep file could not be used; asked once per host
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
        # A clock() value the run must end by: a wait that would pass it is not taken. Not a
        # constructor option: it skips no rule, it only stops early.
        self.stop_at = math.inf

    def fetch(self, url: str, purpose: Purpose) -> httpx.Response:
        """GET url if the site allows it; every redirect target is checked again."""
        for _ in range(MAX_REDIRECTS + 1):
            if not public_http_url(url):
                raise Refused(url, "private")
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
        if site.tdmrep_refused:
            raise Refused(url, site.tdmrep_refused)
        if site.tdmrep is None:
            try:
                site.tdmrep = self._tdmrep(url)
            except Refused as e:
                site.tdmrep_refused = e.reason
                raise
        s = urlsplit(url)
        if tdmrep_reserves(site.tdmrep, (s.path or "/") + (f"?{s.query}" if s.query else "")):
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
        """The site's TDMRep rules. A file we cannot read refuses, like robots.txt: a 5xx or a
        network error may be hiding a reservation. Only "no such file" (4xx) means none."""
        s = urlsplit(url)
        try:
            r = self._get(f"{s.scheme}://{s.netloc}/.well-known/tdmrep.json", follow=True)
        except httpx.HTTPError as e:
            raise Refused(url, "unreachable") from e
        if r.status_code >= 500:
            raise Refused(url, "unreachable")
        if r.status_code != 200:
            return []
        try:
            rules = r.json()
        except ValueError:
            return []
        return [x for x in rules if isinstance(x, dict)] if isinstance(rules, list) else []

    def _get(self, url: str, follow: bool = False) -> httpx.Response:
        host = urlsplit(url).netloc
        site = self._sites.get(host)
        crawl_delay = (site.robots.crawl_delay(AGENT) or 0) if site and site.robots else 0
        last = self._last.get(host)
        wait = last + max(MIN_INTERVAL, crawl_delay) - self._clock() if last is not None else 0
        if self._clock() + max(wait, 0) > self.stop_at:
            raise Refused(url, "deadline")
        if wait > 0:
            self._sleep(wait)
        try:
            return self._read(url, follow)
        finally:
            self._last[host] = self._clock()

    def _read(self, url: str, follow: bool) -> httpx.Response:
        """GET with a byte cap and a deadline for the whole body, not just each read."""
        start = time.monotonic()
        with self._client.stream("GET", url, follow_redirects=follow) as r:
            body = bytearray()
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise Refused(url, "too-large")
                if time.monotonic() - start > TOTAL_TIMEOUT:
                    raise Refused(url, "too-slow")
            # iter_bytes decoded the body: the encoding headers no longer describe it.
            headers = [
                (k, v)
                for k, v in r.headers.multi_items()
                if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
            ]
            return httpx.Response(
                r.status_code, headers=headers, content=bytes(body), request=r.request
            )
