from pathlib import Path

import pytest
import respx

from sentiment import polite

FIXTURES = Path(__file__).parent / "fixtures"


class Clock:
    """A fake monotonic clock: sleeping advances it and is recorded, never real."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Test hosts (x.test, …) do not resolve: every name is a public address unless a test
    says otherwise."""
    monkeypatch.setattr(polite, "_resolve", lambda host: ["93.184.216.34"])


@pytest.fixture
def web():
    """Every HTTP request is mocked; an unmocked one fails the test."""
    with respx.mock(assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def gate(clock, web):
    with polite.client() as c:
        yield polite.Gate(c, clock=clock, sleep=clock.sleep)


def robots_txt(host: str) -> str:
    return (FIXTURES / "robots" / f"{host}.txt").read_text()
