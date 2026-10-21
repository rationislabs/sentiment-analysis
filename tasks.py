"""Developer tasks. List them with `uv run inv --list`."""

from __future__ import annotations

from pathlib import Path

from invoke import task

ROOT = Path(__file__).parent


@task
def test(c):
    c.run("uv run pytest", pty=True)


@task
def lint(c):
    c.run("uv run ruff check . && uv run ruff format --check .")


@task
def fixtures(c):
    """Re-record tests/fixtures/robots through the politeness gate. Feeds and front pages are
    invented (they would republish the publishers' content), so they are never recorded."""
    from sentiment import polite
    from sentiment.collect import load_sources

    with polite.client() as client:
        record_fixtures(
            polite.Gate(client), load_sources(ROOT / "sources.yaml"), ROOT / "tests" / "fixtures"
        )


def record_fixtures(gate, sources, fx: Path) -> None:
    """Every robots.txt goes through the gate: once per host, rate-limited."""
    (fx / "robots").mkdir(parents=True, exist_ok=True)
    hosts = {url.split("/")[2]: url for s in sources for url in (s.rss, s.frontpage_url) if url}
    for host, url in hosts.items():
        if text := gate.robots_text(url):
            (fx / "robots" / f"{host}.txt").write_text(text)


def _dotenv() -> dict[str, str]:
    """os.environ plus .env, without printing it."""
    import os

    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


@task
def collect(c, local=False):
    """One collection run. --local: into data/ only, the Hub untouched, scored with the models
    `inv train` wrote to models/ (unless SENTIMENT_MODEL_DIR says otherwise)."""
    env = _dotenv()
    if local:
        env["SENTIMENT_DATA_DIR"] = str(ROOT / "data")
        env["SENTIMENT_MODEL_DIR"] = env.get("SENTIMENT_MODEL_DIR") or str(ROOT / "models")
    c.run("uv run python -m sentiment collect", env=env, pty=True)


@task(name="hub-init")
def hub_init(c):
    """Create the private dataset and model repos on the Hub (HF_TOKEN with write access)."""
    from sentiment import config, hfsync

    hfsync.init_repos(config.load(_dotenv()))
    print("Both repos exist and are private.")


@task(name="train")
def train_models(c, limit=20000, epochs=3, skip_bert=False):
    """Download the public datasets and train into models/ (DistilBERT: about an hour on CPU)."""
    flags = f"--limit {limit} --epochs {epochs}" + (" --skip-bert" if skip_bert else "")
    c.run(f"uv run --group train --group bert python -m train fit {flags}", env=_dotenv(), pty=True)


@task
def evaluate(c):
    """Score models/ on the gold set: metrics.json, model.yaml, models/README.md."""
    c.run("uv run --group bert python -m train evaluate", env=_dotenv(), pty=True)


@task(name="push-model")
def push_model(c):
    """Upload models/ to the private model repo."""
    c.run("uv run python -m train push", env=_dotenv(), pty=True)


@task
def app(c, local=False):
    """The dashboard with the Label tab on http://127.0.0.1:8501. --local: data/ only."""
    env = {**_dotenv(), "LABEL_MODE": "1"}
    if local:
        env["SENTIMENT_DATA_DIR"] = str(ROOT / "data")
    c.run(
        "uv run streamlit run app/main.py --server.address 127.0.0.1 --server.port 8501",
        env=env,
        pty=True,
    )
