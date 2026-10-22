from datetime import UTC, datetime
from pathlib import Path

import pytest
import skops.io

from sentiment import config, run, score, store
from sentiment.__main__ import main
from sentiment.rows import new_row
from tests.conftest import FIXTURES, robots_txt
from tests.test_score import TOY

NOW = datetime(2026, 10, 8, 14, tzinfo=UTC)
LATER = datetime(2026, 10, 8, 18, tzinfo=UTC)
SOURCES = """\
- name: hn
  lang: en
  rss: https://news.ycombinator.com/rss
  frontpage_url: https://news.ycombinator.com/
  selectors: {item: tr.athing, link: span.titleline > a}
- name: guardian
  lang: en
  rss: https://www.theguardian.com/world/rss
  frontpage_url: https://www.theguardian.com/international
"""


@pytest.fixture
def site(web):
    web.get("https://news.ycombinator.com/robots.txt").respond(
        200, text=robots_txt("news.ycombinator.com")
    )
    web.get("https://news.ycombinator.com/.well-known/tdmrep.json").respond(404)
    web.get("https://news.ycombinator.com/rss").respond(
        200, content=(FIXTURES / "feeds" / "hn.xml").read_bytes()
    )
    page = web.get("https://news.ycombinator.com/").respond(
        200,
        text=(FIXTURES / "frontpages" / "hn.html").read_text(),
        headers={"content-type": "text/html"},
    )
    web.get("https://www.theguardian.com/robots.txt").respond(
        200, text=robots_txt("www.theguardian.com")
    )
    web.get("https://www.theguardian.com/world/rss").respond(
        200, content=(FIXTURES / "feeds" / "guardian.xml").read_bytes()
    )
    return page


@pytest.fixture
def files(tmp_path):
    sources = tmp_path / "sources.yaml"
    sources.write_text(SOURCES)
    choice = tmp_path / "model.yaml"
    choice.write_text("de: tfidf\nen: tfidf\n")
    models = tmp_path / "models"
    models.mkdir()
    return sources, choice, models


def toy_model(models: Path):
    p = score.Tfidf.build().fit([t for t, _ in TOY], [lab for _, lab in TOY])
    skops.io.dump(p, models / "tfidf-en.skops")


def go(settings, files, clock, now=NOW):
    sources, choice, _ = files
    return run.run(
        settings, sources_path=sources, choice_path=choice, now=now, clock=clock, sleep=clock.sleep
    )


def local(tmp_path, models=None):
    env = {"SENTIMENT_DATA_DIR": str(tmp_path / "data")}
    if models:
        env["SENTIMENT_MODEL_DIR"] = str(models)
    return config.load(env)


def test_local_run_collects_scores_and_reports(tmp_path, site, files, clock, capsys, monkeypatch):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    toy_model(files[2])
    assert go(local(tmp_path, files[2]), files, clock) == 0
    df = store.read_headlines(tmp_path / "data")
    assert len(df) == 30 + 45 - 1  # one Guardian story is also on HN: kept once, as hn
    assert df["label"].notna().all() and set(df["model"]) == {"tfidf"}
    hn = df[df["source"] == "hn"]
    assert hn["rank"].notna().all() and df[df["source"] == "guardian"]["rank"].isna().all()
    status = store.read_status(tmp_path / "data")
    assert status["run_at"] == NOW.isoformat()
    by = {s["source"]: s for s in status["sources"]}
    assert by["guardian"]["refused"] == {"ai-agent": 1} and by["hn"]["new"] == 30
    assert by["guardian"]["new"] == 44
    text = summary_file.read_text()
    assert "| guardian | 45 | 44 | ai-agent: 1 |" in text
    assert text in capsys.readouterr().out


def test_second_run_adds_nothing(tmp_path, site, files, clock):
    toy_model(files[2])
    s = local(tmp_path, files[2])
    go(s, files, clock)
    go(s, files, clock, now=LATER)
    assert len(store.read_headlines(tmp_path / "data")) == 74
    assert {x["new"] for x in store.read_status(tmp_path / "data")["sources"]} == {0}


def test_without_model_rows_stay_unscored_then_backfill(tmp_path, site, files, clock, monkeypatch):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    go(local(tmp_path), files, clock)
    df = store.read_headlines(tmp_path / "data")
    assert df["label"].isna().all()
    assert "Unscored headlines: 74" in summary_file.read_text()
    toy_model(files[2])
    go(local(tmp_path, files[2]), files, clock, now=LATER)
    assert store.read_headlines(tmp_path / "data")["label"].notna().all()


def test_backfill_rewrites_only_changed_months(tmp_path, site, files, clock):
    data = tmp_path / "data"
    unscored = store.frame(
        [
            new_row(
                url="https://old.test/a",
                title="great day",
                source="hn",
                lang="en",
                published_at=None,
                fetched_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]
    )
    labelled = store.frame(
        [
            new_row(
                url="https://old.test/b",
                title="x",
                source="hn",
                lang="en",
                published_at=None,
                fetched_at=datetime(2026, 7, 1, tzinfo=UTC),
            )
        ]
    )
    labelled.loc[0, ["model", "label"]] = ["vader", "neutral"]
    store.write_headlines(data, unscored, {"2026-08"})
    store.write_headlines(data, labelled, {"2026-07"})
    july = (data / "headlines" / "2026-07.parquet").stat().st_mtime_ns
    toy_model(files[2])
    go(local(tmp_path, files[2]), files, clock)
    assert (data / "headlines" / "2026-07.parquet").stat().st_mtime_ns == july
    df = store.read_headlines(data)
    assert df.loc[df["url"] == "https://old.test/a", "label"].item() == "positive"
    assert sorted(p.name for p in (data / "headlines").iterdir()) == [
        "2026-07.parquet",
        "2026-08.parquet",
        "2026-10.parquet",
    ]


def test_selector_alarm_exits_1_after_writing(tmp_path, site, files, clock):
    site.respond(
        200, text="<html><body>redesign</body></html>", headers={"content-type": "text/html"}
    )
    assert go(local(tmp_path), files, clock) == 1
    assert len(store.read_headlines(tmp_path / "data")) == 74


def test_one_broken_source_does_not_stop_the_others(tmp_path, site, files, clock, monkeypatch):
    real = run.collect

    def flaky(source, *a, **kw):
        if source.name == "hn":
            raise RuntimeError("boom")
        return real(source, *a, **kw)

    monkeypatch.setattr(run, "collect", flaky)
    assert go(local(tmp_path), files, clock) == 0
    status = store.read_status(tmp_path / "data")
    by = {s["source"]: s for s in status["sources"]}
    assert by["hn"]["errors"] == ["RuntimeError('boom')"] and by["guardian"]["new"] == 45


def test_hub_mode_pulls_and_pushes_changed_files(tmp_path, site, files, clock, monkeypatch):
    calls = {}

    def pull(repo, token, dest):
        calls["pull"] = (repo, token)
        return dest

    monkeypatch.setattr(run.hfsync, "pull", pull)
    monkeypatch.setattr(run.hfsync, "pull_models", lambda repo, token, revision: None)
    monkeypatch.setattr(
        run.hfsync,
        "push",
        lambda repo, token, root, paths, message: calls.update(
            push=(repo, token, sorted(paths), message)
        ),
    )
    s = config.load({"HF_TOKEN": "hf_fake", "HF_DATASET_REPO": "me/ds"})
    assert go(s, files, clock) == 0
    assert calls["pull"] == ("me/ds", "hf_fake")
    repo, token, paths, message = calls["push"]
    assert (repo, token) == ("me/ds", "hf_fake")
    assert paths == ["headlines/2026-10.parquet", "status.json"]
    assert message == "collect 2026-10-08 14:00 UTC: 74 new"


def test_hub_push_failure_fails_the_run(tmp_path, site, files, clock, monkeypatch):
    monkeypatch.setattr(run.hfsync, "pull", lambda repo, token, dest: dest)
    monkeypatch.setattr(run.hfsync, "pull_models", lambda repo, token, revision: None)

    def boom(*a, **k):
        raise OSError("hub down")

    monkeypatch.setattr(run.hfsync, "push", boom)
    with pytest.raises(OSError):
        go(config.load({"HF_TOKEN": "hf_fake"}), files, clock)


def test_summary_table():
    from sentiment.collect import Report

    a = Report("hn", fetched=30, new=2, alarms=["selector 'x' matched nothing"])
    b = Report("guardian", fetched=45, new=0, errors=["feed: boom"])
    b.refused["ai-agent"] += 1
    text = run.summary([a, b], unscored=3, missing=["de"])
    assert "| Source | Fetched | New | Refused | Alarms | Errors |" in text
    assert "| hn | 30 | 2 | — | selector 'x' matched nothing | — |" in text
    assert "| guardian | 45 | 0 | ai-agent: 1 | — | feed: boom |" in text
    assert "Unscored headlines: 3 (no model yet for: de)" in text


def test_main_usage_and_missing_token(monkeypatch, capsys):
    assert main(["nope"]) == 2
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("SENTIMENT_DATA_DIR", raising=False)
    assert main(["collect"]) == 2
    assert "HF_TOKEN is not set" in capsys.readouterr().err


def test_every_source_shares_one_run_deadline(tmp_path, files, clock, monkeypatch):
    seen, stops = [], []
    real = run.collect

    def spy(source, gate, now, score_body=None, seen_hashes=frozenset(), *, deadline):
        seen.append(deadline)
        stops.append(gate.stop_at)  # waits inside the gate stop at the deadline too
        return [], real.__globals__["Report"](source.name)

    monkeypatch.setattr(run, "collect", spy)
    go(local(tmp_path), files, clock)
    assert seen and len(set(seen)) == 1
    assert seen[0] == pytest.approx(clock() + run.RUN_BUDGET, abs=1)
    assert stops == seen
