import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from sentiment import store
from sentiment.rows import new_row

APP = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    """st.cache_data lives in this process: without this, one test sees another's data."""
    st.cache_data.clear()


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    rows = []
    for i, (source, lang) in enumerate([("hn", "en"), ("orf", "de"), ("hn", "en")]):
        r = new_row(
            url=f"https://x.test/{i}",
            title=f"Headline number {i}",
            source=source,
            lang=lang,
            published_at=None,
            fetched_at=datetime(2026, 10, 8, 9 + i, tzinfo=UTC),
        )
        r.update(model="tfidf", label="neutral", p_negative=0.2, p_neutral=0.6, p_positive=0.2)
        rows.append(r)
    df = store.frame(rows)
    store.write_headlines(root, df, set(store.months(df)))
    store.write_status(
        root,
        {
            "run_at": "2026-10-08T14:00:00+00:00",
            "sources": [
                {
                    "source": "guardian",
                    "fetched": 45,
                    "new": 0,
                    "refused": {"ai-agent": 1},
                    "alarms": [],
                    "errors": [],
                }
            ],
        },
    )
    monkeypatch.setenv("SENTIMENT_DATA_DIR", str(root))
    monkeypatch.setenv("SENTIMENT_METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.delenv("LABEL_MODE", raising=False)
    return root


def start():
    return AppTest.from_file(APP, default_timeout=30).run()


def texts(at):
    kinds = ("markdown", "info", "caption", "error", "subheader", "metric")
    return " ".join(str(e.value) for kind in kinds for e in getattr(at, kind))


def test_tabs_without_label_mode(data):
    at = start()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Trends", "Headlines", "Models"]
    assert any("ai-agent: 1" in d.value.to_csv() for d in at.dataframe)


def test_headline_search(data):
    at = start()
    at.text_input(key="q").input("number 1").run()
    tables = [d.value for d in at.dataframe if "title" in d.value.columns]
    assert tables and tables[0]["title"].tolist() == ["Headline number 1"]


def test_models_without_metrics(data):
    assert "No gold set yet" in texts(start())


def test_models_with_metrics(data, tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-11-01T10:00:00+00:00",
                "self_agreement": {"n": 50, "kappa": 0.71},
                "languages": {
                    "de": {"status": "no gold set yet"},
                    "en": {
                        "n_dev": 100,
                        "n_test": 200,
                        "majority_label": "neutral",
                        "winner": "tfidf",
                        "dev": [],
                        "test": [
                            {
                                "model": "tfidf",
                                "lang": "en",
                                "split": "test",
                                "n": 200,
                                "macro_f1": 0.5,
                                "ci": [0.45, 0.55],
                                "per_class": {},
                                "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            }
                        ],
                        "worst_errors": {"tfidf": []},
                        "disagreements": [],
                    },
                },
            }
        )
    )
    at = start()
    assert not at.exception
    assert "tfidf" in texts(at) and "0.71" in texts(at)


def test_label_tab_writes_gold(data, monkeypatch):
    monkeypatch.setenv("LABEL_MODE", "1")
    at = start()
    assert [t.label for t in at.tabs][-1] == "Label"
    at.button(key="label-positive").click().run()
    assert not at.exception
    gold = store.read_gold(data)
    assert len(gold) == 1 and gold.iloc[0]["label"] == "positive"


def test_label_tab_nothing_to_label(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("SENTIMENT_METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setenv("LABEL_MODE", "1")
    at = start()
    assert not at.exception
    assert "Nothing left to label" in texts(at)


def test_hub_unreachable_shows_message_and_retry(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTIMENT_DATA_DIR", raising=False)
    monkeypatch.delenv("LABEL_MODE", raising=False)
    monkeypatch.setenv("HF_DATASET_REPO", "me/ds")
    from sentiment import hfsync

    def down(*a, **k):
        raise OSError("hub down")

    monkeypatch.setattr(hfsync, "snapshot_download", down)
    at = start()
    assert not at.exception
    assert "Data unavailable" in texts(at)
    assert [b.label for b in at.button] == ["Retry"]


def test_hub_mode_offers_push_in_a_fresh_session(tmp_path, monkeypatch):
    # Labels saved in an earlier session must stay pushable after a browser refresh.
    monkeypatch.delenv("SENTIMENT_DATA_DIR", raising=False)
    monkeypatch.setenv("SENTIMENT_METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASET_REPO", "me/ds")
    monkeypatch.setenv("LABEL_MODE", "1")
    from sentiment import dashboard, hfsync

    def pull(repo_id, **kw):
        root = kw["local_dir"]
        root.mkdir(parents=True, exist_ok=True)
        r = new_row(
            url="https://x.test/1",
            title="Headline",
            source="hn",
            lang="en",
            published_at=None,
            fetched_at=datetime(2026, 10, 8, tzinfo=UTC),
        )
        df = store.frame([r])
        store.write_headlines(root, df, set(store.months(df)))
        store.write_gold(
            root,
            dashboard.add_label(
                store.read_gold(root), df.iloc[0], "neutral", datetime(2026, 10, 8, tzinfo=UTC)
            ),
        )
        return str(root)

    monkeypatch.setattr(hfsync, "snapshot_download", pull)
    at = start()
    assert not at.exception
    assert any(b.key == "push" for b in at.button)


def test_push_sends_every_label_and_clears_the_local_copy(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTIMENT_DATA_DIR", raising=False)
    monkeypatch.setenv("SENTIMENT_METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASET_REPO", "me/ds")
    monkeypatch.setenv("LABEL_MODE", "1")
    from sentiment import dashboard, hfsync

    def pull(repo_id, **kw):
        root = kw["local_dir"]
        root.mkdir(parents=True, exist_ok=True)
        r = new_row(
            url="https://x.test/1",
            title="Headline",
            source="hn",
            lang="en",
            published_at=None,
            fetched_at=datetime(2026, 10, 8, tzinfo=UTC),
        )
        df = store.frame([r])
        store.write_headlines(root, df, set(store.months(df)))
        if not (root / store.LOCAL_GOLD).exists():
            store.write_gold(
                root,
                dashboard.add_label(
                    store.read_gold(root), df.iloc[0], "neutral", datetime(2026, 10, 8, tzinfo=UTC)
                ),
            )
        return str(root)

    sent = []
    monkeypatch.setattr(hfsync, "snapshot_download", pull)
    monkeypatch.setattr(hfsync, "push", lambda repo, token, root, paths, msg: sent.append(paths))
    at = start()
    next(b for b in at.button if b.key == "push").click().run()
    root = tmp_path / "cache" / "sentiment-data"
    assert sent == [[store.GOLD]]
    assert len(store.read_gold(root)) == 1
    assert not (root / store.LOCAL_GOLD).exists()


def test_label_tab_shows_the_headline_as_plain_text(data, monkeypatch):
    # A title is not markdown: "![](https://tracker/x)" must not load a remote image.
    monkeypatch.setenv("LABEL_MODE", "1")
    at = start()
    assert any("Headline number" in str(e.value) for e in at.markdown)  # shown as escaped text
    from sentiment import dashboard

    out = dashboard.plain_markdown("![](https://tracker/x) *b* :red[x] $x$ <b> `c` # h")
    assert re.search(r"(?<!\\)[!\[\]()*:$<>`#]", out) is None, out


def test_a_label_saved_in_another_tab_survives_a_click_here(data, monkeypatch):
    monkeypatch.setenv("LABEL_MODE", "1")
    at = start()  # this tab loaded the gold set before the other tab saved
    other = store.read_gold(data)
    from sentiment import dashboard

    here = next(str(e.value) for e in at.markdown if "Headline number" in str(e.value))
    first = next(r for _, r in store.read_headlines(data).iterrows() if r["title"][-1] not in here)
    store.write_gold(data, dashboard.add_label(other, first, "negative", datetime.now(UTC)))
    at.button(key="label-positive").click().run()
    assert not at.exception
    assert len(store.read_gold(data)) == 2
