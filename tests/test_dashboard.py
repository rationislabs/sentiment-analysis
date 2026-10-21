from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from sentiment import dashboard, store
from sentiment.rows import new_row

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


def headlines(specs):
    """specs: (source, lang, day, p_neg, p_pos) with None probs for unscored rows."""
    rows = []
    for i, (source, lang, day, neg, pos) in enumerate(specs):
        r = new_row(
            url=f"https://x.test/{i}",
            title=f"title {i}",
            source=source,
            lang=lang,
            published_at=None,
            fetched_at=datetime(2026, 10, day, 9, tzinfo=UTC),
        )
        if neg is not None:
            r.update(
                model="tfidf",
                label="neutral",
                p_negative=neg,
                p_neutral=1 - neg - pos,
                p_positive=pos,
            )
        rows.append(r)
    return store.frame(rows)


def test_trends_mean_polarity_per_day_and_source():
    df = headlines(
        [
            ("hn", "en", 1, 0.1, 0.5),
            ("hn", "en", 1, 0.3, 0.3),
            ("orf", "de", 2, 0.6, 0.2),
            ("hn", "en", 2, None, None),
        ]
    )
    t = dashboard.trends(df)
    assert list(t.columns) == ["hn", "orf"]
    assert t.loc[pd.Timestamp("2026-10-01", tz="UTC"), "hn"] == pytest.approx(0.2)
    assert t.loc[pd.Timestamp("2026-10-02", tz="UTC"), "orf"] == pytest.approx(-0.4)
    assert list(dashboard.trends(df, "de").columns) == ["orf"]
    assert dashboard.trends(headlines([("hn", "en", 1, None, None)])).empty


def test_headline_table_search_and_order():
    df = headlines([("hn", "en", 1, 0.1, 0.5), ("orf", "de", 3, None, None)])
    t = dashboard.headline_table(df)
    assert list(t.columns) == ["title", "source", "lang", "label", "published_at", "rank", "url"]
    assert t["title"].tolist() == ["title 1", "title 0"]
    assert dashboard.headline_table(df, "TITLE 0")["title"].tolist() == ["title 0"]
    assert dashboard.headline_table(df, "[unclosed").empty  # plain text, not a regex


def test_status_table():
    t = dashboard.status_table(
        {
            "run_at": "x",
            "sources": [
                {
                    "source": "guardian",
                    "fetched": 45,
                    "new": 3,
                    "refused": {"ai-agent": 1},
                    "alarms": [],
                    "errors": [],
                }
            ],
        }
    )
    assert t.to_dict("records") == [
        {
            "Source": "guardian",
            "Fetched": 45,
            "New": 3,
            "Refused": "ai-agent: 1",
            "Alarms": "",
            "Errors": 0,
        }
    ]


def test_read_metrics_missing(tmp_path):
    assert dashboard.read_metrics(tmp_path / "metrics.json") is None


def test_scores_table():
    t = dashboard.scores_table(
        [{"model": "tfidf", "n": 200, "macro_f1": 0.51234, "ci": [0.45, 0.57]}]
    )
    assert t.to_dict("records") == [
        {"Model": "tfidf", "n": 200, "Macro-F1": 0.512, "95 % CI": "0.450–0.570"}
    ]


def empty_gold():
    return pd.DataFrame(columns=list(store.GOLD_DTYPES)).astype(store.GOLD_DTYPES)


def test_next_to_label_balances_language_then_source():
    df = headlines(
        [("hn", "en", 1, None, None), ("bbc", "en", 1, None, None), ("orf", "de", 1, None, None)]
    )
    gold = empty_gold()
    seen = []
    for _ in range(3):
        row = dashboard.next_to_label(df, gold)
        seen.append(row["source"])
        gold = dashboard.add_label(gold, row, "neutral", NOW)
    assert seen[0] == "orf"  # tie on languages: de before en; then sources in name order
    assert set(seen) == {"orf", "bbc", "hn"}
    assert dashboard.next_to_label(df, gold) is None


def test_next_to_label_with_no_headlines():
    assert dashboard.next_to_label(store.frame([]), empty_gold()) is None


def test_add_label_sets_split_and_rejects_unknown():
    df = headlines([("hn", "en", 1, None, None)])
    gold = dashboard.add_label(empty_gold(), df.iloc[0], "positive", NOW)
    assert gold.iloc[0]["split"] == store.split_of(df.iloc[0]["url_hash"])
    assert gold.iloc[0]["label"] == "positive" and pd.isna(gold.iloc[0]["label2"])
    with pytest.raises(ValueError):
        dashboard.add_label(empty_gold(), df.iloc[0], "mixed", NOW)


def test_relabel_queue_waits_a_week_and_stops_at_50():
    df = headlines([("hn", "en", 1, None, None)] * 60)
    gold = empty_gold()
    for i in range(60):
        gold = dashboard.add_label(
            gold, df.iloc[i], "neutral", NOW - timedelta(days=8 if i < 55 else 1)
        )
    q = dashboard.relabel_queue(gold, NOW)
    assert len(q) == 50 and (q["labelled_at"] <= NOW - timedelta(days=7)).all()
    for h in q["url_hash"][:50]:
        gold = dashboard.add_relabel(gold, h, "neutral", NOW)
    assert dashboard.relabel_queue(gold, NOW).empty
    assert gold["label2"].notna().sum() == 50
