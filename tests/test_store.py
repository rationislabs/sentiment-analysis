from datetime import UTC, datetime

import pandas as pd

from sentiment import hfsync, store
from sentiment.rows import COLUMNS, DTYPES, new_row

OCT = datetime(2026, 10, 8, 12, tzinfo=UTC)
SEP = datetime(2026, 9, 30, 23, tzinfo=UTC)


def row(url, when=OCT, source="hn", lang="en"):
    return new_row(
        url=url, title=f"T {url}", source=source, lang=lang, published_at=None, fetched_at=when
    )


def test_frame_has_schema_even_when_empty():
    df = store.frame([])
    assert tuple(df.columns) == COLUMNS
    assert {c: str(t) for c, t in df.dtypes.items()} == {
        c: str(pd.Series(dtype=t).dtype) for c, t in DTYPES.items()
    }


def test_merge_dedupes_against_existing_and_within_batch():
    existing = store.frame([row("https://a.test/1")])
    all_, fresh = store.merge(
        existing,
        [
            row("https://a.test/1"),
            row("https://a.test/2"),
            row("https://a.test/2#x"),
            row("https://a.test/3", source="guardian"),
        ],
    )
    assert list(fresh["url"]) == ["https://a.test/2", "https://a.test/3"]
    assert len(all_) == 3 and all_.index.is_unique


def test_same_url_from_two_sources_is_kept_once():
    _, fresh = store.merge(
        store.frame([]),
        [row("https://g.test/a", source="hn"), row("https://g.test/a", source="guardian")],
    )
    assert list(fresh["source"]) == ["hn"]


def test_write_and_read_monthly_partitions(tmp_path):
    df = store.frame([row("https://a.test/1"), row("https://a.test/2", when=SEP)])
    paths = store.write_headlines(tmp_path, df, set(store.months(df)))
    assert sorted(paths) == ["headlines/2026-09.parquet", "headlines/2026-10.parquet"]
    back = store.read_headlines(tmp_path)
    assert sorted(back["url"]) == ["https://a.test/1", "https://a.test/2"]
    assert dict(back.dtypes) == dict(df.dtypes)


def test_write_only_requested_months(tmp_path):
    df = store.frame([row("https://a.test/1"), row("https://a.test/2", when=SEP)])
    store.write_headlines(tmp_path, df, {"2026-09", "2026-10"})
    sep = tmp_path / "headlines" / "2026-09.parquet"
    before = sep.stat().st_mtime_ns
    df2, _ = store.merge(df, [row("https://a.test/3")])
    assert store.write_headlines(tmp_path, df2, {"2026-10"}) == ["headlines/2026-10.parquet"]
    assert sep.stat().st_mtime_ns == before
    assert len(store.read_headlines(tmp_path)) == 3


def test_read_headlines_empty_folder(tmp_path):
    assert store.read_headlines(tmp_path).empty


def test_gold_roundtrip_and_empty(tmp_path):
    assert store.read_gold(tmp_path).empty
    assert tuple(store.read_gold(tmp_path).columns) == tuple(store.GOLD_DTYPES)
    gold = pd.DataFrame(
        [
            {
                "url_hash": "00",
                "title": "t",
                "source": "hn",
                "lang": "en",
                "label": "neutral",
                "labelled_at": OCT,
                "split": "dev",
                "label2": None,
                "relabelled_at": None,
            }
        ]
    ).astype(store.GOLD_DTYPES)
    assert store.write_gold(tmp_path, gold) == store.LOCAL_GOLD  # unpushed labels
    assert store.read_gold(tmp_path).equals(gold)


def test_split_is_deterministic_and_about_a_third_dev():
    assert store.split_of("0000000000000000") == "dev"
    assert store.split_of("0000000000000001") == "test"
    hashes = [f"{i:016x}" for i in range(3000)]
    dev = sum(store.split_of(h) == "dev" for h in hashes)
    assert dev == 1000


def test_status_roundtrip(tmp_path):
    assert store.read_status(tmp_path) is None
    assert store.write_status(tmp_path, {"run_at": "x", "sources": []}) == "status.json"
    assert store.read_status(tmp_path) == {"run_at": "x", "sources": []}


NOW = datetime(2026, 10, 22, tzinfo=UTC)


def labels(n, label="positive"):
    rows = [
        {
            "url_hash": f"{i:032x}",
            "title": f"t{i}",
            "source": "hn",
            "lang": "en",
            "label": label,
            "labelled_at": NOW,
            "split": store.split_of(f"{i:032x}"),
            "label2": None,
            "relabelled_at": None,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=list(store.GOLD_DTYPES)).astype(store.GOLD_DTYPES)


def test_pull_keeps_unpushed_labels(monkeypatch, tmp_path):
    # The Hub has gold.parquet; a pull rewrites it. Labels not pushed yet must survive.
    labels(0).to_parquet(tmp_path / store.GOLD, index=False)
    store.write_gold(tmp_path, labels(30))

    def hub(repo_id, **kw):
        labels(0).to_parquet(kw["local_dir"] / store.GOLD, index=False)
        return str(kw["local_dir"])

    monkeypatch.setattr(hfsync, "snapshot_download", hub)
    hfsync.pull("me/ds", None, tmp_path)
    assert len(store.read_gold(tmp_path)) == 30


def test_a_local_label_wins_over_the_hub_copy(tmp_path):
    labels(3, "negative").to_parquet(tmp_path / store.GOLD, index=False)
    store.write_gold(tmp_path, labels(1, "positive"))
    gold = store.read_gold(tmp_path).set_index("url_hash")["label"]
    assert gold.to_dict() == {
        f"{0:032x}": "positive",
        f"{1:032x}": "negative",
        f"{2:032x}": "negative",
    }


def test_two_tabs_never_overwrite_each_others_labels(tmp_path):
    # Each tab holds the gold set it loaded; saving it whole dropped the other tab's labels.
    tab_a, tab_b = labels(2).iloc[[0]], labels(2).iloc[[1]]
    store.write_gold(tmp_path, tab_a)
    store.write_gold(tmp_path, tab_b)  # tab B never saw tab A's label
    assert sorted(store.read_gold(tmp_path)["url_hash"]) == [f"{0:032x}", f"{1:032x}"]


def test_publish_then_clear(tmp_path):
    store.write_gold(tmp_path, labels(5))
    published = store.publish_gold(tmp_path)
    assert len(pd.read_parquet(tmp_path / store.GOLD)) == 5
    store.clear_local_gold(tmp_path, published)
    assert not (tmp_path / store.LOCAL_GOLD).exists()
    assert len(store.read_gold(tmp_path)) == 5


def test_a_label_saved_during_a_push_stays_local(tmp_path):
    # Another tab labels while this one pushes: only what was pushed leaves the local file.
    store.write_gold(tmp_path, labels(1))
    published = store.publish_gold(tmp_path)
    store.write_gold(tmp_path, labels(2).iloc[[1]])  # the other tab, mid-push
    store.clear_local_gold(tmp_path, published)
    left = store._read(tmp_path / store.LOCAL_GOLD)
    assert list(left["url_hash"]) == [f"{1:032x}"]
