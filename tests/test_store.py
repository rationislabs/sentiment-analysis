from datetime import UTC, datetime

import pandas as pd

from sentiment import store
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
    assert store.write_gold(tmp_path, gold) == "gold.parquet"
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
