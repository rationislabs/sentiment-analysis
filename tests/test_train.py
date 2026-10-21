import importlib.util
import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from sentiment import score, store
from sentiment.score import Prediction
from tests.test_score import TOY
from train import data, evaluation, tfidf

NOW = datetime(2026, 11, 1, 10, tzinfo=UTC)


def test_to_frame_maps_class_ids_and_strings():
    tweets = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 2]})
    assert data.to_frame("tweet_eval", tweets)["label"].tolist() == [
        "negative",
        "neutral",
        "positive",
    ]
    sb = pd.DataFrame(
        {"ID": [1, 2, 3], "Text": ["x", None, "z"], "Sentiment": ["positive", "neutral", "mixed"]}
    )
    out = data.to_frame("sb10k", sb)
    assert out.to_dict("records") == [
        {"text": "x", "label": "positive", "lang": "de", "dataset": "sb10k"}
    ]


def test_cap_per_language_is_seeded():
    df = pd.DataFrame(
        {
            "text": [str(i) for i in range(30)],
            "label": ["neutral"] * 30,
            "lang": ["de"] * 20 + ["en"] * 10,
        }
    )
    out = data.cap(df, 5)
    assert out["lang"].value_counts().to_dict() == {"de": 5, "en": 5}
    assert out.equals(data.cap(df, 5))


def test_sets_cover_both_languages():
    assert {lang for lang, *_ in data.SETS.values()} == {"de", "en"}


def toy_df(langs=("de", "en")):
    return pd.DataFrame(
        [{"text": t, "label": lab, "lang": lang} for lang in langs for t, lab in TOY]
    )


def test_tfidf_fit_and_save_load_with_score(tmp_path):
    tfidf.save(tfidf.fit(toy_df()), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tfidf-de.joblib", "tfidf-en.joblib"]
    m = score.Tfidf.load(tmp_path)
    assert m.predict(["great day"], "de")[0].label == "positive"


class Fixed:
    """A fake model: the label is the first word of the title; German only if told so."""

    def __init__(self, name, langs=("de", "en")):
        self.name, self.langs = name, langs

    def predict(self, texts, lang):
        if lang not in self.langs:
            return [None] * len(texts)
        return [Prediction(t.split()[0], (0.2, 0.6, 0.2)) for t in texts]


def gold_rows(lang, n_dev, n_test):
    rows = []
    for split, n in (("dev", n_dev), ("test", n_test)):
        for i in range(n):
            lab = ["negative", "neutral", "positive"][i % 3]
            rows.append(
                {
                    "url_hash": f"{lang}{split}{i}",
                    "title": f"{lab} headline {i}",
                    "source": "hn",
                    "lang": lang,
                    "label": lab,
                    "labelled_at": NOW,
                    "split": split,
                    "label2": None,
                    "relabelled_at": None,
                }
            )
    return rows


def test_evaluate_without_gold():
    empty = pd.DataFrame(columns=list(store.GOLD_DTYPES)).astype(store.GOLD_DTYPES)
    m = evaluation.evaluate(empty, {}, NOW)
    assert m["languages"] == {
        "de": {"status": "no gold set yet"},
        "en": {"status": "no gold set yet"},
    }
    assert m["self_agreement"] is None and m["generated_at"] == NOW.isoformat()


def test_evaluate_with_gold():
    gold = pd.DataFrame(gold_rows("en", 6, 9) + gold_rows("de", 6, 9)).astype(store.GOLD_DTYPES)
    gold.loc[gold.index[:3], "label2"] = gold.loc[gold.index[:3], "label"]
    models = {"vader": Fixed("vader", langs=("en",)), "tfidf": Fixed("tfidf")}
    m = evaluation.evaluate(gold, models, NOW)
    json.dumps(m)  # serialisable
    en, de = m["languages"]["en"], m["languages"]["de"]
    assert (en["n_dev"], en["n_test"]) == (6, 9)
    assert {r["model"] for r in en["test"]} == {"majority", "vader", "tfidf"}
    assert {r["model"] for r in de["test"]} == {"majority", "tfidf"}  # VADER cannot score German
    assert en["winner"] == "vader"  # both perfect: overlap, the cheaper wins
    assert de["winner"] == "tfidf"
    assert all(r["split"] == "dev" for r in en["dev"]) and all(
        r["model"] != "majority" for r in en["dev"]
    )
    assert m["self_agreement"] == {"n": 3, "kappa": 1.0}


def test_model_card_names_licences_and_results():
    gold = pd.DataFrame(gold_rows("en", 6, 9)).astype(store.GOLD_DTYPES)
    card = evaluation.model_card(evaluation.evaluate(gold, {"tfidf": Fixed("tfidf")}, NOW))
    assert card.startswith("---\n") and "license: cc-by-nc-4.0" in card
    for name in (
        "cardiffnlp/tweet_eval",
        "Alienmaster/SB10k",
        "uhhlt/GermEval2017",
        "CC BY-NC 4.0",
    ):
        assert name in card
    assert "| en | tfidf |" in card and "no gold set yet" in card


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs the bert group")
def test_bert_finetune_smoke(tmp_path):
    from train import bert

    bert.finetune(toy_df().head(16), tmp_path, epochs=1, batch=8)
    got = score.Bert(tmp_path / score.BERT_DIR).predict(["great day"], "en")
    assert got[0].label in score.LABELS
