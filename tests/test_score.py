import importlib.util

import pytest

from sentiment import score
from sentiment.score import Prediction

TOY = [
    ("great win today", "positive"),
    ("great success again", "positive"),
    ("great day", "positive"),
    ("terrible loss today", "negative"),
    ("terrible crash again", "negative"),
    ("terrible day", "negative"),
    ("meeting on monday", "neutral"),
    ("meeting on tuesday", "neutral"),
    ("meeting today", "neutral"),
    ("meeting again on friday", "neutral"),
]


def toy_tfidf(langs=("en",)):
    pipes = {}
    for lang in langs:
        p = score.Tfidf.build()
        p.fit([t for t, _ in TOY], [lab for _, lab in TOY])
        pipes[lang] = p
    return score.Tfidf(pipes)


def test_vader_english():
    v = score.Vader()
    got = v.predict(
        [
            "This is a wonderful, great success",
            "Terrible disaster kills dozens",
            "The meeting is on Tuesday",
        ],
        "en",
    )
    assert [p.label for p in got] == ["positive", "negative", "neutral"]
    assert all(abs(sum(p.probs) - 1) < 0.01 for p in got)


def test_vader_has_no_german():
    assert score.Vader().predict(["Großer Erfolg", "Katastrophe"], "de") == [None, None]


def test_tfidf_predicts_and_probs_follow_labels():
    m = toy_tfidf()
    got = m.predict(["a great result", "terrible news", "meeting on wednesday"], "en")
    assert [p.label for p in got] == ["positive", "negative", "neutral"]
    for p in got:
        assert len(p.probs) == 3 and abs(sum(p.probs) - 1) < 1e-6
        assert p.probs[score.LABELS.index(p.label)] == max(p.probs)


def test_tfidf_missing_language_is_none():
    assert toy_tfidf(("en",)).predict(["x"], "de") == [None]


def test_tfidf_roundtrip(tmp_path):
    import skops.io

    m = toy_tfidf(("de", "en"))
    for lang, p in m._pipelines.items():
        skops.io.dump(p, tmp_path / score.TFIDF_FILE.format(lang=lang))
    loaded = score.Tfidf.load(tmp_path)
    assert loaded.predict(["great day"], "de")[0].label == "positive"


def test_a_pickle_in_the_model_repo_is_never_loaded(tmp_path):
    # joblib.load runs whatever the file says: a model repo is not trusted that far.
    import joblib

    m = toy_tfidf(("de",))
    joblib.dump(m._pipelines["de"], tmp_path / "tfidf-de.joblib")
    assert score.TFIDF_FILE.endswith(".skops")
    assert score.Tfidf.load(tmp_path) is None


def test_revision_file(tmp_path):
    lock = tmp_path / "models.lock"
    assert score.read_revision(lock) is None
    score.write_revision(lock, "a" * 40)
    assert score.read_revision(lock) == "a" * 40
    lock.write_text("main\n")
    with pytest.raises(ValueError):
        score.read_revision(lock)


def test_tfidf_load_without_files_is_none(tmp_path):
    assert score.Tfidf.load(tmp_path) is None


def test_load_by_name(tmp_path):
    assert isinstance(score.load("vader", None), score.Vader)
    assert score.load("tfidf", None) is None
    assert score.load("tfidf", tmp_path) is None
    assert score.load("distilbert", tmp_path) is None
    with pytest.raises(ValueError):
        score.load("gpt", tmp_path)


def test_scorer_uses_choice_per_language(tmp_path):
    import skops.io

    p = score.Tfidf.build().fit([t for t, _ in TOY], [lab for _, lab in TOY])
    skops.io.dump(p, tmp_path / "tfidf-de.skops")
    s = score.Scorer({"de": "tfidf", "en": "vader"}, tmp_path)
    name, preds = s.score(["great day"], "de")
    assert name == "tfidf" and preds[0].label == "positive"
    name, preds = s.score(["What a wonderful day"], "en")
    assert name == "vader" and preds[0].label == "positive"
    assert s.label("terrible day", "de") == "negative"
    assert s.missing() == []


def test_scorer_without_models_scores_nothing():
    s = score.Scorer({"de": "tfidf", "en": "tfidf"}, None)
    assert s.score(["x", "y"], "en") == (None, [None, None])
    assert s.label("x", "de") is None
    assert s.missing() == ["de", "en"]


def test_scorer_vader_for_german_is_missing():
    s = score.Scorer({"de": "vader", "en": "vader"}, None)
    assert s.score(["Gut"], "de") == ("vader", [None])


def test_read_and_write_choice(tmp_path):
    p = tmp_path / "model.yaml"
    p.write_text("de: tfidf\nen: tfidf\n")
    assert score.read_choice(p) == {"de": "tfidf", "en": "tfidf"}
    score.write_choice(p, {"en": "distilbert", "de": None})
    assert score.read_choice(p) == {"de": "tfidf", "en": "distilbert"}
    assert p.read_text().startswith("#")


@pytest.mark.parametrize("text", ["de: gpt\nen: tfidf\n", "fr: tfidf\n", "de: tfidf\n", "- a\n"])
def test_read_choice_rejects_bad_files(tmp_path, text):
    p = tmp_path / "model.yaml"
    p.write_text(text)
    with pytest.raises(ValueError):
        score.read_choice(p)


def test_repo_model_yaml_is_valid():
    from sentiment.config import ROOT

    assert score.read_choice(ROOT / "model.yaml") == {"de": "tfidf", "en": "tfidf"}


def test_prediction_is_frozen():
    p = Prediction("neutral", (0.1, 0.8, 0.1))
    with pytest.raises(AttributeError):
        p.label = "positive"


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs the bert group")
def test_bert_predicts_shape(tmp_path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base = "distilbert/distilbert-base-multilingual-cased"
    AutoTokenizer.from_pretrained(base).save_pretrained(tmp_path)
    AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=3,
        id2label=dict(enumerate(score.LABELS)),
        label2id={lab: i for i, lab in enumerate(score.LABELS)},
    ).save_pretrained(tmp_path)
    got = score.Bert(tmp_path).predict(["Guter Tag", "A bad day", "x"], "de")
    assert [p.label in score.LABELS for p in got] == [True, True, True]
    assert all(abs(sum(p.probs) - 1) < 1e-5 for p in got)
