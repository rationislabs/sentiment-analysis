import pytest

from sentiment import evaluate
from sentiment.evaluate import Result
from sentiment.score import Prediction

Y = ["negative", "neutral", "positive"] * 4


def test_macro_f1_perfect_and_all_neutral():
    assert evaluate.macro_f1(Y, Y) == 1.0
    assert evaluate.macro_f1(Y, ["neutral"] * 12) == pytest.approx(1 / 6)


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    pred = Y[:8] + ["neutral"] * 4
    a = evaluate.bootstrap_ci(Y, pred, n=200, seed=1)
    assert a == evaluate.bootstrap_ci(Y, pred, n=200, seed=1)
    assert a[0] <= evaluate.macro_f1(Y, pred) <= a[1]
    assert evaluate.bootstrap_ci(Y, Y, n=50) == (1.0, 1.0)


def test_result_fields():
    r = evaluate.result("tfidf", "en", "test", Y, Y[:9] + ["neutral"] * 3)
    assert (r.model, r.lang, r.split, r.n) == ("tfidf", "en", "test", 12)
    assert set(r.per_class) == {"negative", "neutral", "positive"}
    assert set(r.per_class["neutral"]) == {"precision", "recall"}
    assert r.confusion[1][1] == 4 and sum(map(sum, r.confusion)) == 12


def res(model, f1, lo, hi):
    return Result(model, "en", "dev", 100, f1, (lo, hi), {}, [])


def test_winner_clear():
    assert (
        evaluate.winner([res("tfidf", 0.5, 0.45, 0.55), res("distilbert", 0.7, 0.65, 0.75)])
        == "distilbert"
    )


def test_winner_overlap_goes_to_cheaper():
    assert (
        evaluate.winner([res("tfidf", 0.6, 0.55, 0.66), res("distilbert", 0.62, 0.57, 0.68)])
        == "tfidf"
    )
    assert (
        evaluate.winner([res("vader", 0.58, 0.5, 0.66), res("distilbert", 0.62, 0.57, 0.68)])
        == "vader"
    )


def test_winner_ignores_majority_and_empty():
    assert evaluate.winner([]) is None
    assert evaluate.winner([res("majority", 0.9, 0.85, 0.95)]) is None


def test_majority_ties_go_by_label_order():
    assert evaluate.majority(["positive", "neutral", "neutral"]) == "neutral"
    assert evaluate.majority(["positive", "negative"]) == "negative"


def test_kappa():
    assert evaluate.kappa(Y, Y) == 1.0
    assert evaluate.kappa(["neutral"], ["neutral"]) is None


def test_worst_errors_most_confident_first():
    preds = [
        Prediction("positive", (0.1, 0.1, 0.8)),
        Prediction("negative", (0.6, 0.2, 0.2)),
        Prediction("neutral", (0.0, 1.0, 0.0)),
    ]
    got = evaluate.worst_errors(["a", "b", "c"], ["negative", "positive", "neutral"], preds, k=5)
    assert [g["title"] for g in got] == ["a", "b"]
    assert got[0] == {"title": "a", "gold": "negative", "predicted": "positive", "confidence": 0.8}


def test_disagreements():
    got = evaluate.disagreements(
        ["a", "b", "c"],
        ["neutral", "neutral", "positive"],
        {
            "vader": ["neutral", "positive", "negative"],
            "tfidf": ["neutral", "negative", "neutral"],
            "distilbert": ["neutral", "negative", "positive"],
        },
    )
    assert [g["title"] for g in got] == ["c", "b"]
    assert got[0]["predictions"] == {
        "vader": "negative",
        "tfidf": "neutral",
        "distilbert": "positive",
    }
    assert got[0]["gold"] == "positive"
