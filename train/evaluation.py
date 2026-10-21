"""Score every available model on the owner's gold set, per language: the dev split picks
the winner, the test split gives the reported numbers. Also the model card."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

import pandas as pd

from sentiment.evaluate import disagreements, kappa, majority, result, winner, worst_errors
from sentiment.score import LANGS

NO_GOLD = {"status": "no gold set yet"}


def evaluate(gold: pd.DataFrame, models: dict, now: datetime) -> dict:
    out: dict = {"generated_at": now.isoformat(), "self_agreement": None, "languages": {}}
    both = gold[gold["label"].notna() & gold["label2"].notna()]
    if (k := kappa(both["label"].tolist(), both["label2"].tolist())) is not None:
        out["self_agreement"] = {"n": len(both), "kappa": round(k, 3)}
    for lang in LANGS:
        g = gold[(gold["lang"] == lang) & gold["label"].notna()]
        dev, test = g[g["split"] == "dev"], g[g["split"] == "test"]
        if dev.empty or test.empty:
            out["languages"][lang] = dict(NO_GOLD)
            continue
        dev_y, test_y = dev["label"].tolist(), test["label"].tolist()
        test_titles = test["title"].tolist()
        majority_label = majority(dev_y)
        dev_results = []
        test_results = [result("majority", lang, "test", test_y, [majority_label] * len(test_y))]
        test_preds = {}
        for name, model in models.items():
            dp = model.predict(dev["title"].tolist(), lang)
            tp = model.predict(test_titles, lang)
            if any(p is None for p in dp + tp):
                continue  # this model cannot score this language (VADER on German)
            dev_results.append(result(name, lang, "dev", dev_y, [p.label for p in dp]))
            test_results.append(result(name, lang, "test", test_y, [p.label for p in tp]))
            test_preds[name] = tp
        out["languages"][lang] = {
            "n_dev": len(dev),
            "n_test": len(test),
            "majority_label": majority_label,
            "winner": winner(dev_results),
            "dev": [asdict(r) for r in dev_results],
            "test": [asdict(r) for r in test_results],
            "worst_errors": {
                n: worst_errors(test_titles, test_y, p) for n, p in test_preds.items()
            },
            "disagreements": disagreements(
                test_titles, test_y, {n: [x.label for x in p] for n, p in test_preds.items()}
            ),
        }
    return out


def model_card(metrics: dict) -> str:
    lines = [
        "---",
        "license: cc-by-nc-4.0",
        "language: [de, en]",
        "base_model: distilbert/distilbert-base-multilingual-cased",
        "pipeline_tag: text-classification",
        "---",
        "",
        "# Headline sentiment models (Rationis Labs)",
        "",
        "Three-class sentiment (negative, neutral, positive) for German and English news",
        "headlines: per-language TF-IDF + logistic regression (`tfidf-de.joblib`,",
        "`tfidf-en.joblib`) and a fine-tuned multilingual DistilBERT (`distilbert/`).",
        "",
        "## Training data",
        "",
        "| Lang | Dataset | Content | Licence |",
        "|---|---|---|---|",
        "| en | cardiffnlp/tweet_eval (sentiment) | tweets | CC BY 3.0 |",
        "| de | Alienmaster/SB10k | tweets | CC BY 4.0 |",
        "| de | uhhlt/GermEval2017 | Deutsche Bahn customer feedback | CC BY-NC 4.0 |",
        "",
        "Because GermEval-2017 is non-commercial, so are these models.",
        "",
        "## Results on hand-labelled headlines (test split, macro-F1, 95 % CI)",
        "",
        "| Lang | Model | n | Macro-F1 | 95 % CI |",
        "|---|---|---|---|---|",
    ]
    for lang, m in metrics["languages"].items():
        if "status" in m:
            lines.append(f"| {lang} | — | 0 | {m['status']} | — |")
            continue
        for r in m["test"]:
            lo, hi = r["ci"]
            lines.append(
                f"| {lang} | {r['model']} | {r['n']} | {r['macro_f1']:.3f} | {lo:.3f}–{hi:.3f} |"
            )
    sa = metrics.get("self_agreement")
    lines += [
        "",
        "## Limitations",
        "",
        "- Trained on tweets and customer feedback, tested on news headlines: the domain shift",
        "  is measured above, not hidden.",
        "- One annotator labelled the gold set. "
        + (
            f"Self-agreement on {sa['n']} blind relabels: Cohen's kappa {sa['kappa']}."
            if sa
            else "Self-agreement not measured yet."
        ),
        "",
    ]
    return "\n".join(lines)
