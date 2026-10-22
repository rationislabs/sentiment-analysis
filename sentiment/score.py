"""Three sentiment models behind one interface: predict(texts, lang) → one Prediction per
text, or None where the model cannot score that language. The pipeline scores each language
with the model model.yaml names (the evaluation's winner).

The TF-IDF files are skops files, not pickles: loading one cannot run code, and the pipeline
downloads only the model-repo commit that models.lock pins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import skops.io
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from .rows import LABELS

NAMES = ("vader", "tfidf", "distilbert")  # cheapest first: the tie-break order
LANGS = ("de", "en")
MAX_LEN = 64
TFIDF_FILE = "tfidf-{lang}.skops"
BERT_DIR = "distilbert"
CHOICE_HEADER = (
    "# The model the pipeline scores with, per language. `uv run inv evaluate` rewrites this\n"
    "# with the winner on the gold dev split.\n"
)


@dataclass(frozen=True)
class Prediction:
    label: str
    probs: tuple[float, float, float]  # negative, neutral, positive


def _argmax(probs) -> str:
    return LABELS[max(range(3), key=lambda i: probs[i])]


class Vader:
    name = "vader"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._a = SentimentIntensityAnalyzer()

    def predict(self, texts: list[str], lang: str) -> list[Prediction | None]:
        if lang != "en":
            return [None] * len(texts)  # VADER's lexicon is English only
        out = []
        for t in texts:
            s = self._a.polarity_scores(t)
            c = s["compound"]
            label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
            out.append(Prediction(label, (s["neg"], s["neu"], s["pos"])))
        return out


class Tfidf:
    name = "tfidf"

    def __init__(self, pipelines: dict[str, Pipeline]) -> None:
        self._pipelines = pipelines

    @staticmethod
    def build() -> Pipeline:
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    @classmethod
    def load(cls, model_dir: Path) -> Tfidf | None:
        pipes = {
            lang: skops.io.load(p)  # refuses any type outside skops' trusted set
            for lang in LANGS
            if (p := model_dir / TFIDF_FILE.format(lang=lang)).exists()
        }
        return cls(pipes) if pipes else None

    def predict(self, texts: list[str], lang: str) -> list[Prediction | None]:
        p = self._pipelines.get(lang)
        if p is None:
            return [None] * len(texts)
        classes = list(p.classes_)
        out = []
        for row in p.predict_proba(texts):
            probs = tuple(
                float(row[classes.index(lab)]) if lab in classes else 0.0 for lab in LABELS
            )
            out.append(Prediction(_argmax(probs), probs))
        return out


class Bert:
    name = "distilbert"

    def __init__(self, path: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(self, texts: list[str], lang: str) -> list[Prediction | None]:
        out: list[Prediction | None] = []
        for i in range(0, len(texts), 32):
            batch = self._tok(
                texts[i : i + 32],
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                probs = self._torch.softmax(self._model(**batch).logits, dim=-1).tolist()
            out += [Prediction(_argmax(p), tuple(p)) for p in probs]
        return out


def load(name: str, model_dir: Path | None):
    """The named model, or None when its files are not there (yet)."""
    if name == "vader":
        return Vader()
    if name == "tfidf":
        return Tfidf.load(model_dir) if model_dir else None
    if name == "distilbert":
        path = model_dir / BERT_DIR if model_dir else None
        return Bert(path) if path and (path / "config.json").exists() else None
    raise ValueError(f"unknown model {name!r}; known: {', '.join(NAMES)}")


class Scorer:
    """The pipeline's models: one per language, as model.yaml names them."""

    def __init__(self, choice: dict[str, str], model_dir: Path | None) -> None:
        self.choice = choice
        self._models = {name: load(name, model_dir) for name in set(choice.values())}

    def score(self, texts: list[str], lang: str) -> tuple[str | None, list[Prediction | None]]:
        name = self.choice.get(lang)
        model = self._models.get(name)
        if model is None:
            return None, [None] * len(texts)
        return name, model.predict(texts, lang)

    def label(self, text: str, lang: str) -> str | None:
        _, [p] = self.score([text], lang)
        return p.label if p else None

    def missing(self) -> list[str]:
        """Languages that cannot be scored right now (no trained model yet)."""
        return [lang for lang in LANGS if self._models.get(self.choice.get(lang)) is None]


def read_choice(path: Path) -> dict[str, str]:
    choice = yaml.safe_load(path.read_text())
    if (
        not isinstance(choice, dict)
        or set(choice) != set(LANGS)
        or any(v not in NAMES for v in choice.values())
    ):
        raise ValueError(f"{path}: needs `de:` and `en:`, each one of {', '.join(NAMES)}")
    return choice


def write_choice(path: Path, winners: dict[str, str | None]) -> None:
    """Record new winners; a language without a winner keeps its current model."""
    choice = read_choice(path)
    choice.update({lang: name for lang, name in winners.items() if name})
    path.write_text(CHOICE_HEADER + yaml.safe_dump(choice, sort_keys=True))


def read_revision(path: Path) -> str | None:
    """The model-repo commit the pipeline loads (models.lock), or None before the first push."""
    if not path.exists():
        return None
    rev = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError(f"{path}: needs a 40-character commit hash, not {rev!r}")
    return rev


def write_revision(path: Path, rev: str) -> None:
    path.write_text(rev + "\n")
