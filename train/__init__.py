"""The local half: fit the models on public data, score them on the gold set, publish."""

from sentiment.config import ROOT

MODELS_DIR = ROOT / "models"  # gitignored; `inv push-model` uploads it to the private model repo
