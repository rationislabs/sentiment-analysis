from pathlib import Path

from sentiment import config


def test_defaults():
    s = config.load({})
    assert s.hf_token is None
    assert s.dataset_repo == "rationislabs/sentiment-headlines"
    assert s.model_repo == "rationislabs/sentiment-models"
    assert s.data_dir is None and s.model_dir is None
    assert s.metrics_path == config.ROOT / "metrics.json"
    assert s.label_mode is False


def test_values_from_env():
    s = config.load(
        {
            "HF_TOKEN": "hf_fake",
            "HF_DATASET_REPO": "me/ds",
            "HF_MODEL_REPO": "me/m",
            "SENTIMENT_DATA_DIR": "data",
            "SENTIMENT_MODEL_DIR": " models ",
            "SENTIMENT_METRICS_PATH": "/tmp/m.json",
            "LABEL_MODE": "1",
        }
    )
    assert s.hf_token == "hf_fake"
    assert (s.dataset_repo, s.model_repo) == ("me/ds", "me/m")
    assert s.data_dir == Path("data") and s.model_dir == Path("models")
    assert s.metrics_path == Path("/tmp/m.json")
    assert s.label_mode is True


def test_empty_values_fall_back_to_defaults():
    s = config.load({"HF_DATASET_REPO": "", "SENTIMENT_DATA_DIR": "  ", "LABEL_MODE": "0"})
    assert s.dataset_repo == "rationislabs/sentiment-headlines"
    assert s.data_dir is None and s.label_mode is False


def test_token_never_in_repr():
    s = config.load({"HF_TOKEN": "hf_secret_value"})
    assert "hf_secret_value" not in repr(s) and "hf_secret_value" not in str(s)


def test_readme_covers_the_promises():
    text = (config.ROOT / "README.md").read_text()
    for needle in (
        "RationisLabsSentimentBot/1.0 (+https://github.com/rationislabs/sentiment-analysis)",
        "You are responsible",
        "cannot be switched off",
        "CC BY-NC 4.0",
        "Flood kills 12",
        "uv run inv hub-init",
        "HF_DATASET_REPO",
        "LABEL_MODE",
        "Never stored",
    ):
        assert needle in text, needle
