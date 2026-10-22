"""python -m train fit|evaluate|push

fit       train TF-IDF (and DistilBERT unless --skip-bert) into models/
evaluate  score models/ on the gold set → metrics.json, model.yaml, models/README.md
push      upload models/ to the private model repo and pin that commit in models.lock"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime

from sentiment import config, hfsync, score, store

from . import MODELS_DIR, data, evaluation, tfidf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m train")
    sub = p.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--limit", type=int, default=data.LIMIT_PER_LANG)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--skip-bert", action="store_true")
    sub.add_parser("evaluate")
    sub.add_parser("push")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = config.load()

    if args.cmd == "fit":
        df = data.load_public(args.limit)
        tfidf.save(tfidf.fit(df), MODELS_DIR)
        if not args.skip_bert:
            from . import bert

            bert.finetune(df, MODELS_DIR, epochs=args.epochs)
        print(f"Trained on {len(df)} rows → {MODELS_DIR}")
    elif args.cmd == "evaluate":
        gold = store.read_gold(hfsync.data_root(settings))
        models = {n: m for n in score.NAMES if (m := score.load(n, MODELS_DIR)) is not None}
        metrics = evaluation.evaluate(gold, models, datetime.now(UTC))
        settings.metrics_path.write_text(json.dumps(metrics, indent=1) + "\n")
        score.write_choice(
            config.ROOT / "model.yaml",
            {lang: m.get("winner") for lang, m in metrics["languages"].items()},
        )
        (MODELS_DIR / "README.md").write_text(evaluation.model_card(metrics))
        print(f"Wrote {settings.metrics_path.name}, model.yaml and the model card")
    else:
        rev = hfsync.push_folder(settings.model_repo, settings.hf_token, MODELS_DIR, "models")
        score.write_revision(config.ROOT / "models.lock", rev)
        print(f"Pushed {MODELS_DIR} to {settings.model_repo} at {rev}; commit models.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
