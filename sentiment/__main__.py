"""python -m sentiment collect — one collection run (the cron calls this every 4 hours)."""

from __future__ import annotations

import logging
import sys

from . import config, run


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["collect"]:
        print("usage: python -m sentiment collect", file=sys.stderr)
        return 2
    settings = config.load()
    if settings.data_dir is None and not settings.hf_token:
        print(
            "HF_TOKEN is not set (or set SENTIMENT_DATA_DIR for a local run without the Hub)",
            file=sys.stderr,
        )
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run.run(settings)


if __name__ == "__main__":
    sys.exit(main())
