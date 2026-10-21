# sentiment-analysis

German and English news headlines, collected politely every 4 hours, scored for sentiment,
and shown on a dashboard — plus an honest comparison of three sentiment models on headlines
labelled by hand. A Rationis Labs portfolio project.

```
GitHub Actions cron (every 4 h)
  collect ─► politeness gate ─► score ─► store ─► Hugging Face dataset (private, Parquet)
                                                       │
Streamlit Community Cloud ◄── read-only token ─────────┘
local only: train ─► evaluate on the gold set ─► winner ─► Hugging Face model repo (private)
```

## Sources

| Source | Language | Feed | Scrapers today |
|---|---|---|---|
| BBC News (World) | en | `feeds.bbci.co.uk/news/world/rss.xml` | refused: AI-crawler block, TDM reservation text |
| The Guardian (World) | en | `theguardian.com/world/rss` | refused: AI-crawler block |
| Hacker News | en | `news.ycombinator.com/rss` | front page allowed (30 s crawl delay) |
| Tagesschau | de | `tagesschau.de/index~rss2.xml` | refused: AI-crawler block, reservation of rights |
| ORF | de | `rss.orf.at/news.xml` | refused: AI-crawler block |
| Der Standard | de | `derstandard.at/rss` | refused: AI-crawler block |
| heise online | de | `heise.de/rss/heise-atom.xml` | refused: AI-crawler block, TDM reservation text |

Checked on 2026-10-08; the gate re-checks every run, and the dashboard's Trends tab shows
what was refused and why. Headlines belong to their publishers.

## Politeness: what cannot be switched off

Every request goes through `sentiment/polite.py`. None of this has a switch:

- **User-Agent** `RationisLabsSentimentBot/1.0 (+https://github.com/rationislabs/sentiment-analysis)`.
- **robots.txt** (RFC 9309, longest-match rules): our agent must be allowed. No robots.txt
  (4xx) means no rules; an unreachable one (5xx, network error) means no requests at all.
- **Rate limit:** one request per 5 seconds per host, or the site's `Crawl-delay` if longer.
- **Scrapers only** (front pages, article bodies) are also refused when the site:
  - blocks any named AI or TDM crawler for that URL (GPTBot, ClaudeBot, CCBot, Google-Extended
    and six more), or
  - writes a rights reservation in its robots.txt comments ("text and data mining",
    "Art. 4", "reservation of rights", "Nutzungsvorbehalt", …), or
  - reserves TDM with TDMRep (`/.well-known/tdmrep.json`, the `tdm-reservation` header or
    meta tag).

  A site that blocks AI crawlers has said no to automated reading of its pages. We do not
  slip through because it did not name our bot.

## Scraper switches

`sources.yaml` has two switches per site:

- `frontpage: true` (default) — rank, section and lead story from the front page.
- `body: false` (default) — the sentiment of the article text as well.

**You are responsible for switching body scraping on**, and for checking that your use is
allowed where you run it. The article text is scored in memory and dropped: **Never stored**,
never logged, never shown. The politeness rules above still apply to every request.

## Models and the honest metrics

| Model | Languages | Notes |
|---|---|---|
| Majority class | de, en | the floor: always the most common gold label |
| VADER (`vaderSentiment` 3.3.2) | en | lexicon; no maintained German version exists |
| TF-IDF + logistic regression | de, en | one per language |
| `distilbert/distilbert-base-multilingual-cased`, fine-tuned | de, en | one model for both |

Training data (downloaded on use, never committed):

| Lang | Dataset | Content | Licence |
|---|---|---|---|
| en | `cardiffnlp/tweet_eval` (sentiment) | tweets | CC BY 3.0 |
| de | `Alienmaster/SB10k` | tweets | CC BY 4.0 |
| de | `uhhlt/GermEval2017` | Deutsche Bahn customer feedback | CC BY-NC 4.0 |

Because GermEval-2017 is non-commercial, the trained models are too.

**The gold set** is ~600 of the collected headlines, labelled by hand: a third is the *dev*
split (chooses the model), the rest the *test* split (the reported numbers, never tuned on).
50 of them are relabelled blind a week later; Cohen's kappa of the two passes is the label
noise, and a model gap smaller than that means nothing.

**Metrics,** per language: macro-F1 (most headlines are neutral, so accuracy would flatter
"always neutral"), per-class precision and recall, confusion matrices, and a 95 % bootstrap
interval (1,000 resamples). **Winner rule:** the highest dev macro-F1; if its interval
overlaps another model's, the cheaper one wins. `model.yaml` records the winner per language
and the pipeline follows it.

Results appear on the dashboard's Models tab and in `metrics.json` once the gold set exists.

### Labelling guideline

Label the event as a reader experiences it, not the writer's tone. "Flood kills 12" is
negative; "ECB holds rates" is neutral; "Vaccine approved for children" is positive.

## Running it locally

```bash
uv sync
uv run inv test                 # offline tests
uv run inv train --skip-bert    # TF-IDF into models/ (~1 minute); without it nothing is scored
uv run inv collect --local      # one run into data/, the Hub untouched (takes ~2 minutes)
uv run inv app --local          # dashboard with the Label tab on http://127.0.0.1:8501
```

| Task | What it does |
|---|---|
| `uv run inv collect [--local]` | one collection run |
| `uv run inv app [--local]` | dashboard with the Label tab (`LABEL_MODE=1`) |
| `uv run inv train [--skip-bert]` | train TF-IDF and DistilBERT into `models/` (about an hour on CPU) |
| `uv run inv evaluate` | score `models/` on the gold set: `metrics.json`, `model.yaml`, model card |
| `uv run inv push-model` | upload `models/` to the private model repo |
| `uv run inv hub-init` | create both private Hub repos |
| `uv run inv fixtures` | re-record the robots.txt fixtures through the gate (feeds and front pages are invented) |

Settings (`.env`, see `.env.example`): `HF_TOKEN`, `HF_DATASET_REPO`, `HF_MODEL_REPO`,
`SENTIMENT_DATA_DIR`, `SENTIMENT_MODEL_DIR`, `LABEL_MODE`.

## One-time setup (free plans only)

1. Create a Hugging Face account for Rationis Labs and two tokens: one with **write** access
   (local use and GitHub Actions), one **read-only** (the dashboard).
2. Put the write token in `.env` as `HF_TOKEN`, then `uv run inv hub-init`.
3. In the GitHub repo: secret `HF_TOKEN` (write token); variables `HF_DATASET_REPO` and
   `HF_MODEL_REPO`. The cron stays off until `HF_DATASET_REPO` is set.
4. Streamlit Community Cloud: new app from this repo, main file `app/main.py`, Python 3.12,
   secret `HF_TOKEN` = the read-only token, plus `HF_DATASET_REPO`. Invite the viewers you want.
5. After a few days of collection: `uv run inv app`, label ~600 headlines in the Label tab,
   press "Push labels" (until then they are kept in `~/.cache/sentiment-data`). A week later,
   do the 50 blind relabels the tab offers.
6. `uv run inv train`, `uv run inv evaluate`, `uv run inv push-model`, then commit
   `metrics.json` and `model.yaml`.

Contact: [github.com/rationislabs](https://github.com/rationislabs)
