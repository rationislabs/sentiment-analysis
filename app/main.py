"""The dashboard. Streamlit Community Cloud runs this file; locally: `uv run inv app`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sentiment/ is not installed

import logging  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sentiment import config, dashboard, hfsync, store  # noqa: E402
from sentiment.rows import LABELS  # noqa: E402

LANG_NAMES = {"de": "German", "en": "English"}

st.set_page_config(page_title="Headline sentiment", layout="wide")
settings = config.load()


@st.cache_data(ttl=3600, show_spinner="Loading headlines…")
def load() -> tuple[pd.DataFrame, pd.DataFrame, dict | None, str]:
    root = hfsync.data_root(settings)
    return (
        store.read_headlines(root),
        store.read_gold(root),
        store.read_status(root),
        str(root),
    )


def render_trends(headlines: pd.DataFrame, status: dict | None) -> None:
    lang = st.selectbox(
        "Language",
        ["all", "de", "en"],
        key="trend_lang",
        format_func=lambda x: LANG_NAMES.get(x, "All"),
    )
    t = dashboard.trends(headlines, None if lang == "all" else lang)
    if t.empty:
        st.info("No scored headlines yet.")
    else:
        st.caption(
            "Daily mean of P(positive) − P(negative) per source: −1 all negative, +1 all positive."
        )
        st.line_chart(t)
    st.subheader("Last run")
    if status:
        st.caption(f"Run at {status['run_at']}")
        st.dataframe(dashboard.status_table(status), hide_index=True)
    st.caption(
        "Refused: the site's robots.txt, its AI-crawler rules, a rights-reservation text or "
        "TDMRep said no, so the scraper did not run there. Feeds are still read."
    )


def render_headlines(headlines: pd.DataFrame) -> None:
    query = st.text_input("Search titles", key="q")
    st.dataframe(
        dashboard.headline_table(headlines, query),
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("Link")},
    )


def render_models() -> None:
    metrics = dashboard.read_metrics(settings.metrics_path)
    if metrics is None:
        st.info(dashboard.NO_GOLD)
        return
    st.caption(f"Evaluated {metrics['generated_at']} on hand-labelled headlines.")
    if sa := metrics.get("self_agreement"):
        st.metric(
            "Self-agreement (Cohen's kappa)", f"{sa['kappa']:.2f}", help=f"{sa['n']} blind relabels"
        )
    for lang, m in metrics["languages"].items():
        st.subheader(LANG_NAMES[lang])
        if "status" in m:
            st.info(dashboard.NO_GOLD)
            continue
        st.markdown(
            f"Winner on the dev split ({m['n_dev']} headlines): **{m['winner'] or 'none'}**. "
            f"Test split: {m['n_test']} headlines; the floor always says “{m['majority_label']}”."
        )
        st.dataframe(dashboard.scores_table(m["test"]), hide_index=True)
        for r in m["test"]:
            with st.expander(f"Confusion matrix: {r['model']}"):
                st.dataframe(pd.DataFrame(r["confusion"], index=LABELS, columns=LABELS))
        with st.expander("Worst errors"):
            for name, errors in m["worst_errors"].items():
                st.markdown(f"**{name}**")
                st.dataframe(pd.DataFrame(errors), hide_index=True)
        with st.expander("Where the models disagree"):
            st.dataframe(pd.DataFrame(m["disagreements"]), hide_index=True)


def render_label(headlines: pd.DataFrame, root: Path) -> None:
    if "gold" not in st.session_state:
        st.session_state.gold = store.read_gold(root)
        st.session_state.unpushed = 0
    gold = st.session_state.gold
    st.markdown(dashboard.GUIDELINE)
    st.progress(
        min(len(gold) / dashboard.GOLD_TARGET, 1.0),
        text=f"{len(gold)} of {dashboard.GOLD_TARGET} labelled",
    )
    now = datetime.now(UTC)
    queue = dashboard.relabel_queue(gold, now)
    if not queue.empty:
        row, mode = queue.iloc[0], "relabel"
        st.caption("Blind relabel: label it again without looking back.")
    else:
        row, mode = dashboard.next_to_label(headlines, gold), "label"
    if row is None:
        st.info("Nothing left to label: wait for the next collection run.")
    else:
        st.markdown(f"### {row['title']}")
        st.caption(f"{row['source']} · {LANG_NAMES[row['lang']]}")
        for col, label in zip(st.columns(3), LABELS, strict=True):
            if col.button(label.capitalize(), key=f"{mode}-{label}"):
                if mode == "relabel":
                    gold = dashboard.add_relabel(gold, row["url_hash"], label, now)
                else:
                    gold = dashboard.add_label(gold, row, label, now)
                store.write_gold(root, gold)
                st.session_state.gold = gold
                st.session_state.unpushed += 1
                st.rerun()
    # Offered whenever there are labels, not only after labelling in this session: a refresh
    # must never hide labels that are saved locally but not yet on the Hub.
    if settings.data_dir is None and len(st.session_state.gold):
        unpushed = st.session_state.unpushed
        text = f"Push {unpushed} new labels to the Hub" if unpushed else "Push labels to the Hub"
        if st.button(text, key="push"):
            try:
                hfsync.push(
                    settings.dataset_repo, settings.hf_token, root, [store.GOLD], "gold labels"
                )
            except Exception:  # noqa: BLE001 — keep the labels on disk; say so
                logging.getLogger(__name__).exception("gold push failed")
                st.error("Push failed; the labels are saved locally. Try again.")
            else:
                st.session_state.unpushed = 0
                st.success("Pushed.")


st.title("Headline sentiment")
st.caption(
    "German and English news headlines, collected politely and scored every 4 hours. Rationis Labs."
)

try:
    headlines, _gold, status, root = load()
except Exception:  # noqa: BLE001 — the Hub or the network is down: say so, never a stack trace
    logging.getLogger(__name__).exception("loading the dataset failed")
    st.error("Data unavailable — the dataset could not be loaded.")
    if st.button("Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

names = ["Trends", "Headlines", "Models"] + (["Label"] if settings.label_mode else [])
tabs = st.tabs(names)
with tabs[0]:
    render_trends(headlines, status)
with tabs[1]:
    render_headlines(headlines)
with tabs[2]:
    render_models()
if settings.label_mode:
    with tabs[3]:
        render_label(headlines, Path(root))
