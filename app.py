"""
RoomRead — aspect-based sentiment analysis for hotel reviews.

Pages
  Priorities       Ranking of service areas by share of reviews with a negative mention
  Check a review   Sentence-by-sentence analysis of one pasted review
  Analyse a file   Upload a CSV / Excel / TXT of reviews and get your own ranking
  Model evidence   Test-set results for RoBERTa, BERT and zero-shot BART
  About            Pipeline, data, limitations, author

Inference logic (sentence split, keyword aspect tagger, input format, relevance
gate at 0.55) is kept identical to the previously deployed app so predictions
do not change.
"""

import html
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="RoomRead | Hotel review priorities",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================================
# CONFIGURATION
# =====================================================================
def setting(name, default=None):
    """Read a value from Streamlit secrets, then environment, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def find_local_model():
    """Find a Hugging Face model folder committed to the repo (config.json + weights)."""
    skip = {".git", ".streamlit", "venv", ".venv", "__pycache__", "node_modules"}
    for cfg in sorted(APP_DIR.rglob("config.json")):
        rel = cfg.relative_to(APP_DIR)
        if len(rel.parts) > 4 or any(p in skip for p in rel.parts):
            continue
        folder = cfg.parent
        has_weights = any(folder.glob("*.safetensors")) or any(folder.glob("pytorch_model*.bin"))
        if has_weights:
            return str(folder)
    return None


MODEL_ID = setting("MODEL_ID") or find_local_model() or "nethika/aspecta"
RELEVANCE_MODEL = setting("RELEVANCE_MODEL", "valhalla/distilbart-mnli-12-3")
RANKING_CSV = APP_DIR / "aspect_priority_ranking.csv"

MAX_LEN = 256
RELEVANCE_THRESHOLD = 0.55       # unchanged from the deployed app
LOW_CONFIDENCE = 0.60            # predictions below this are flagged for a manual check
BENCHMARK_REVIEWS = 31_219       # size of the unlabelled corpus behind the ranking
MAX_FILE_REVIEWS = 500

FALLBACK_ID_TO_SENTIMENT = {0: "positive", 1: "negative", 2: "neutral"}

RELEVANCE_LABELS = [
    "a hotel guest review describing a stay",
    "unrelated text with no connection to a hotel stay",
]

ASPECT_KEYWORDS = {
    "Room quality": ["room", "bed", "pillow", "bathroom", "shower", "clean", "dirty", "smell",
                     "noisy", "air condition", "ac ", "view", "dust"],
    "Staff & Service": ["staff", "service", "receptionist", "front desk", "concierge", "employee",
                        "manager", "helpful", "rude", "friendly"],
    "Food & Beverage": ["breakfast", "food", "restaurant", "buffet", "dinner", "lunch", "meal",
                        "coffee", "bar ", "drink"],
    "Location": ["location", "located", "walk", "distance", "nearby", "downtown", "beach",
                 "airport", "transport"],
    "Value for money": ["price", "value", "expensive", "cheap", "worth", "cost", "money",
                        "overpriced", "affordable"],
    "Facilities": ["pool", "gym", "wifi", "parking", "spa", "elevator", "facility", "facilities",
                   "amenities"],
    "Overall": ["overall", "stay", "hotel", "experience", "recommend", "trip", "manage", "manag"],
}
ASPECTS = list(ASPECT_KEYWORDS)

ASPECT_ACTIONS = {
    "Room quality": "Spot-check rooms before check-in for cleanliness, bed and bathroom condition, "
                    "noise and air-conditioning. Start with the rooms that come up most in complaints.",
    "Overall": "These complaints don't point at one department. Read a sample in full to find "
               "the pattern behind general disappointment.",
    "Food & Beverage": "Check breakfast at peak time: food temperature, how fast the buffet is "
                       "refilled, and variety across a week.",
    "Value for money": "Compare what the rate promises with what guests get. Make included extras "
                       "visible at booking and at check-in.",
    "Staff & Service": "Review how the front desk handles complaints and how long guest requests "
                       "take to close.",
    "Facilities": "Keep a fault log for Wi-Fi, lifts, parking and the pool with a fix-by date, "
                  "and tell guests what is out of service.",
    "Location": "The building can't move, but expectations can. Describe the location honestly "
                "in listings and offer directions or transport.",
}

SAMPLES = {
    "Mixed stay": "The room was spotless and the bed was really comfortable. Breakfast was cold "
                  "and the buffet ran out early. Staff at the front desk were friendly and helpful. "
                  "For the price, I expected better wifi.",
    "Unhappy guest": "Our room smelled of damp and the air conditioning was noisy all night. "
                     "The receptionist was rude when we asked to move. Overpriced for what you get.",
    "Happy guest": "Great location, just a short walk to the beach. The staff went out of their "
                   "way to help us. Our room had a lovely view and the pool was clean. "
                   "Would definitely recommend this hotel.",
    "Not a review": "Can you send me the lecture slides and the lab sheet for tomorrow's class?",
}

# Colours (kept in sync with .streamlit/config.toml)
INK = "#17202A"
MUTED = "#56616C"
NIGHT = "#1F3B57"
BRASS = "#A27B3F"
HAIR = "#DEDFDA"
SENT_COLOURS = {"negative": "#B3261E", "positive": "#2F7A55", "neutral": "#7C8591", "mixed": "#B7791F"}
TIER_COLOURS = {"Critical": "#A61E17", "High": "#D9776A", "Watch": "#7D8B99"}
BODY_FONT = "Public Sans, system-ui, sans-serif"


# =====================================================================
# TEXT PROCESSING (unchanged from the deployed app)
# =====================================================================
def split_sentences(text):
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def detect_aspects(sentence):
    s = sentence.lower()
    return [asp for asp, kws in ASPECT_KEYWORDS.items() if any(kw in s for kw in kws)]


def matched_keywords(sentence, aspect):
    s = sentence.lower()
    return sorted({kw.strip() for kw in ASPECT_KEYWORDS[aspect] if kw in s})


def build_input_text(sentence, aspect):
    return f"{sentence} [SEP] {aspect}"


# =====================================================================
# MODELS
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_id):
    """Returns a dict. Failures are returned (not raised) so the rest of the app keeps working."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        cfg_labels = getattr(model.config, "id2label", None) or {}
        labels = {int(k): str(v).lower() for k, v in cfg_labels.items()}
        if set(labels.values()) != {"positive", "negative", "neutral"}:
            labels = dict(FALLBACK_ID_TO_SENTIMENT)
        return {"tok": tok, "model": model, "labels": labels, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@st.cache_resource(show_spinner=False)
def load_relevance_model(name):
    try:
        from transformers import pipeline

        return pipeline("zero-shot-classification", model=name, device=-1)
    except Exception:  # noqa: BLE001
        return None


def score_pairs(pairs, batch_size=16, on_progress=None):
    """Predict (sentiment, confidence) for each (sentence, aspect) pair."""
    import torch

    bundle = load_sentiment_model(MODEL_ID)
    if bundle.get("error"):
        raise RuntimeError(bundle["error"])
    tok, model, labels = bundle["tok"], bundle["model"], bundle["labels"]

    texts = [build_input_text(s, a) for s, a in pairs]
    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**enc).logits.detach().cpu().numpy().astype("float64")
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        for row in probs:
            k = int(row.argmax())
            results.append((labels[k], float(row[k])))
        if on_progress:
            on_progress(min(start + batch_size, len(texts)), len(texts))
    return results


def check_relevance(text):
    """Returns (is_review, review_score, method)."""
    if len(text.split()) < 4:
        return False, 0.0, "length"
    pipe = load_relevance_model(RELEVANCE_MODEL)
    if pipe is None:
        has_aspect = any(detect_aspects(s) for s in split_sentences(text))
        return has_aspect, 1.0 if has_aspect else 0.0, "keywords"
    out = pipe(text[:600], RELEVANCE_LABELS, truncation=True)
    scores = dict(zip(out["labels"], out["scores"]))
    review_score = float(scores.get(RELEVANCE_LABELS[0], 0.0))
    ok = out["labels"][0] == RELEVANCE_LABELS[0] and out["scores"][0] >= RELEVANCE_THRESHOLD
    return ok, review_score, "model"


def flag_for(sentiment, confidence):
    if sentiment == "neutral":
        return "Neutral: check"
    if confidence < LOW_CONFIDENCE:
        return "Low confidence"
    return ""


@st.cache_data(max_entries=256, show_spinner=False)
def analyse_review(text, model_id):
    ok, review_score, method = check_relevance(text)
    sentences = split_sentences(text)
    if not ok:
        return {"status": "rejected", "score": review_score, "method": method, "sentences": sentences}

    meta, pairs = [], []
    for i, sent in enumerate(sentences):
        for asp in detect_aspects(sent):
            pairs.append((sent, asp))
            meta.append({"sentence_no": i + 1, "sentence": sent, "aspect": asp,
                         "matched_words": ", ".join(matched_keywords(sent, asp))})
    if not pairs:
        return {"status": "no_aspect", "score": review_score, "method": method, "sentences": sentences}

    preds = score_pairs(pairs)
    df = pd.DataFrame(meta)
    df["sentiment"] = [p[0] for p in preds]
    df["confidence"] = [round(p[1], 3) for p in preds]
    df["check"] = [flag_for(s, c) for s, c in preds]
    return {"status": "ok", "score": review_score, "method": method, "sentences": sentences, "df": df}


# =====================================================================
# DATA
# =====================================================================
FALLBACK_RANKING = pd.DataFrame({
    "aspect_group": ["Room quality", "Overall", "Food & Beverage", "Value for money",
                     "Staff & Service", "Facilities", "Location"],
    "reviews_with_negative_mentions": [16678, 12157, 8359, 7874, 6661, 5754, 3550],
    "pct_of_reviews": [53.4, 38.9, 26.8, 25.2, 21.3, 18.4, 11.4],
    "priority_rank": [1, 2, 3, 4, 5, 6, 7],
})


@st.cache_data(show_spinner=False)
def load_ranking():
    try:
        df = pd.read_csv(RANKING_CSV)
        source = "file"
    except Exception:  # noqa: BLE001
        return FALLBACK_RANKING.copy(), "built-in"

    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {"aspect": "aspect_group", "negative_count": "reviews_with_negative_mentions",
               "pct_reviews": "pct_of_reviews", "pct": "pct_of_reviews", "priority": "priority_rank"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    needed = {"aspect_group", "reviews_with_negative_mentions", "pct_of_reviews"}
    if not needed.issubset(df.columns):
        return FALLBACK_RANKING.copy(), "built-in"
    df = df.sort_values("pct_of_reviews", ascending=False).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)
    return df[["aspect_group", "reviews_with_negative_mentions", "pct_of_reviews", "priority_rank"]], source


def tier(pct):
    if pct >= 40:
        return "Critical"
    if pct >= 20:
        return "High"
    return "Watch"


def build_ranking(pred_df, n_reviews):
    neg = pred_df[pred_df["sentiment"] == "negative"].groupby("aspect")["review_no"].nunique()
    out = pd.DataFrame({"aspect_group": ASPECTS})
    out["reviews_with_negative_mentions"] = out["aspect_group"].map(neg).fillna(0).astype(int)
    out["pct_of_reviews"] = (100 * out["reviews_with_negative_mentions"] / max(n_reviews, 1)).round(1)
    out = out.sort_values("pct_of_reviews", ascending=False).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


MODEL_RESULTS = pd.DataFrame({
    "Model": ["RoBERTa-base", "BERT-base", "BART-large-MNLI"],
    "Approach": ["Fine-tuned", "Fine-tuned", "Zero-shot"],
    "Accuracy": [0.923, 0.908, 0.917],
    "Macro-F1": [0.721, 0.667, 0.664],
})
MCNEMAR = pd.DataFrame({
    "Comparison": ["RoBERTa vs BERT", "RoBERTa vs BART (zero-shot)", "BERT vs BART (zero-shot)"],
    "p-value": [0.019, 0.448, 0.155],
})
MCNEMAR["Result at α = 0.05"] = np.where(MCNEMAR["p-value"] < 0.05, "Significant", "Not significant")


# =====================================================================
# STYLE
# =====================================================================
CSS = f"""
<style>
.block-container {{ max-width: 1180px; padding-top: 5rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ letter-spacing: 0.01em; }}

.rr-mast {{ display:flex; align-items:flex-end; justify-content:space-between; gap:2rem;
            border-bottom:1px solid {HAIR}; padding-bottom:1.1rem; margin-bottom:0.6rem; flex-wrap:wrap; }}
.rr-brand {{ font-family:'Marcellus', Georgia, serif; font-size:2.7rem; line-height:1; color:{INK}; }}
.rr-brand small {{ display:block; font-family:{BODY_FONT}; font-size:0.95rem; color:{MUTED};
                   margin-top:0.55rem; letter-spacing:0; max-width:62ch; line-height:1.45; }}
.rr-key {{ width:54px; height:54px; flex:none; }}

.rr-lead {{ font-family:'Marcellus', Georgia, serif; font-size:2.05rem; line-height:1.18; color:{INK};
            margin:0.4rem 0 0.35rem; max-width:30ch; }}
.rr-sub {{ color:{MUTED}; font-size:1.02rem; max-width:68ch; margin:0 0 1.2rem; line-height:1.55; }}
.rr-facts {{ display:flex; gap:2.6rem; flex-wrap:wrap; padding:0.9rem 0 1.1rem;
             border-top:1px solid {HAIR}; border-bottom:1px solid {HAIR}; margin-bottom:1.6rem; }}
.rr-fact b {{ display:block; font-size:1.35rem; color:{INK}; font-weight:600; }}
.rr-fact span {{ color:{MUTED}; font-size:0.86rem; }}

.rr-h {{ font-family:'Marcellus', Georgia, serif; font-size:1.3rem; color:{INK}; margin:0 0 0.3rem; }}
.rr-note {{ color:{MUTED}; font-size:0.86rem; margin:0 0 0.8rem; line-height:1.5; }}

.rr-rank {{ display:grid; grid-template-columns:50px 1fr; gap:16px; align-items:center;
            padding:13px 0; border-bottom:1px solid {HAIR}; }}
.rr-rank:last-child {{ border-bottom:none; }}
.rr-plate {{ width:46px; height:46px; border-radius:9px; display:flex; align-items:center; justify-content:center;
             font-family:'Marcellus', Georgia, serif; font-size:1.45rem; color:#FFF8EA;
             background:linear-gradient(150deg,#CDA96C 0%,{BRASS} 52%,#7E5B2B 100%);
             box-shadow: inset 0 1px 0 rgba(255,255,255,.45), inset 0 -2px 0 rgba(0,0,0,.18); }}
.rr-row1 {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; }}
.rr-name {{ font-weight:600; color:{INK}; font-size:1.02rem; }}
.rr-pct {{ font-weight:600; color:{INK}; font-variant-numeric:tabular-nums; font-size:1.02rem; }}
.rr-bar {{ height:7px; background:#E9EAE5; border-radius:4px; margin:7px 0 6px; overflow:hidden; }}
.rr-bar span {{ display:block; height:100%; border-radius:4px; }}
.rr-meta {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:0.82rem; gap:1rem; }}
.rr-tier {{ font-weight:600; }}
.rr-delta-up {{ color:#B3261E; font-weight:600; }}
.rr-delta-down {{ color:#2F7A55; font-weight:600; }}

.rr-step {{ border-left:3px solid {BRASS}; padding:0.1rem 0 0.1rem 0.9rem; }}
.rr-step b {{ color:{INK}; }}
.rr-step p {{ color:{MUTED}; font-size:0.9rem; margin:0.3rem 0 0; line-height:1.5; }}

.rr-verdict {{ font-size:1.12rem; color:{INK}; margin:0.3rem 0 1rem; }}
.rr-chips {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:1.3rem; }}
.rr-chip {{ border:1px solid {HAIR}; background:#fff; border-radius:10px; padding:9px 14px; min-width:150px; }}
.rr-chip .a {{ font-size:0.82rem; color:{MUTED}; }}
.rr-chip .s {{ font-weight:700; font-size:1rem; text-transform:capitalize; }}

.rr-read {{ background:#fff; border:1px solid {HAIR}; border-radius:12px; padding:1.1rem 1.3rem;
            font-size:1.04rem; line-height:2.05; color:{INK}; max-width:80ch; }}
.rr-s-positive {{ background:linear-gradient(transparent 58%, rgba(47,122,85,.20) 58%); }}
.rr-s-negative {{ background:linear-gradient(transparent 58%, rgba(179,38,30,.20) 58%); }}
.rr-s-neutral  {{ background:linear-gradient(transparent 58%, rgba(124,133,145,.25) 58%); }}
.rr-s-mixed    {{ background:linear-gradient(transparent 58%, rgba(183,121,31,.25) 58%); }}
.rr-s-none     {{ color:{MUTED}; }}
.rr-tag {{ display:inline-block; font-size:0.7rem; line-height:1.35; padding:1px 7px; margin:0 3px;
           border-radius:999px; border:1px solid currentColor; vertical-align:2px; font-weight:600; }}
.rr-legend {{ display:flex; gap:1.2rem; flex-wrap:wrap; font-size:0.82rem; color:{MUTED}; margin:0.6rem 0 1.4rem; }}
.rr-legend i {{ display:inline-block; width:18px; height:8px; border-radius:2px; margin-right:6px; vertical-align:1px; }}

.rr-steps {{ counter-reset: step; list-style:none; padding:0 !important; margin:0.5rem 0 0 !important; }}
.rr-steps li {{ counter-increment: step; display:grid; grid-template-columns:40px 1fr; gap:12px;
               padding:0.8rem 0; margin:0 !important; border-bottom:1px solid {HAIR}; }}
.rr-steps li::before {{ content: counter(step); font-family:'Marcellus', Georgia, serif; font-size:1.4rem;
                        color:{BRASS}; }}
.rr-steps b {{ color:{INK}; }}
.rr-steps p {{ margin:0.2rem 0 0; color:{MUTED}; font-size:0.92rem; line-height:1.5; }}

.rr-foot {{ margin-top:3rem; padding-top:1rem; border-top:1px solid {HAIR}; color:{MUTED}; font-size:0.8rem; }}

@media (max-width: 640px) {{
  .rr-brand {{ font-size:2.1rem; }}
  .rr-lead {{ font-size:1.6rem; }}
  .rr-facts {{ gap:1.4rem; }}
  .rr-key {{ display:none; }}
}}
</style>
"""

KEY_SVG = f"""
<svg class="rr-key" viewBox="0 0 54 54" aria-hidden="true">
  <rect x="9" y="3" width="36" height="48" rx="11" fill="none" stroke="{BRASS}" stroke-width="2"/>
  <circle cx="27" cy="13" r="3.5" fill="none" stroke="{BRASS}" stroke-width="2"/>
  <text x="27" y="37" text-anchor="middle" font-family="Marcellus, Georgia, serif" font-size="14" fill="{BRASS}">RR</text>
</svg>
"""


def esc(x):
    return html.escape(str(x))


def masthead():
    st.html(CSS)
    st.html(
        f"""<div class="rr-mast">
              <div class="rr-brand">RoomRead
                <small>Reads hotel reviews sentence by sentence, works out which part of the stay each
                remark is about, and ranks what guests complain about most.</small>
              </div>{KEY_SVG}
            </div>"""
    )


def footer():
    st.html(
        """<div class="rr-foot">RoomRead, research prototype by Nethika Alagarathnam,
        MSc in Data Science and Artificial Intelligence, PGIS, University of Peradeniya.
        Predictions can be wrong: read flagged lines yourself before acting on them.</div>"""
    )


def plotly_base(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, size=13, color=INK),
        hoverlabel=dict(font_family=BODY_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    return fig


def show_chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def ranking_rows(df, benchmark=None):
    rows = []
    for _, r in df.iterrows():
        pct = float(r["pct_of_reviews"])
        t = tier(pct)
        delta = ""
        if benchmark is not None and r["aspect_group"] in benchmark:
            d = pct - benchmark[r["aspect_group"]]
            cls = "rr-delta-up" if d > 0 else "rr-delta-down"
            delta = f'<span class="{cls}">{d:+.1f} pts vs benchmark</span>'
        rows.append(f"""
        <div class="rr-rank">
          <div class="rr-plate">{int(r['priority_rank'])}</div>
          <div>
            <div class="rr-row1"><span class="rr-name">{esc(r['aspect_group'])}</span>
                 <span class="rr-pct">{pct:.1f}%</span></div>
            <div class="rr-bar"><span style="width:{min(pct, 100):.1f}%;background:{TIER_COLOURS[t]}"></span></div>
            <div class="rr-meta"><span>{int(r['reviews_with_negative_mentions']):,} reviews with a complaint</span>
                 <span>{delta or f'<span class="rr-tier" style="color:{TIER_COLOURS[t]}">{t}</span>'}</span></div>
          </div>
        </div>""")
    st.html("<div>" + "".join(rows) + "</div>")


def pareto_chart(df):
    total = df["reviews_with_negative_mentions"].sum()
    share = 100 * df["reviews_with_negative_mentions"] / total
    cum = share.cumsum()
    fig = go.Figure()
    fig.add_bar(x=df["aspect_group"], y=share, name="Share of negative mentions",
                marker_color=[TIER_COLOURS[tier(p)] for p in df["pct_of_reviews"]],
                hovertemplate="%{x}<br>%{y:.1f}% of negative mentions<extra></extra>")
    fig.add_scatter(x=df["aspect_group"], y=cum, name="Cumulative share", mode="lines+markers",
                    line=dict(color=NIGHT, width=2.5), marker=dict(size=7), yaxis="y2",
                    hovertemplate="Top %{x}: %{y:.1f}% cumulative<extra></extra>")
    fig.update_layout(
        yaxis=dict(title="% of negative mentions", gridcolor="#ECECE8", ticksuffix="%"),
        yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False),
        xaxis=dict(tickangle=-30),
        bargap=0.35,
    )
    return plotly_base(fig, 390)


def ranking_table(df, key):
    st.dataframe(
        df.rename(columns={"aspect_group": "Service area",
                           "reviews_with_negative_mentions": "Reviews with a complaint",
                           "pct_of_reviews": "% of reviews", "priority_rank": "Rank"}),
        hide_index=True, width="stretch", key=key,
        column_order=["Rank", "Service area", "Reviews with a complaint", "% of reviews"],
        column_config={"% of reviews": st.column_config.ProgressColumn(
            "% of reviews", format="%.1f%%", min_value=0, max_value=100)},
    )


# =====================================================================
# PAGE: PRIORITIES
# =====================================================================
def page_priorities():
    df, source = load_ranking()
    top = df.iloc[0]
    total_neg = df["reviews_with_negative_mentions"].sum()
    top3_share = 100 * df["reviews_with_negative_mentions"].head(3).sum() / total_neg

    st.html(f"""
      <div class="rr-lead">{esc(top['aspect_group'])} is the first thing to fix.</div>
      <p class="rr-sub">{top['pct_of_reviews']:.1f}% of {BENCHMARK_REVIEWS:,} guest reviews contain at least
      one negative remark about {esc(top['aspect_group'].lower())}. Service areas are ranked below by the share
      of reviews that complain about them.</p>
      <div class="rr-facts">
        <div class="rr-fact"><b>{BENCHMARK_REVIEWS:,}</b><span>reviews analysed</span></div>
        <div class="rr-fact"><b>{len(df)}</b><span>service areas tracked</span></div>
        <div class="rr-fact"><b>{top3_share:.1f}%</b><span>of all complaints fall in the top three areas</span></div>
        <div class="rr-fact"><b>{total_neg:,}</b><span>negative mentions in total</span></div>
      </div>""")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.html('<div class="rr-h">Priority ranking</div>'
                '<p class="rr-note">Share of all reviews with at least one negative mention of the area.</p>')
        ranking_rows(df)
        st.html(f"""<p class="rr-note" style="margin-top:.8rem">
            <b style="color:{TIER_COLOURS['Critical']}">Critical</b> 40% or more of reviews &nbsp;
            <b style="color:{TIER_COLOURS['High']}">High</b> 20 to 40% &nbsp;
            <b style="color:{TIER_COLOURS['Watch']}">Watch</b> under 20%</p>""")
    with right:
        st.html('<div class="rr-h">Where complaints concentrate</div>'
                f'<p class="rr-note">Each bar is the area\'s share of all {total_neg:,} negative mentions. '
                'The line shows how quickly the top areas add up.</p>')
        show_chart(pareto_chart(df))

    st.html('<div class="rr-h" style="margin-top:1.2rem">Suggested first steps</div>'
            '<p class="rr-note">Starting points for the three highest-ranked areas. '
            'These are general suggestions, not model output.</p>')
    cols = st.columns(3, gap="large")
    for col, (_, r) in zip(cols, df.head(3).iterrows()):
        with col:
            st.html(f"""<div class="rr-step"><b>{int(r['priority_rank'])}. {esc(r['aspect_group'])}</b>
                        <p>{esc(ASPECT_ACTIONS.get(r['aspect_group'], ''))}</p></div>""")

    st.write("")
    with st.expander("Full ranking table and download"):
        ranking_table(df, key="bench_table")
        st.download_button("Download ranking as CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="roomread_priority_ranking.csv", mime="text/csv",
                           icon=":material/download:")
        if source == "built-in":
            st.caption("Showing the built-in figures. Add aspect_priority_ranking.csv to the repo to update them.")


# =====================================================================
# PAGE: CHECK A REVIEW
# =====================================================================
def _load_sample():
    choice = st.session_state.get("sample_choice")
    if choice:
        st.session_state["review_text"] = SAMPLES[choice]
        st.session_state.pop("single_result", None)


def _clear_review():
    st.session_state["review_text"] = ""
    st.session_state["sample_choice"] = None
    st.session_state.pop("single_result", None)


def model_unavailable(error):
    st.error(
        f"The sentiment model couldn't be loaded from `{MODEL_ID}`. "
        "Check that the model folder is in the repo (or that MODEL_ID in the app secrets points to it), "
        "then select Retry.\n\n"
        f"Details: {error}",
        icon=":material/error:",
    )
    if st.button("Retry loading the model", icon=":material/refresh:"):
        load_sentiment_model.clear()
        st.rerun()


def ensure_sentiment_model():
    with st.spinner("Loading the model. The first visit after the app wakes up takes about a minute."):
        bundle = load_sentiment_model(MODEL_ID)
    if bundle.get("error"):
        model_unavailable(bundle["error"])
        return False
    return True


def render_single(res):
    if res["status"] == "rejected":
        why = ("It's too short to analyse. Paste at least one full sentence about the stay."
               if res["method"] == "length" else
               f"This doesn't read like a hotel review (review score {res['score']:.2f}; "
               f"{RELEVANCE_THRESHOLD:.2f} or more is needed). Paste a guest's description of their stay.")
        st.warning(why, icon=":material/do_not_disturb_on:")
        return
    if res["status"] == "no_aspect":
        st.info("No tracked service area is mentioned. RoomRead looks for remarks about rooms, staff, "
                "food, location, price, facilities and the stay overall.", icon=":material/search_off:")
        return

    df = res["df"]
    counts = df["sentiment"].value_counts()
    n_flag = int((df["check"] != "").sum())
    n_areas = df["aspect"].nunique()
    parts = [f"{counts.get(s, 0)} {s}" for s in ("negative", "neutral", "positive") if counts.get(s, 0)]
    flag_txt = f", {n_flag} to check by hand" if n_flag else ""
    st.html(f'<div class="rr-verdict"><b>{len(df)} mention{"s" if len(df) != 1 else ""}</b> across '
            f'{n_areas} service area{"s" if n_areas != 1 else ""}: {", ".join(parts)}{flag_txt}.</div>')

    chips = []
    for asp in ASPECTS:
        sub = df[df["aspect"] == asp]
        if sub.empty:
            continue
        kinds = set(sub["sentiment"])
        label = kinds.pop() if len(kinds) == 1 else "mixed"
        chips.append(f"""<div class="rr-chip"><div class="a">{esc(asp)}</div>
            <div class="s" style="color:{SENT_COLOURS[label]}">{label}</div></div>""")
    st.html('<div class="rr-chips">' + "".join(chips) + "</div>")

    st.html('<div class="rr-h">The review, line by line</div>')
    spans = []
    for i, sent in enumerate(res["sentences"], start=1):
        sub = df[df["sentence_no"] == i]
        if sub.empty:
            spans.append(f'<span class="rr-s-none">{esc(sent)}</span> ')
            continue
        kinds = set(sub["sentiment"])
        cls = f"rr-s-{next(iter(kinds))}" if len(kinds) == 1 else "rr-s-mixed"
        tags = "".join(
            f'<span class="rr-tag" style="color:{SENT_COLOURS[r.sentiment]}">{esc(r.aspect)}</span>'
            for r in sub.itertuples())
        spans.append(f'<span class="{cls}">{esc(sent)}</span>{tags} ')
    legend = "".join(f'<span><i style="background:{SENT_COLOURS[k]};opacity:.55"></i>{k.capitalize()}</span>'
                     for k in ("negative", "neutral", "positive", "mixed"))
    st.html(f'<div class="rr-read">{"".join(spans)}</div>'
            f'<div class="rr-legend">{legend}<span>Grey text: no tracked area mentioned</span></div>')

    st.html('<div class="rr-h">Details</div>')
    table = df.rename(columns={"sentence_no": "#", "sentence": "Sentence", "aspect": "Service area",
                               "sentiment": "Sentiment", "confidence": "Confidence",
                               "check": "Check", "matched_words": "Matched words"})
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_order=["#", "Sentence", "Service area", "Sentiment", "Confidence", "Check", "Matched words"],
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Sentence": st.column_config.TextColumn(width="large"),
            "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        },
    )
    st.caption(f"Neutral predictions and any prediction under {LOW_CONFIDENCE:.0%} confidence are marked for a "
               "manual check, because neutral was the least reliable class in testing. "
               + ("Relevance was checked with the zero-shot model."
                  if res["method"] == "model" else
                  "The relevance model wasn't available, so relevance was judged by keyword matches."))
    st.download_button("Download this analysis as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="roomread_review_analysis.csv", mime="text/csv",
                       icon=":material/download:")


def page_check():
    st.html('<div class="rr-lead">Check a review</div>'
            '<p class="rr-sub">Paste a guest review. RoomRead splits it into sentences, finds the service areas '
            'each one mentions, and predicts whether the guest is positive, negative or neutral about each.</p>')

    st.session_state.setdefault("review_text", "")
    st.pills("Try an example", list(SAMPLES), key="sample_choice", on_change=_load_sample)
    st.text_area("Guest review", key="review_text", height=190,
                 placeholder="The room was spotless but breakfast was cold…")
    c1, c2, _ = st.columns([1.2, 0.8, 4])
    run = c1.button("Analyse review", type="primary", icon=":material/manage_search:", width="stretch")
    c2.button("Clear", on_click=_clear_review, width="stretch")

    if not ensure_sentiment_model():
        return

    text = st.session_state["review_text"].strip()
    if run:
        if not text:
            st.info("Paste a review first, or pick one of the examples above.", icon=":material/edit_note:")
            return
        with st.spinner("Reading the review…"):
            try:
                st.session_state["single_result"] = analyse_review(text, MODEL_ID)
            except Exception as exc:  # noqa: BLE001
                st.error(f"The analysis stopped with an error: {type(exc).__name__}: {exc}")
                return

    if "single_result" in st.session_state:
        st.divider()
        render_single(st.session_state["single_result"])


# =====================================================================
# PAGE: ANALYSE A FILE
# =====================================================================
def read_upload(uploaded):
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".txt"):
        lines = [ln.strip() for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        return pd.DataFrame({"review": lines})
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Couldn't read the file's text encoding. Save it as UTF-8 CSV and try again.")


def guess_text_column(df):
    text_cols = [c for c in df.columns
                 if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    if not text_cols:
        return None
    named = [c for c in text_cols if "review" in str(c).lower() or "text" in str(c).lower()]
    if named:
        return named[0]
    return max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())


def run_file_analysis(reviews, use_gate):
    status = st.progress(0.0, text="Preparing…")
    kept, rejected = [], 0
    if use_gate:
        for i, rv in enumerate(reviews):
            ok, _, _ = check_relevance(rv)
            if ok:
                kept.append(rv)
            else:
                rejected += 1
            status.progress(0.3 * (i + 1) / len(reviews), text=f"Checking relevance: {i + 1} of {len(reviews)}")
    else:
        kept = [rv for rv in reviews if len(rv.split()) >= 4]
        rejected = len(reviews) - len(kept)

    meta, pairs = [], []
    for r_no, rv in enumerate(kept, start=1):
        for s_no, sent in enumerate(split_sentences(rv), start=1):
            for asp in detect_aspects(sent):
                pairs.append((sent, asp))
                meta.append({"review_no": r_no, "sentence_no": s_no, "sentence": sent, "aspect": asp})

    if not pairs:
        status.empty()
        return {"n_reviews": len(kept), "rejected": rejected, "pred": pd.DataFrame(), "ranking": None}

    base = 0.3 if use_gate else 0.0

    def progress(done, total):
        status.progress(base + (1 - base) * done / total, text=f"Scoring mentions: {done:,} of {total:,}")

    preds = score_pairs(pairs, batch_size=32, on_progress=progress)
    status.empty()
    pred = pd.DataFrame(meta)
    pred["sentiment"] = [p[0] for p in preds]
    pred["confidence"] = [round(p[1], 3) for p in preds]
    pred["check"] = [flag_for(s, c) for s, c in preds]
    return {"n_reviews": len(kept), "rejected": rejected, "pred": pred,
            "ranking": build_ranking(pred, len(kept))}


def comparison_chart(yours, bench):
    merged = yours.merge(bench[["aspect_group", "pct_of_reviews"]], on="aspect_group",
                         suffixes=("_yours", "_bench"), how="left")
    merged = merged.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=merged["aspect_group"], x=merged["pct_of_reviews_bench"], orientation="h",
                name=f"Benchmark ({BENCHMARK_REVIEWS:,} reviews)", marker_color="#C9CCC6",
                hovertemplate="%{y}: %{x:.1f}% (benchmark)<extra></extra>")
    fig.add_bar(y=merged["aspect_group"], x=merged["pct_of_reviews_yours"], orientation="h",
                name="Your reviews", marker_color=NIGHT,
                hovertemplate="%{y}: %{x:.1f}% (yours)<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08,
                      xaxis=dict(title="% of reviews with a complaint", ticksuffix="%", gridcolor="#ECECE8"))
    return plotly_base(fig, 420)


def page_file():
    st.html('<div class="rr-lead">Analyse a file</div>'
            '<p class="rr-sub">Upload your own reviews as CSV, Excel or a text file with one review per line. '
            'RoomRead builds the same priority ranking for them and sets it against the '
            f'{BENCHMARK_REVIEWS:,}-review benchmark.</p>')

    up = st.file_uploader("Reviews file", type=["csv", "xlsx", "txt"])
    if up is None:
        st.caption("Tip: start with 100 reviews. The free server scores a few mentions per second, "
                   "so a few hundred reviews take a few minutes.")
        st.session_state.pop("file_result", None)
        return

    try:
        data = read_upload(up)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The file couldn't be read: {exc}")
        return
    if data.empty:
        st.warning("The file has no rows.")
        return

    guess = guess_text_column(data)
    if guess is None:
        st.error("No text column found. The file needs a column containing the review text.")
        return

    c1, c2, c3 = st.columns([1.3, 1, 1.2], gap="medium")
    col = c1.selectbox("Column with the review text", list(data.columns),
                       index=list(data.columns).index(guess))
    available = int(data[col].dropna().astype(str).str.strip().ne("").sum())
    limit = c2.number_input("Reviews to analyse", min_value=1, max_value=min(available, MAX_FILE_REVIEWS),
                            value=min(available, 100), step=10)
    use_gate = c3.toggle("Skip text that isn't a review", value=False,
                         help="Runs the zero-shot relevance check on every row. Slower; use it when the "
                              "file may contain non-review text.")
    st.caption(f"{available:,} non-empty rows found in “{col}”. At most {MAX_FILE_REVIEWS} are analysed per run.")

    with st.expander("Preview the first rows"):
        st.dataframe(data.head(8), hide_index=True, width="stretch")

    run_key = (up.name, up.size, col, int(limit), use_gate)
    if st.button("Analyse file", type="primary", icon=":material/play_arrow:"):
        if not ensure_sentiment_model():
            return
        reviews = data[col].dropna().astype(str).str.strip()
        reviews = reviews[reviews.ne("")].head(int(limit)).tolist()
        try:
            st.session_state["file_result"] = {"key": run_key, **run_file_analysis(reviews, use_gate)}
        except Exception as exc:  # noqa: BLE001
            st.error(f"The analysis stopped with an error: {type(exc).__name__}: {exc}")
            return

    res = st.session_state.get("file_result")
    if not res or res["key"] != run_key:
        return

    st.divider()
    if res["ranking"] is None:
        st.info("None of the analysed reviews mention a tracked service area.")
        return

    rk, pred, n = res["ranking"], res["pred"], res["n_reviews"]
    bench, _ = load_ranking()
    bench_map = dict(zip(bench["aspect_group"], bench["pct_of_reviews"]))
    top = rk.iloc[0]
    skipped = f" {res['rejected']} rows were skipped as too short or not a review." if res["rejected"] else ""
    st.html(f"""<div class="rr-lead" style="font-size:1.7rem">In your reviews, {esc(top['aspect_group'])}
                comes first.</div>
                <p class="rr-sub">{top['pct_of_reviews']:.1f}% of {n:,} reviews complain about it, against
                {bench_map.get(top['aspect_group'], float('nan')):.1f}% in the benchmark.
                RoomRead scored {len(pred):,} mentions.{esc(skipped)}</p>""")

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.html('<div class="rr-h">Your priority ranking</div>'
                '<p class="rr-note">Share of your reviews with at least one negative mention, and the '
                'difference from the benchmark in percentage points.</p>')
        ranking_rows(rk, benchmark=bench_map)
    with right:
        st.html('<div class="rr-h">Your reviews against the benchmark</div>'
                '<p class="rr-note">A dark bar longer than the grey one means guests complain about '
                'that area more often than usual.</p>')
        show_chart(comparison_chart(rk, bench))

    n_flag = int((pred["check"] != "").sum())
    with st.expander(f"All {len(pred):,} scored mentions ({n_flag:,} marked to check)"):
        st.dataframe(pred, hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", format="%.2f", min_value=0, max_value=1)})
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download ranking (CSV)", rk.to_csv(index=False).encode("utf-8"),
                       file_name="roomread_my_ranking.csv", mime="text/csv", icon=":material/download:",
                       width="stretch")
    d2.download_button("Download all mentions (CSV)", pred.to_csv(index=False).encode("utf-8"),
                       file_name="roomread_my_mentions.csv", mime="text/csv", icon=":material/download:",
                       width="stretch")


# =====================================================================
# PAGE: MODEL EVIDENCE
# =====================================================================
def dumbbell_chart():
    df = MODEL_RESULTS.iloc[::-1]
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_shape(type="line", x0=r["Macro-F1"], x1=r["Accuracy"], y0=r["Model"], y1=r["Model"],
                      line=dict(color="#C9CCC6", width=4), layer="below")
    fig.add_scatter(x=df["Macro-F1"], y=df["Model"], mode="markers+text", name="Macro-F1",
                    marker=dict(size=16, color=BRASS), text=[f"{v:.3f}" for v in df["Macro-F1"]],
                    textposition="bottom center", hovertemplate="%{y}: macro-F1 %{x:.3f}<extra></extra>")
    fig.add_scatter(x=df["Accuracy"], y=df["Model"], mode="markers+text", name="Accuracy",
                    marker=dict(size=16, color=NIGHT), text=[f"{v:.3f}" for v in df["Accuracy"]],
                    textposition="bottom center", hovertemplate="%{y}: accuracy %{x:.3f}<extra></extra>")
    fig.update_layout(xaxis=dict(range=[0.6, 0.96], gridcolor="#ECECE8", title="Score on the held-out test set"),
                      yaxis=dict(automargin=True))
    return plotly_base(fig, 330)


def page_model():
    best = MODEL_RESULTS.iloc[0]
    st.html(f"""<div class="rr-lead">Fine-tuned RoBERTa reads sentiment best.</div>
        <p class="rr-sub">Three models were tested on the human-labelled OATS-Hotels test set of 1,692 examples.
        RoBERTa reached a macro-F1 of {best['Macro-F1']:.3f} and accuracy of {best['Accuracy']:.3f}.
        It is the model behind this app.</p>""")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.html('<div class="rr-h">Accuracy and macro-F1 by model</div>')
        show_chart(dumbbell_chart())
    with right:
        st.html(f"""<div class="rr-h">Why two numbers</div>
          <p class="rr-note" style="font-size:.95rem">Most test examples are positive, so a model can score
          high accuracy while missing many negative and neutral cases. Macro-F1 weighs the three classes
          equally, which makes it the fairer yardstick here. The gap between the two dots is the cost of
          that imbalance.</p>
          <p class="rr-note" style="font-size:.95rem">The neutral class is the weakest point: it had only
          38 test examples. That is why the app marks neutral predictions for a manual check.</p>""")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.html('<div class="rr-h">Are the differences real?</div>'
                '<p class="rr-note">McNemar\'s test on paired predictions over the same test set.</p>')
        st.dataframe(MCNEMAR, hide_index=True, width="stretch",
                     column_config={"p-value": st.column_config.NumberColumn(format="%.3f")})
        st.caption("RoBERTa beats BERT with statistical significance. Its lead over zero-shot BART is not "
                   "significant, so a zero-shot model is a reasonable choice when there is no labelled data.")
    with c2:
        st.html('<div class="rr-h">Training setup</div>'
                '<p class="rr-note">Class weights offset the imbalance during fine-tuning.</p>')
        st.dataframe(pd.DataFrame({
            "Item": ["Training examples", "Validation examples", "Test examples",
                     "Class weight: positive", "Class weight: negative", "Class weight: neutral"],
            "Value": ["8,009", "1,767", "1,692", "0.402", "2.236", "14.998"],
        }), hide_index=True, width="stretch")

    st.html('<div class="rr-h" style="margin-top:1rem">Full results</div>')
    st.dataframe(MODEL_RESULTS, hide_index=True, width="stretch",
                 column_config={"Accuracy": st.column_config.NumberColumn(format="%.3f"),
                                "Macro-F1": st.column_config.NumberColumn(format="%.3f")})


# =====================================================================
# PAGE: ABOUT
# =====================================================================
def page_about():
    st.html(f"""<div class="rr-lead">How RoomRead works</div>
      <p class="rr-sub">A review goes through five steps. The same pipeline produced the benchmark ranking from
      {BENCHMARK_REVIEWS:,} unlabelled reviews.</p>
      <ol class="rr-steps">
        <li><div><b>Split into sentences</b><p>Each sentence is analysed on its own, so one review can praise the
            room and criticise the breakfast.</p></div></li>
        <li><div><b>Check it is a review</b><p>A zero-shot classifier (DistilBART-MNLI) rejects text that isn't
            about a hotel stay. The review label needs a score of {RELEVANCE_THRESHOLD:.2f} or more.</p></div></li>
        <li><div><b>Find the service areas</b><p>Keyword matching assigns each sentence to one or more of seven
            areas: {esc(', '.join(ASPECTS))}.</p></div></li>
        <li><div><b>Predict sentiment for each area</b><p>A fine-tuned RoBERTa-base model reads the sentence
            together with the area name and predicts positive, negative or neutral.</p></div></li>
        <li><div><b>Rank the areas</b><p>For a set of reviews, each area's score is the share of reviews with at
            least one negative mention of it.</p></div></li>
      </ol>""")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.html(f"""<div class="rr-h" style="margin-top:1.4rem">Data</div>
          <p class="rr-note" style="font-size:.95rem">The sentiment model was trained and evaluated on the
          human-annotated OATS-Hotels dataset (Chebolu et al., 2024). The benchmark ranking comes from
          applying it to {BENCHMARK_REVIEWS:,} unlabelled hotel reviews.</p>""")
    with c2:
        st.html("""<div class="rr-h" style="margin-top:1.4rem">Limitations</div>
          <p class="rr-note" style="font-size:.95rem">Keyword matching misses areas described in other words
          and can tag the wrong area. Neutral predictions are the least reliable. The model reads English only.
          The benchmark mixes many hotels, so it describes typical guests rather than any single property.</p>""")

    with st.expander("Technical details"):
        st.markdown(
            f"- Sentiment model: `{MODEL_ID}`\n"
            f"- Relevance model: `{RELEVANCE_MODEL}` (threshold {RELEVANCE_THRESHOLD})\n"
            f"- Input format: `sentence [SEP] aspect`, max {MAX_LEN} tokens\n"
            f"- Manual-check flag: neutral, or confidence below {LOW_CONFIDENCE:.2f}\n"
            "- Hosting: Streamlit Community Cloud; the app sleeps when idle and wakes on the next visit."
        )


# =====================================================================
# NAVIGATION
# =====================================================================
pages = [
    st.Page(page_priorities, title="Priorities", icon=":material/leaderboard:", default=True),
    st.Page(page_check, title="Check a review", icon=":material/rate_review:", url_path="check"),
    st.Page(page_file, title="Analyse a file", icon=":material/upload_file:", url_path="file"),
    st.Page(page_model, title="Model evidence", icon=":material/analytics:", url_path="model"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages, position="top")
masthead()
nav.run()
footer()
