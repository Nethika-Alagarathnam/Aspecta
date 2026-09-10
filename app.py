"""
RoomRead — aspect-based sentiment analysis for hotel reviews.

Pages
  Dashboard        What guests complain about most, from 31,219 reviews
  Check a review   Analyse one pasted review sentence by sentence
  Upload reviews   Upload a CSV / Excel / TXT file and get your own ranking
  Model results    Test-set scores for RoBERTa, BERT and zero-shot BART
  About            How it works, data, limitations, author

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
    page_title="AspectaRoomRead | Hotel review priorities",
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
    "Room quality": "Check rooms before guests arrive: cleaning, beds, bathrooms, noise and air-conditioning.",
    "Overall": "Read a few of these reviews in full. They usually describe a general letdown rather than one problem.",
    "Food & Beverage": "Look at breakfast during busy hours. Is the food warm, and is the buffet refilled quickly?",
    "Value for money": "Make it clear what the price includes, both when guests book and when they check in.",
    "Staff & Service": "Coach front-desk staff on handling complaints and keep track of how fast requests are solved.",
    "Facilities": "Keep a list of Wi-Fi, lift, parking and pool problems and fix each one by a set date.",
    "Location": "Describe the location honestly online and give guests clear directions or transport options.",
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
TEAL = "#008080"
BLUE_GREEN = "#088F8F"
NAVY = "#000080"
MIDNIGHT = "#191970"
LIGHT_BLUE = "#ADD8E6"
ROBIN = "#96DED1"
TEXT = "#23284A"
MUTED = "#5B6380"
BORDER = "#D5E8EE"
SOFT_BG = "#F2F9FB"
SENT_COLOURS = {"positive": TEAL, "negative": "#D64545", "neutral": "#8A94A6", "mixed": NAVY}
TIER_COLOURS = {"Critical": MIDNIGHT, "High": TEAL, "Watch": ROBIN}
BODY_FONT = "Nunito, 'Segoe UI', Arial, sans-serif"


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
.block-container {{ max-width: 1150px; padding-top: 4.6rem; padding-bottom: 2rem; }}

/* Top bar */
header[data-testid="stHeader"] {{ background: {MIDNIGHT} !important; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] {{ border-radius: 8px; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] span {{ color: #FFFFFF !important; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"]:hover {{ background: rgba(150,222,209,0.18) !important; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"][aria-current="page"] {{ background: {TEAL} !important; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] [data-testid="stMainMenuButton"] {{ color: #FFFFFF !important; }}

header[data-testid="stHeader"] [data-testid="stIconMaterial"] {{ color: #FFFFFF !important; }}
/* Mobile menu (sidebar) */
section[data-testid="stSidebar"] {{ background: {MIDNIGHT} !important; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] span,
section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {TEAL} !important; }}

/* Hero */
.rr-hero {{ background: {TEAL}; color: #fff; border-radius: 16px; padding: 2rem 2.2rem;
            display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; flex-wrap: wrap; }}
.rr-hero h1 {{ color: #fff; font-size: 2rem; font-weight: 800; margin: 0 0 .35rem; padding: 0; }}
.rr-hero p {{ color: #E6F7F5; font-size: 1.05rem; margin: 0; max-width: 58ch; }}
.rr-hero a {{ background: #fff; color: {MIDNIGHT} !important; text-decoration: none; font-weight: 700;
              padding: .7rem 1.3rem; border-radius: 10px; white-space: nowrap; }}
.rr-hero a:hover {{ background: {ROBIN}; }}

/* Page title */
.rr-title h1 {{ color: {MIDNIGHT}; font-size: 1.9rem; font-weight: 800; margin: 0 0 .25rem; padding: 0; }}
.rr-title p {{ color: {MUTED}; font-size: 1.02rem; margin: 0 0 1rem; max-width: 70ch; }}
.rr-h {{ color: {MIDNIGHT}; font-size: 1.2rem; font-weight: 800; margin: .2rem 0 .7rem; }}

/* Cards */
.rr-cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.2rem 0 1.6rem; }}
.rr-card {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.1rem; }}
.rr-card .v {{ color: {MIDNIGHT}; font-size: 1.55rem; font-weight: 800; line-height: 1.2; }}
.rr-card .l {{ color: {MUTED}; font-size: .88rem; margin-top: .2rem; }}
.rr-card.accent {{ background: {SOFT_BG}; border-color: {ROBIN}; }}

/* Top issues list */
.rr-issue {{ display: flex; gap: 14px; align-items: flex-start; background: #fff; border: 1px solid {BORDER};
             border-radius: 12px; padding: .95rem 1rem; margin-bottom: 10px; }}
.rr-num {{ flex: none; width: 34px; height: 34px; border-radius: 50%; background: {TEAL}; color: #fff;
           font-weight: 800; display: flex; align-items: center; justify-content: center; }}
.rr-issue b {{ color: {MIDNIGHT}; }}
.rr-issue .pct {{ color: {TEAL}; font-weight: 800; margin-left: .35rem; }}
.rr-issue p {{ color: {MUTED}; font-size: .9rem; margin: .25rem 0 0; line-height: 1.45; }}

/* Review result */
.rr-summary {{ background: {SOFT_BG}; border-left: 5px solid {TEAL}; border-radius: 8px; padding: .8rem 1rem;
               color: {TEXT}; font-size: 1.05rem; margin: .4rem 0 1rem; }}
.rr-tiles {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 1.2rem; }}
.rr-tile {{ background: #fff; border: 1px solid {BORDER}; border-top-width: 4px; border-radius: 10px;
            padding: .6rem .9rem; min-width: 145px; }}
.rr-tile .a {{ color: {MUTED}; font-size: .85rem; }}
.rr-tile .s {{ font-weight: 800; text-transform: capitalize; }}
.rr-read {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.2rem;
            font-size: 1.03rem; line-height: 2; color: {TEXT}; }}
.rr-s-positive {{ background: rgba(0,128,128,.14); border-radius: 4px; padding: 1px 2px; }}
.rr-s-negative {{ background: rgba(214,69,69,.14); border-radius: 4px; padding: 1px 2px; }}
.rr-s-neutral  {{ background: rgba(138,148,166,.18); border-radius: 4px; padding: 1px 2px; }}
.rr-s-mixed    {{ background: rgba(0,0,128,.10); border-radius: 4px; padding: 1px 2px; }}
.rr-s-none     {{ color: {MUTED}; }}
.rr-tag {{ display: inline-block; font-size: .72rem; font-weight: 700; color: #fff; padding: 0 8px;
           margin: 0 4px; border-radius: 999px; line-height: 1.6; vertical-align: 1px; }}
.rr-legend {{ display: flex; gap: 1.1rem; flex-wrap: wrap; font-size: .85rem; color: {MUTED}; margin: .6rem 0 1.2rem; }}
.rr-legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }}

/* Steps (About) */
.rr-step {{ display: flex; gap: 14px; align-items: flex-start; padding: .7rem 0; border-bottom: 1px solid {BORDER}; }}
.rr-step:last-child {{ border-bottom: none; }}
.rr-step b {{ color: {MIDNIGHT}; }}
.rr-step p {{ color: {MUTED}; margin: .15rem 0 0; font-size: .93rem; }}

.rr-foot {{ margin-top: 2.5rem; background: {MIDNIGHT}; color: #DCE4F5; border-radius: 12px;
            padding: 1rem 1.3rem; font-size: .85rem; display: flex; justify-content: space-between;
            flex-wrap: wrap; gap: .5rem; }}
.rr-foot b {{ color: #fff; }}

@media (max-width: 800px) {{
  .rr-cards {{ grid-template-columns: repeat(2, 1fr); }}
  .rr-hero {{ padding: 1.4rem; }}
  .rr-hero h1 {{ font-size: 1.6rem; }}
}}
</style>
"""

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="190" height="40" viewBox="0 0 190 40">
<rect x="3" y="5" width="26" height="31" rx="3" fill="#96DED1"/>
<rect x="8" y="10" width="5" height="5" rx="1" fill="#191970"/><rect x="19" y="10" width="5" height="5" rx="1" fill="#191970"/>
<rect x="8" y="19" width="5" height="5" rx="1" fill="#191970"/><rect x="19" y="19" width="5" height="5" rx="1" fill="#191970"/>
<rect x="13" y="28" width="6" height="8" rx="1" fill="#191970"/>
<text x="38" y="28" font-family="Nunito, Segoe UI, Arial, sans-serif" font-size="22" font-weight="800" fill="#FFFFFF">RoomRead</text>
</svg>"""


def esc(x):
    return html.escape(str(x))


def setup_page():
    import base64
    st.logo("data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode()).decode(), size="large")
    st.html(CSS)


def page_title(title, text):
    st.html(f'<div class="rr-title"><h1>{esc(title)}</h1><p>{text}</p></div>')


def footer():
    st.html("""<div class="rr-foot">
        <span><b>RoomRead</b> &copy; 2026 Nethika Alagarathnam</span>
        <span>MSc in Data Science and Artificial Intelligence, PGIS, University of Peradeniya</span>
      </div>""")


def plotly_base(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, size=13, color=TEXT),
        hoverlabel=dict(font_family=BODY_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    return fig


def show_chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def area_bar_chart(df, benchmark=None):
    d = df.iloc[::-1]
    fig = go.Figure()
    if benchmark is not None:
        fig.add_bar(y=d["aspect_group"], x=[benchmark.get(a, 0) for a in d["aspect_group"]], orientation="h",
                    name=f"All {BENCHMARK_REVIEWS:,} reviews", marker_color=LIGHT_BLUE,
                    hovertemplate="%{y}: %{x:.1f}%<extra>All reviews</extra>")
        fig.add_bar(y=d["aspect_group"], x=d["pct_of_reviews"], orientation="h", name="Your reviews",
                    marker_color=TEAL, hovertemplate="%{y}: %{x:.1f}%<extra>Your reviews</extra>")
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    else:
        fig.add_bar(y=d["aspect_group"], x=d["pct_of_reviews"], orientation="h",
                    marker_color=[TIER_COLOURS[tier(p)] for p in d["pct_of_reviews"]],
                    text=[f"{p:.1f}%" for p in d["pct_of_reviews"]], textposition="outside",
                    cliponaxis=False, showlegend=False,
                    hovertemplate="%{y}: %{x:.1f}% of reviews<extra></extra>")
        fig.update_layout(bargap=0.35)
    fig.update_layout(xaxis=dict(ticksuffix="%", gridcolor="#E6F0F3", zeroline=False,
                                 range=[0, max(df["pct_of_reviews"].max() * 1.18, 10)]),
                      yaxis=dict(automargin=True))
    return plotly_base(fig, 380)


def ranking_table(df, key):
    st.dataframe(
        df.rename(columns={"aspect_group": "Area", "reviews_with_negative_mentions": "Reviews with a complaint",
                           "pct_of_reviews": "% of reviews", "priority_rank": "Rank"}),
        hide_index=True, width="stretch", key=key,
        column_order=["Rank", "Area", "Reviews with a complaint", "% of reviews"],
        column_config={"% of reviews": st.column_config.ProgressColumn(
            "% of reviews", format="%.1f%%", min_value=0, max_value=100, color=TEAL)},
    )


def top_issues(df, n=3):
    items = []
    for _, r in df.head(n).iterrows():
        items.append(f"""<div class="rr-issue"><div class="rr-num">{int(r['priority_rank'])}</div>
            <div><b>{esc(r['aspect_group'])}</b><span class="pct">{r['pct_of_reviews']:.1f}%</span>
            <p>{esc(ASPECT_ACTIONS.get(r['aspect_group'], ''))}</p></div></div>""")
    st.html("".join(items))


# =====================================================================
# PAGE: DASHBOARD
# =====================================================================
def page_dashboard():
    df, source = load_ranking()
    top = df.iloc[0]
    best = MODEL_RESULTS.iloc[0]

    st.html(f"""<div class="rr-hero">
        <div><h1>Hotel Review Dashboard</h1>
        <p>See what guests complain about most, based on {BENCHMARK_REVIEWS:,} hotel reviews,
        and check any review yourself.</p></div>
        <a href="check" target="_self">Check a review</a>
      </div>
      <div class="rr-cards">
        <div class="rr-card"><div class="v">{BENCHMARK_REVIEWS:,}</div><div class="l">Reviews analysed</div></div>
        <div class="rr-card accent"><div class="v">{esc(top['aspect_group'])}</div><div class="l">Most complained about</div></div>
        <div class="rr-card"><div class="v">{top['pct_of_reviews']:.1f}%</div><div class="l">of reviews complain about it</div></div>
        <div class="rr-card"><div class="v">{best['Accuracy'] * 100:.1f}%</div><div class="l">Model accuracy (RoBERTa)</div></div>
      </div>""")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.html('<div class="rr-h">Complaints by area</div>')
        show_chart(area_bar_chart(df))
        st.caption("Percentage of reviews with at least one negative comment about each area.")
    with right:
        st.html('<div class="rr-h">Top 3 to fix</div>')
        top_issues(df)

    with st.expander("See the full table"):
        ranking_table(df, key="bench_table")
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="roomread_ranking.csv", mime="text/csv", icon=":material/download:")
        if source == "built-in":
            st.caption("Showing built-in figures because aspect_priority_ranking.csv was not found.")


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


def ensure_sentiment_model():
    with st.spinner("Loading the model. This can take a minute the first time."):
        bundle = load_sentiment_model(MODEL_ID)
    if bundle.get("error"):
        st.error(f"The model could not be loaded from `{MODEL_ID}`. If you moved the model, set MODEL_ID in "
                 f"the app's Secrets, then press Retry.\n\nDetails: {bundle['error']}", icon=":material/error:")
        if st.button("Retry", icon=":material/refresh:"):
            load_sentiment_model.clear()
            st.rerun()
        return False
    return True


def render_single(res):
    if res["status"] == "rejected":
        if res["method"] == "length":
            st.warning("That's too short. Please paste at least one full sentence about the stay.")
        else:
            st.warning("This doesn't look like a hotel review, so it wasn't analysed. "
                       "Try pasting a guest's comments about their stay.")
        return
    if res["status"] == "no_aspect":
        st.info("We couldn't find any comments about rooms, staff, food, location, price, facilities "
                "or the stay overall.")
        return

    df = res["df"]
    counts = df["sentiment"].value_counts()
    parts = [f"{counts[s]} {s}" for s in ("positive", "negative", "neutral") if s in counts]
    n_flag = int((df["check"] != "").sum())
    extra = f" {n_flag} of them may need a second look." if n_flag else ""
    st.html(f'<div class="rr-summary">We found <b>{len(df)} comment{"s" if len(df) != 1 else ""}</b>: '
            f'{", ".join(parts)}.{extra}</div>')

    tiles = []
    for asp in ASPECTS:
        sub = df[df["aspect"] == asp]
        if sub.empty:
            continue
        kinds = set(sub["sentiment"])
        label = kinds.pop() if len(kinds) == 1 else "mixed"
        c = SENT_COLOURS[label]
        tiles.append(f'<div class="rr-tile" style="border-top-color:{c}"><div class="a">{esc(asp)}</div>'
                     f'<div class="s" style="color:{c}">{label}</div></div>')
    st.html('<div class="rr-tiles">' + "".join(tiles) + "</div>")

    st.html('<div class="rr-h">Your review</div>')
    spans = []
    for i, sent in enumerate(res["sentences"], start=1):
        sub = df[df["sentence_no"] == i]
        if sub.empty:
            spans.append(f'<span class="rr-s-none">{esc(sent)}</span> ')
            continue
        kinds = set(sub["sentiment"])
        cls = f"rr-s-{next(iter(kinds))}" if len(kinds) == 1 else "rr-s-mixed"
        tags = "".join(f'<span class="rr-tag" style="background:{SENT_COLOURS[r.sentiment]}">{esc(r.aspect)}</span>'
                       for r in sub.itertuples())
        spans.append(f'<span class="{cls}">{esc(sent)}</span>{tags} ')
    legend = "".join(f'<span><i style="background:{SENT_COLOURS[k]}"></i>{k.capitalize()}</span>'
                     for k in ("positive", "negative", "neutral", "mixed"))
    st.html(f'<div class="rr-read">{"".join(spans)}</div><div class="rr-legend">{legend}</div>')

    with st.expander("Show details"):
        table = df.rename(columns={"sentence_no": "#", "sentence": "Sentence", "aspect": "Area",
                                   "sentiment": "Sentiment", "confidence": "Confidence",
                                   "check": "Note", "matched_words": "Matched words"})
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_order=["#", "Sentence", "Area", "Sentiment", "Confidence", "Note", "Matched words"],
            column_config={"Sentence": st.column_config.TextColumn(width="large"),
                           "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0,
                                                                         max_value=1, color=TEAL)},
        )
        st.caption(f"Neutral results and results below {LOW_CONFIDENCE:.0%} confidence are marked, "
                   "because the model is less reliable on them.")
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="roomread_review.csv", mime="text/csv", icon=":material/download:")


def page_check():
    page_title("Check a review", "Paste a guest review to see what they liked and what they didn't.")
    st.session_state.setdefault("review_text", "")
    st.pills("Or try an example", list(SAMPLES), key="sample_choice", on_change=_load_sample)
    st.text_area("Review", key="review_text", height=170, label_visibility="collapsed",
                 placeholder="Example: The room was clean but breakfast was cold.")
    c1, c2, _ = st.columns([1.1, 0.7, 4])
    run = c1.button("Analyse", type="primary", icon=":material/search:", width="stretch")
    c2.button("Clear", on_click=_clear_review, width="stretch")

    if not ensure_sentiment_model():
        return

    text = st.session_state["review_text"].strip()
    if run:
        if not text:
            st.info("Please paste a review first, or pick an example.")
            return
        with st.spinner("Analysing…"):
            try:
                st.session_state["single_result"] = analyse_review(text, MODEL_ID)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
                return

    if "single_result" in st.session_state:
        render_single(st.session_state["single_result"])


# =====================================================================
# PAGE: UPLOAD REVIEWS
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
    raise ValueError("Please save the file as a UTF-8 CSV and try again.")


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
    status = st.progress(0.0, text="Starting…")
    kept, rejected = [], 0
    if use_gate:
        for i, rv in enumerate(reviews):
            ok, _, _ = check_relevance(rv)
            if ok:
                kept.append(rv)
            else:
                rejected += 1
            status.progress(0.3 * (i + 1) / len(reviews), text=f"Checking reviews: {i + 1} of {len(reviews)}")
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
        status.progress(base + (1 - base) * done / total, text=f"Analysing comments: {done:,} of {total:,}")

    preds = score_pairs(pairs, batch_size=32, on_progress=progress)
    status.empty()
    pred = pd.DataFrame(meta)
    pred["sentiment"] = [p[0] for p in preds]
    pred["confidence"] = [round(p[1], 3) for p in preds]
    pred["check"] = [flag_for(s, c) for s, c in preds]
    return {"n_reviews": len(kept), "rejected": rejected, "pred": pred,
            "ranking": build_ranking(pred, len(kept))}


def page_file():
    page_title("Upload reviews", "Upload your hotel's reviews and see which areas guests complain about most. "
                                 "CSV, Excel or a text file with one review per line.")
    up = st.file_uploader("Reviews file", type=["csv", "xlsx", "txt"], label_visibility="collapsed")
    if up is None:
        st.caption("Tip: start with around 100 reviews. Larger files take a few minutes.")
        st.session_state.pop("file_result", None)
        return

    try:
        data = read_upload(up)
    except Exception as exc:  # noqa: BLE001
        st.error(f"We couldn't read this file. {exc}")
        return
    if data.empty:
        st.warning("The file is empty.")
        return
    guess = guess_text_column(data)
    if guess is None:
        st.error("We couldn't find a column with review text.")
        return

    c1, c2, c3 = st.columns([1.3, 1, 1.3], gap="medium")
    col = c1.selectbox("Review column", list(data.columns), index=list(data.columns).index(guess))
    available = int(data[col].dropna().astype(str).str.strip().ne("").sum())
    limit = c2.number_input("How many reviews", min_value=1, max_value=min(available, MAX_FILE_REVIEWS),
                            value=min(available, 100), step=10)
    c3.write("")
    use_gate = c3.toggle("Skip rows that aren't reviews", value=False,
                         help="Slower. Use it if the file has other text mixed in.")

    run_key = (up.name, up.size, col, int(limit), use_gate)
    if st.button("Analyse reviews", type="primary", icon=":material/play_arrow:"):
        if not ensure_sentiment_model():
            return
        reviews = data[col].dropna().astype(str).str.strip()
        reviews = reviews[reviews.ne("")].head(int(limit)).tolist()
        try:
            st.session_state["file_result"] = {"key": run_key, **run_file_analysis(reviews, use_gate)}
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
            return

    res = st.session_state.get("file_result")
    if not res or res["key"] != run_key:
        return
    if res["ranking"] is None:
        st.info("None of these reviews mention the areas we track.")
        return

    rk, pred, n = res["ranking"], res["pred"], res["n_reviews"]
    bench, _ = load_ranking()
    bench_map = dict(zip(bench["aspect_group"], bench["pct_of_reviews"]))
    top = rk.iloc[0]
    skipped = f"{res['rejected']} rows skipped" if res["rejected"] else "No rows skipped"

    st.html(f"""<div class="rr-cards">
        <div class="rr-card"><div class="v">{n:,}</div><div class="l">Reviews analysed</div></div>
        <div class="rr-card accent"><div class="v">{esc(top['aspect_group'])}</div><div class="l">Most complained about</div></div>
        <div class="rr-card"><div class="v">{top['pct_of_reviews']:.1f}%</div><div class="l">of your reviews complain about it</div></div>
        <div class="rr-card"><div class="v">{len(pred):,}</div><div class="l">Comments found ({esc(skipped.lower())})</div></div>
      </div>""")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.html('<div class="rr-h">Your reviews compared with all reviews</div>')
        show_chart(area_bar_chart(rk, benchmark=bench_map))
    with right:
        st.html('<div class="rr-h">Top 3 to fix</div>')
        top_issues(rk)

    with st.expander(f"See all {len(pred):,} comments"):
        st.dataframe(pred, hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", format="%.2f", min_value=0, max_value=1, color=TEAL)})
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download ranking", rk.to_csv(index=False).encode("utf-8"),
                       file_name="my_ranking.csv", mime="text/csv", icon=":material/download:", width="stretch")
    d2.download_button("Download comments", pred.to_csv(index=False).encode("utf-8"),
                       file_name="my_comments.csv", mime="text/csv", icon=":material/download:", width="stretch")


# =====================================================================
# PAGE: MODEL RESULTS
# =====================================================================
def page_model():
    page_title("Model results", "Three models were tested on 1,692 hand-labelled examples from the OATS-Hotels "
                                "dataset. Fine-tuned RoBERTa did best and is the one used in this app.")
    fig = go.Figure()
    fig.add_bar(x=MODEL_RESULTS["Model"], y=MODEL_RESULTS["Accuracy"], name="Accuracy", marker_color=MIDNIGHT,
                text=[f"{v:.3f}" for v in MODEL_RESULTS["Accuracy"]], textposition="outside")
    fig.add_bar(x=MODEL_RESULTS["Model"], y=MODEL_RESULTS["Macro-F1"], name="Macro-F1", marker_color=TEAL,
                text=[f"{v:.3f}" for v in MODEL_RESULTS["Macro-F1"]], textposition="outside")
    fig.update_layout(barmode="group", bargap=0.3, yaxis=dict(range=[0, 1.08], gridcolor="#E6F0F3"))

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        show_chart(plotly_base(fig, 360))
    with right:
        st.html(f"""<div class="rr-h">Good to know</div>
          <div class="rr-issue"><div><b>Accuracy</b><p>How many predictions were right overall. It looks high
            for every model because most examples are positive.</p></div></div>
          <div class="rr-issue"><div><b>Macro-F1</b><p>Treats positive, negative and neutral equally, so it is
            the fairer score here.</p></div></div>
          <div class="rr-issue"><div><b>Neutral is the hardest</b><p>There were only 38 neutral test examples,
            so the app marks neutral results for a second look.</p></div></div>""")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.html('<div class="rr-h">Scores</div>')
        st.dataframe(MODEL_RESULTS, hide_index=True, width="stretch",
                     column_config={"Accuracy": st.column_config.NumberColumn(format="%.3f"),
                                    "Macro-F1": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        st.html('<div class="rr-h">Is the difference real? (McNemar test)</div>')
        st.dataframe(MCNEMAR, hide_index=True, width="stretch",
                     column_config={"p-value": st.column_config.NumberColumn(format="%.3f")})
    st.caption("RoBERTa is significantly better than BERT. Its lead over zero-shot BART is not significant.")


# =====================================================================
# PAGE: ABOUT
# =====================================================================
def page_about():
    page_title("About this project", 
                                     "RoomRead looks at each sentence of a review and works out what the guest is "
                                     "talking about and how they feel about it.")
    steps = [
        ("Split the review into sentences", "So one review can praise the room and criticise the breakfast."),
        ("Check that it's a hotel review", f"A zero-shot model filters out unrelated text "
                                           f"(it needs a score of {RELEVANCE_THRESHOLD:.2f} or more)."),
        ("Find what each sentence is about", "Keywords match each sentence to one or more of seven areas: "
                                             + ", ".join(ASPECTS) + "."),
        ("Predict the sentiment", "A fine-tuned RoBERTa model labels each area as positive, negative or neutral."),
        ("Rank the areas", "For many reviews, each area is scored by the share of reviews that complain about it."),
    ]
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.html('<div class="rr-h">How it works</div>' + "".join(
            f'<div class="rr-step"><div class="rr-num">{i}</div><div><b>{esc(t)}</b><p>{esc(d)}</p></div></div>'
            for i, (t, d) in enumerate(steps, start=1)))
    with right:
        st.html(f"""<div class="rr-h">Data</div>
          <div class="rr-issue"><div><p style="margin:0">The model was trained and tested on the OATS-Hotels dataset
            (Chebolu et al., 2024). The dashboard comes from {BENCHMARK_REVIEWS:,} other hotel reviews.</p></div></div>
          <div class="rr-h" style="margin-top:1rem">Limitations</div>
          <div class="rr-issue"><div><p style="margin:0">Keyword matching can miss or mislabel areas. Neutral results
            are the least reliable. It works in English only.</p></div></div>
          <div class="rr-h" style="margin-top:1rem">Author</div>
          <div class="rr-issue"><div><b>Nethika Alagarathnam</b></div>""")


# =====================================================================
# NAVIGATION
# =====================================================================
setup_page()
pages = [
    st.Page(page_dashboard, title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page(page_check, title="Check a review", icon=":material/rate_review:", url_path="check"),
    st.Page(page_file, title="Upload reviews", icon=":material/upload_file:", url_path="upload"),
    st.Page(page_model, title="Model results", icon=":material/bar_chart:", url_path="model"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages, position="top")
nav.run()
footer()
