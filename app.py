import re
import traceback
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# =========================================================
# CONFIG
# =========================================================
MODEL_ID = "nethika/Aspecta"
ID_TO_SENTIMENT = {
    0: "positive",
    1: "negative",
    2: "neutral"
}
RANKING_CSV = "aspect_priority_ranking.csv"

DEVICE = "cpu"

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_models():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    zs = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )

    return model, tokenizer, zs

best_model, best_tok, zs_pipeline = load_models()

# =========================================================
# LOAD DASHBOARD DATA
# =========================================================
try:
    ranking_df = pd.read_csv(RANKING_CSV)
except:
    ranking_df = None

# =========================================================
# ASPECT KEYWORDS
# =========================================================
ASPECT_KEYWORDS = {
    'Room quality': ['room','bed','pillow','bathroom','shower','clean','dirty','smell','noisy','air condition','ac ','view','dust'],
    'Staff & Service': ['staff','service','receptionist','front desk','concierge','employee','manager','helpful','rude','friendly'],
    'Food & Beverage': ['breakfast','food','restaurant','buffet','dinner','lunch','meal','coffee','bar ','drink'],
    'Location': ['location','located','walk','distance','nearby','downtown','beach','airport','transport'],
    'Value for money': ['price','value','expensive','cheap','worth','cost','money','overpriced','affordable'],
    'Facilities': ['pool','gym','wifi','parking','spa','elevator','facility','facilities','amenities'],
    'Overall': ['overall','stay','hotel','experience','recommend','trip','manage','manag'],
}

def split_sentences(text):
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def detect_aspects(sentence):
    s = sentence.lower()
    return [
        asp for asp, kws in ASPECT_KEYWORDS.items()
        if any(kw in s for kw in kws)
    ]

def build_input_text(sentence, aspect):
    return f"{sentence} [SEP] {aspect}"

def predict_sentiment(sentence, aspect):

    text = build_input_text(sentence, aspect)

    enc = best_tok(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    best_model.eval()

    with torch.no_grad():
        logits = best_model(**enc).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred = probs.argmax()

    return ID_TO_SENTIMENT[int(pred)], float(probs[pred])

RELEVANCE_LABELS = [
    "a hotel guest review describing a stay",
    "unrelated text with no connection to a hotel stay"
]

def is_relevant_review(text):

    if len(text.split()) < 4:
        return False, 0

    result = zs_pipeline(
        text[:600],
        RELEVANCE_LABELS,
        truncation=True
    )

    return (
        result["labels"][0] == RELEVANCE_LABELS[0]
        and result["scores"][0] >= 0.55
    ), float(result["scores"][0])

def analyze_review(review_text):

    try:

        if review_text.strip() == "":
            return [["-", "-", "Paste a review", "-"]]

        ok, score = is_relevant_review(review_text)

        if not ok:
            return [[
                "-",
                "-",
                f"Not recognised as hotel review ({score:.2f})",
                "-"
            ]]

        rows = []

        for sent in split_sentences(review_text):

            aspects = detect_aspects(sent)

            for aspect in aspects:

                sentiment, conf = predict_sentiment(sent, aspect)

                rows.append([
                    sent,
                    aspect,
                    sentiment,
                    round(conf,3)
                ])

        if len(rows)==0:
            rows=[["-","-","No tracked aspect found.","-"]]

        return rows

    except Exception as e:

        print(traceback.format_exc())

        return [["ERROR",type(e).__name__,str(e),"-"]]

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="RoomRead",
    layout="wide"
)

st.title("🏨 RoomRead")
st.subheader("Aspect-Based Sentiment Analysis for Hotel Reviews")

tab1, tab2 = st.tabs([
    "Priority Dashboard",
    "Check a Review"
])

with tab1:

    st.write("Negative mentions ranked by aspect.")

    if ranking_df is not None:

        st.dataframe(
            ranking_df,
            use_container_width=True
        )

        st.bar_chart(
            ranking_df.set_index(
                ranking_df.columns[0]
            )["pct_of_reviews"]
        )

    else:

        st.warning("Ranking CSV not found.")

with tab2:

    review = st.text_area(
        "Hotel Review",
        height=180,
        placeholder="The room was spotless but breakfast was cold..."
    )

    if st.button("Analyze"):

        rows = analyze_review(review)

        df = pd.DataFrame(
            rows,
            columns=[
                "Sentence",
                "Aspect",
                "Predicted Sentiment",
                "Confidence"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

st.markdown("---")
st.caption("RoomRead © 2026")