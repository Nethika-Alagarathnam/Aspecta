import re
import traceback
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# =========================================================
# CONFIG — edit these three lines
# =========================================================
MODEL_ID = "nethika/Aspecta"          # from Step 1
ID_TO_SENTIMENT = {0: "positive", 1: "negative", 2: "neutral"}  # confirm this matches your actual SENTIMENT_MAP in the notebook
RANKING_CSV = "aspect_priority_ranking.csv"        # uploaded alongside this file, from Step 2

DEVICE = "cpu"

# =========================================================
# LOAD MODEL + RANKING DATA (runs once, at Space startup)
# =========================================================
print("Loading fine-tuned model from the Hub...")
best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
best_tok = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading zero-shot relevance classifier...")
zs_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

print("Loading priority ranking...")
try:
    ranking_df = pd.read_csv(RANKING_CSV)
except FileNotFoundError:
    ranking_df = None
    print(f"WARNING: {RANKING_CSV} not found — upload it alongside app.py. Dashboard tab will show a placeholder.")

# =========================================================
# ASPECT + SENTIMENT LOGIC — same as your notebook's demo cell
# =========================================================
ASPECT_KEYWORDS = {
    'Room quality':     ['room', 'bed', 'pillow', 'bathroom', 'shower', 'clean', 'dirty', 'smell', 'noisy', 'air condition', 'ac ', 'view', 'dust'],
    'Staff & Service':  ['staff', 'service', 'receptionist', 'front desk', 'concierge', 'employee', 'manager', 'helpful', 'rude', 'friendly'],
    'Food & Beverage':  ['breakfast', 'food', 'restaurant', 'buffet', 'dinner', 'lunch', 'meal', 'coffee', 'bar ', 'drink'],
    'Location':         ['location', 'located', 'walk', 'distance', 'nearby', 'downtown', 'beach', 'airport', 'transport'],
    'Value for money':  ['price', 'value', 'expensive', 'cheap', 'worth', 'cost', 'money', 'overpriced', 'affordable'],
    'Facilities':       ['pool', 'gym', 'wifi', 'parking', 'spa', 'elevator', 'facility', 'facilities', 'amenities'],
    'Overall':          ['overall', 'stay', 'hotel', 'experience', 'recommend', 'trip', 'manage', 'manag'],
}

def split_sentences(text):
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def detect_aspects(sentence):
    s = sentence.lower()
    return [asp for asp, kws in ASPECT_KEYWORDS.items() if any(kw in s for kw in kws)]

def build_input_text(sentence, aspect):
    return f"{sentence} [SEP] {aspect}"   # <-- must match your training format exactly

def predict_sentiment(sentence, aspect):
    text = build_input_text(sentence, aspect)
    enc = best_tok(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(DEVICE)
    best_model.eval()
    with torch.no_grad():
        logits = best_model(**enc).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return ID_TO_SENTIMENT[pred_idx], float(probs[pred_idx])

RELEVANCE_LABELS = ['a hotel guest review describing a stay', 'unrelated text with no connection to a hotel stay']

def is_relevant_review(text):
    words = text.strip().split()
    if len(words) < 4:
        return False, 0.0
    sample = text if len(text) <= 600 else text[:600]
    result = zs_pipeline(sample, RELEVANCE_LABELS, truncation=True)
    return (result['labels'][0] == RELEVANCE_LABELS[0] and result['scores'][0] >= 0.55), float(result['scores'][0])

def analyze_review(review_text):
    try:
        if not review_text or not review_text.strip():
            return [["—", "—", "Paste a review above", "—"]]
        is_review, score = is_relevant_review(review_text)
        if not is_review:
            return [["—", "—", f"Not recognised as a hotel review (confidence {score:.2f})", "—"]]
        rows = []
        for sent in split_sentences(review_text):
            for aspect in detect_aspects(sent):
                sentiment, conf = predict_sentiment(sent, aspect)
                label = sentiment + ("  (low-confidence class)" if sentiment == "neutral" else "")
                short_sent = sent if len(sent) <= 120 else sent[:117] + "..."
                rows.append([short_sent, aspect, label, f"{conf:.2f}"])
        if not rows:
            return [["—", "—", "Review recognised, but no tracked aspect was mentioned.", "—"]]
        return rows
    except Exception as e:
        print(traceback.format_exc())
        return [["ERROR", type(e).__name__, str(e)[:150], "check Space logs"]]

# =========================================================
# UI — two tabs: dashboard + live checker
# =========================================================
with gr.Blocks(title="RoomRead") as demo:
    gr.Markdown("# RoomRead — Aspect-Based Sentiment for Hotel Reviews")

    with gr.Tab("Priority Dashboard"):
        gr.Markdown("What guests are telling you to fix, ranked by share of reviews mentioning a negative experience.")
        if ranking_df is not None:
            gr.Dataframe(value=ranking_df, wrap=True)
            gr.BarPlot(
                ranking_df, x=ranking_df.columns[0], y="pct_of_reviews",
                title="Negative mentions by aspect (%)", vertical=False, height=350,
            )
        else:
            gr.Markdown("*Ranking data not found — upload `aspect_priority_ranking.csv` alongside `app.py`.*")

    with gr.Tab("Check a Review"):
        gr.Markdown(
            "Paste a hotel review. Off-topic or non-review input is rejected before scoring. "
            "**Note:** the neutral class had the weakest F1 in evaluation — treat neutral predictions with caution."
        )
        inp = gr.Textbox(label="Review text", lines=6, placeholder="e.g. The room was spotless but breakfast was cold and the staff seemed uninterested.")
        btn = gr.Button("Analyze", variant="primary")
        out = gr.Dataframe(headers=["Sentence", "Aspect", "Predicted sentiment", "Confidence"], wrap=True)
        btn.click(fn=analyze_review, inputs=inp, outputs=out)
        gr.Examples(examples=[
            ["The room was spotless but breakfast was cold and the staff seemed uninterested."],
            ["Hello I'm Nethika"],
        ], inputs=inp)

demo.launch()
