import streamlit as st
from transformers import pipeline
import pandas as pd

# Normalize model labels (Positive/Negative/LABEL_0 etc.) to POSITIVE/NEGATIVE/NEUTRAL
_LABEL_MAP = {
    "label_0":    "NEGATIVE",
    "label_1":    "NEUTRAL",
    "label_2":    "POSITIVE",
    "negative":   "NEGATIVE",
    "neutral":    "NEUTRAL",
    "positive":   "POSITIVE",
    "negatywny":  "NEGATIVE",
    "neutralny":  "NEUTRAL",
    "pozytywny":  "POSITIVE",
}

MODEL_ID = "eevvgg/bert-polish-sentiment-politics"

@st.cache_resource(show_spinner="Loading sentiment model...")
def get_classifier():
    return pipeline("text-classification", model=MODEL_ID)

def normalize_label(raw: str) -> str:
    return _LABEL_MAP.get(raw.lower(), raw.upper())

def calculate_sentiment(clf, raw_text: str) -> str:
    raw = clf(str(raw_text)[:500])[0]["label"]
    return normalize_label(raw)


st.set_page_config(page_title="Sentiment Analysis PL")
st.header("Sentiment analysis (Polish model)")

df = pd.read_csv("../data/activity_today_dist2.csv")
st.caption(f"Loaded {len(df)} records from activity_today_dist2.csv")

classifier = get_classifier()

progress_bar = st.progress(0, text="Classifying sentiment...")
total = len(df)
sentiments = []
for i, text in enumerate(df["content"]):
    sentiments.append(calculate_sentiment(classifier, text))
    progress_bar.progress((i + 1) / total, text=f"{i + 1}/{total}")

progress_bar.empty()
df["sentiment"] = sentiments

st.subheader("Sentiment distribution")
dist = df["sentiment"].value_counts().reset_index()
dist.columns = ["sentiment", "count"]
st.bar_chart(dist.set_index("sentiment"))

output_path = "../data/sentiment_analysis_big.csv"
df.to_csv(output_path, index=False)
st.success(f"Results saved to {output_path} - all dashboards will use the updated data.")
st.dataframe(df)