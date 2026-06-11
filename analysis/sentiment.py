import streamlit as st
from transformers import pipeline
import pandas as pd

st.header('Sentiment analysis')
df = pd.read_csv('../data/activity_today_dist2.csv')

classifier = pipeline('text-classification')
total_rows = len(df)
# progress_bar = st.progress(0)

def calculate_sentiment(text):
    return classifier(str(text)[:500])[0]['label']

# Obliczanie etykiet sentymentu i wyświetlanie postępu
sentiments = []
for i, text in enumerate(df['content']):
    sentiment = calculate_sentiment(text)
    sentiments.append(sentiment)
    # progress_bar.progress((i + 1) / total_rows)

df['sentiment'] = sentiments
# progress_bar.empty()

# save = st.checkbox('Zapisać wyniki?')

st.write('Zapisywanie wyników...')
df.to_csv('../data/sentiment_analysis_big.csv', index=False)
st.write('Wyniki zapisane jako sentiment_analysis.csv')
st.write(df)
