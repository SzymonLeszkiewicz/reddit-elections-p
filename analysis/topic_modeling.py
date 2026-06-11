# Gensim
import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.models import CoherenceModel, Phrases, phrases
from streamlit import components
import spacy
import pyLDAvis
import pyLDAvis.gensim_models
import warnings
import numpy as np
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

stop_words = set(open('../data/stop_words.txt', 'r').read().split('\n'))

st.set_page_config(layout="wide")
st.header('Topic modeling')

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header('Parameters')
sample_size = st.sidebar.slider('Documents', 1000, 20000, 5000, step=1000)
topic_number = st.sidebar.slider('Number of topics', 2, 15, 3)
show_coherence_sweep = st.sidebar.checkbox('Coherence vs k plot (slow)')


# ── PREPROCESSING (cache - liczy się raz dla danego sample_size) ───────────────
@st.cache_data(show_spinner='Lemmatising (spaCy)...')
def prepare_data(sample_size):
    df = pd.read_csv('../data/sentiment_analysis.csv').dropna().head(sample_size)
    data = df['content'].values.tolist()

    nlp = spacy.load("pl_core_news_sm", disable=["parser", "ner"])
    allowed = {"NOUN", "ADJ"}

    # nlp.pipe przetwarza wsad (batch) zamiast tekst po tekście - ~5x szybciej
    data_words = []
    for doc in nlp.pipe(data, batch_size=256):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in allowed
            and token.lemma_.lower() not in stop_words
            and len(token.lemma_) >= 3
        ]
        data_words.append(tokens)

    # bigramy
    bigram = Phrases(data_words, min_count=10, threshold=100)
    bigram_mod = phrases.Phraser(bigram)
    data_words = [bigram_mod[doc] for doc in data_words]

    dictionary = corpora.Dictionary(data_words)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
    corpus = [dictionary.doc2bow(text) for text in data_words]

    return data, data_words, dictionary, corpus, df


data, data_words, id2word, corpus, df = prepare_data(sample_size)
st.caption(f'Loaded {len(data)} documents · vocabulary: {len(id2word)} unique tokens')


# ── MODEL (cache osobno - przelicza się tylko gdy zmieni się topic_number lub sample_size) ──
@st.cache_resource(show_spinner='Training LDA...')
def train_lda(sample_size, topic_number):
    _, data_words, id2word, corpus, _ = prepare_data(sample_size)
    model = gensim.models.LdaMulticore(   # wielowątkowy zamiast LdaModel
        corpus=corpus,
        id2word=id2word,
        num_topics=topic_number,
        random_state=100,
        chunksize=200,
        passes=3,
        workers=3,
    )
    return model


lda_model = train_lda(sample_size, topic_number)

# ── METRYKI ────────────────────────────────────────────────────────────────────
st.subheader('Model quality')

@st.cache_data(show_spinner='Computing metrics...')
def compute_metrics(sample_size, topic_number):
    model = train_lda(sample_size, topic_number)
    _, data_words, id2word, corpus, _ = prepare_data(sample_size)
    perplexity = model.log_perplexity(corpus)
    # u_mass: używa gotowego corpus → instant; c_v: liczy PMI po tekstach → wolne
    coherence = CoherenceModel(
        model=model, corpus=corpus, dictionary=id2word, coherence='u_mass'
    ).get_coherence()
    return perplexity, coherence


perplexity, coherence = compute_metrics(sample_size, topic_number)

col1, col2 = st.columns(2)
col1.metric('Coherence (u_mass)', f'{coherence:.4f}',
            help='Closer to 0 is better (negative values). Fast corpus-based metric.')
col2.metric('Perplexity (log)', f'{perplexity:.2f}',
            help='Lower is better.')

# wykres coherence vs k - schowany za checkboxem, bo jest wolny
if show_coherence_sweep:
    @st.cache_data(show_spinner='Searching for optimal k...')
    def coherence_sweep(sample_size):
        _, data_words, id2word, corpus, _ = prepare_data(sample_size)
        scores = []
        for k in range(2, 11):
            m = gensim.models.LdaMulticore(
                corpus=corpus, id2word=id2word, num_topics=k,
                random_state=100, passes=3, workers=3
            )
            scores.append(CoherenceModel(
                model=m, corpus=corpus, dictionary=id2word, coherence='u_mass'
            ).get_coherence())
        return scores

    scores = coherence_sweep(sample_size)
    fig = px.line(x=list(range(2, 11)), y=scores, markers=True,
                  labels={'x': 'Number of topics (k)', 'y': 'Coherence u_mass'},
                  title='Coherence vs k - look for the elbow')
    st.plotly_chart(fig, use_container_width=True)

# ── WIZUALIZACJA ───────────────────────────────────────────────────────────────
st.subheader('Topic visualisation (pyLDAvis)')

@st.cache_data(show_spinner='Generating visualisation...')
def get_vis_html(sample_size, topic_number):
    model = train_lda(sample_size, topic_number)
    _, _, id2word, corpus, _ = prepare_data(sample_size)
    vis = pyLDAvis.gensim_models.prepare(model, corpus, id2word)
    return pyLDAvis.prepared_data_to_html(vis)

components.v1.html(get_vis_html(sample_size, topic_number), height=800)

# ── ROZKŁAD TEMATÓW ────────────────────────────────────────────────────────────
st.subheader('Dominant topic distribution')


@st.cache_data(show_spinner='Assigning topics to documents...')
def format_topics_sentences(sample_size, topic_number):
    model = train_lda(sample_size, topic_number)
    _, _, id2word, corpus, _ = prepare_data(sample_size)
    rows = []
    for row in model[corpus]:
        row = sorted(row, key=lambda x: x[1], reverse=True)
        topic_num, prop_topic = row[0]
        wp = model.show_topic(topic_num)
        keywords = ", ".join([word for word, _ in wp])
        rows.append([int(topic_num), round(prop_topic, 4), keywords])
    return pd.DataFrame(rows, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])


df_topics = format_topics_sentences(sample_size, topic_number)
st.write(df_topics['Dominant_Topic'].value_counts())

# ── TOP vs BOTTOM postów ───────────────────────────────────────────────────────
st.subheader('Topics in popular vs unpopular posts')
df_submissions = df[df['type'] == 'submission'].reset_index(drop=True)
n = min(len(df_submissions), len(df_topics))
df_with_topics = df_submissions.iloc[:n].copy()
df_with_topics['Dominant_Topic'] = df_topics['Dominant_Topic'].iloc[:n].values
df_with_topics = df_with_topics.sort_values(by='replies', ascending=False)

top_topics = df_with_topics.head(100)['Dominant_Topic'].value_counts(normalize=True).rename('Top 100')
bot_topics = df_with_topics.tail(100)['Dominant_Topic'].value_counts(normalize=True).rename('Bottom 100')
st.dataframe(pd.concat([top_topics, bot_topics], axis=1).fillna(0))
