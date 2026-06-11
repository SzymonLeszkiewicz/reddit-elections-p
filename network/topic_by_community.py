# import gensim
import pandas as pd
import spacy
from gensim import corpora
from pprint import pprint
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os

import streamlit as st
from streamlit import components

st.set_page_config(layout="wide")

'''
scripts for topic modeling

input: data/network_activity.csv (with content column)

output: data/network_topic3.csv (with topic column)

output: data/network_topic7.csv (with topic column)

output: output/lda3.html (viz)

output: output/lda7.html (viz)

'''

N_ROWS = 30
SAMPLING = 0

def classify_topic_to_df(text):

    new_text_words = gen_words(lemmatization([text]))
    new_text_corpus = [id2word.doc2bow(text) for text in new_text_words]
    new_text_topic = lda_model[new_text_corpus]
    if isinstance(new_text_topic, gensim.interfaces.TransformedCorpus):
        new_text_topic = new_text_topic[0]
        return max(new_text_topic, key=lambda x: x[1])[0]
    else:
        return -1


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("pl_core_news_md", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final


# read stop words
stopwords = []
with open('../data/stop_words.txt') as f:
    for line in f:
        stopwords.append(line.strip())

# read reddit data
df = pd.read_csv('../output/network_author_uniq.csv', sep=',')

st.write(df.shape)
if SAMPLING:
    df = df.head(N_ROWS)
    st.write(df.shape)


# remove stop words
df['content'] = df['content'].astype(str)
df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

data = df['content'].values.tolist()
lemmatized_texts = lemmatization(data)

df['lemmatized'] = lemmatized_texts

# create dictionary
data_words = gen_words(lemmatized_texts)
id2word = corpora.Dictionary(data_words)

# create corpus
corpus = [id2word.doc2bow(text) for text in data_words]

# # create lda model with 3 topics
# num_topics = 3
# lda_model = gensim.models.ldamodel.LdaModel(
#     corpus=corpus,
#     id2word=id2word,
#     num_topics=num_topics,
#     random_state=100,
#     update_every=1,
#     chunksize=200,
#     passes=15,
#     alpha="auto",
# )
# doc_lda = lda_model[corpus]
#
#
# path = '/Users/szymonleszkiewicz/Desktop/AI2024/AMC/2023-zadanie-3-SzymonLeszkiewicz/output/lda3.html'
# vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
# vis = pyLDAvis.prepared_data_to_html(vis)
# with open(path, 'w') as f:
#     f.write(vis)
#
# components.v1.html(vis, height=800)
#
# st.write(classify_topic_to_df('sekss asasf samochod polska wybory'))
#
#
# # classify 'content' column with classify_topic_to_df function
# # drop null on lemmatized column
# df = df.dropna(subset=['lemmatized'])
#
# df['topic'] = df['content'].apply(lambda x: classify_topic_to_df(x))
# st.write(df.head(10))
# st.write(df['topic'].value_counts())
#
# # save df
# df.to_csv('/Users/szymonleszkiewicz/Desktop/AI2024/AMC/2023-zadanie-3-SzymonLeszkiewicz/data/network_topic3.csv',
#           index=False)

# create lda model with 7 topics
num_topics = 6
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=num_topics,
    random_state=100,
    update_every=1,
    chunksize=200,
    passes=15,
    alpha="auto",
)

doc_lda = lda_model[corpus]

path = f'../output/lda_topics_{num_topics}.html'
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
vis = pyLDAvis.prepared_data_to_html(vis)
with open(path, 'w') as f:
    f.write(vis)

components.v1.html(vis, height=800)

df['topic'] = df['content'].apply(lambda x: classify_topic_to_df(x))
st.write(df)
st.write(df['topic'].value_counts())

# save df
df.to_csv(f'../data/network_topic{num_topics}.csv', index=False)
