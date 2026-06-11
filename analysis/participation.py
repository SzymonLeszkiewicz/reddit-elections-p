import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

df = pd.read_csv('../data/sentiment_analysis_big.csv')

st.header('User participation across subreddits')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
min_date = df['date'].min()
max_date = df['date'].max()
date_range = pd.date_range(min_date, max_date)

full_df = pd.DataFrame({'date': date_range})
df = pd.concat([df, full_df])
df['date'] = df['date'].dt.date
df = df.sort_values(by='date')

df['subreddit_type'] = df['subreddit'].apply(lambda x: 'Polska' if x == 'Polska' else 'Other')
fig = px.pie(df, names='subreddit_type', title='Activity distribution')
st.plotly_chart(fig)

df_comments = df[df['type'] == 'comment']
df_submissions = df[df['type'] == 'submission']

fig = px.bar(df_submissions, x='date', color='subreddit', title='Submission activity over time', labels={'date': 'Date', 'subreddit': 'Subreddit'}, height=600)
st.plotly_chart(fig, use_container_width=True)

fig = px.bar(df_comments, x='date', color='subreddit', title='Comment activity over time', labels={'date': 'Date', 'subreddit': 'Subreddit'}, height=600)
st.plotly_chart(fig, use_container_width=True)


ALL_USERS = df['author'].unique()

df_comments = df[df['type'] == 'comment']
df_submissions = df[df['type'] == 'submission']

# submissions
df_author = dict.fromkeys(ALL_USERS, 0)
for user in ALL_USERS:
    df_author[user] = len(df_submissions[df_submissions['author'] == user]['subreddit'].unique())
df_author = pd.DataFrame.from_dict(df_author, columns=['count'], orient='index')
df_author = df_author.reset_index()
df_author = df_author.rename(columns={'index': 'author'})

# FANS vs REST
fans = sum(df_author[df_author['count'] > 1]['count'])
rest = sum(df_author[df_author['count'] == 1]['count'])
fig = px.pie(df_author, values=[fans, rest], names=['Fans', 'Rest'], title='How many users participate in more than one subreddit?')
st.plotly_chart(fig, use_container_width=True)


# comments
df_author = dict.fromkeys(ALL_USERS, 0)
for user in ALL_USERS:
    df_author[user] = len(df_comments[df_comments['author'] == user]['subreddit'].unique())
df_author = pd.DataFrame.from_dict(df_author, columns=['count'], orient='index')
df_author = df_author.reset_index()
df_author = df_author.rename(columns={'index': 'author'})

# FANS vs REST
fans = sum(df_author[df_author['count'] > 1]['count'])
rest = sum(df_author[df_author['count'] == 1]['count'])
fig = px.pie(df_author, values=[fans, rest], names=['Fans', 'Rest'], title='How many users comments in more than one subreddit?')
st.plotly_chart(fig, use_container_width=True)

st.subheader("Summary")


# SENTIMENT
st.header('Sentiment analysis')
df_sentiment = pd.read_csv('../data/sentiment_analysis_big.csv')

df_sentiment = df_sentiment.drop(columns=['author', 'date', 'ratio'])

# Grupowanie i agregacja
agg_functions = {
    'score': 'sum',  # Kolumna 'Value1' zostanie zsumowana
    'content': 'count',
    'replies': 'sum'
}

df_sentiment = df_sentiment.groupby(['subreddit', 'sentiment', 'type']).agg(agg_functions).reset_index()

# normalize score, replies and content on sentiment
df_sentiment['score'] = df_sentiment['score'] / df_sentiment.groupby(['subreddit', 'type'])['score'].transform('sum')
df_sentiment['replies'] = df_sentiment['replies'] / df_sentiment.groupby(['subreddit', 'type'])['replies'].transform('sum')
df_sentiment['content'] = df_sentiment['content'] / df_sentiment.groupby(['subreddit', 'type'])['content'].transform('count')

df_sentiment = df_sentiment.rename(columns={'content': 'count'})

df_sentiment_com = df_sentiment[df_sentiment['type'] == 'comment']
df_sentiment_sub = df_sentiment[df_sentiment['type'] == 'submission']

fig = px.bar(df_sentiment_sub, x='subreddit', y='score', color='sentiment', title='Submissions - sentiment analysis', labels={'subreddit': 'Subreddit', 'count': 'Posts', 'sentiment': 'Sentiment'}, height=600)
st.plotly_chart(fig, use_container_width=True)

fig = px.bar(df_sentiment_com, x='subreddit', y='score', color='sentiment', title='Comments - sentiment analysis', labels={'subreddit': 'Subreddit', 'count': 'Comments', 'sentiment': 'Sentiment'}, height=600)
st.plotly_chart(fig, use_container_width=True)
