import pandas as pd
import streamlit as st
import plotly.express as px

EXCLUDED_AUTHORS = ['Zealousideal_Life206', 'random_user_216937', 'zoruunwise', 'dawidaloca']

st.set_page_config(layout="wide")
st.header('Reddit Activity - Polish Elections 2023')

df = pd.read_csv('../data/activity_today_dist2.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
df['date'] = df['date'].dt.date
df = df.sort_values(by='date')

fig = px.bar(
    df, x='date', color='subreddit',
    title='Activity over time',
    labels={'date': 'Date', 'subreddit': 'Subreddit', 'count': 'Count'},
    height=600,
)
st.plotly_chart(fig, use_container_width=True)


def get_n_top_users(df, n):
    return df[['author', 'subreddit']].groupby('author').count().sort_values(by='subreddit', ascending=False).head(n)


top_users = get_n_top_users(df, 9)

df3 = df.groupby(['date', 'author']).sum(numeric_only=True).reset_index()
df3 = df3[df3['author'].isin(top_users.index)]
df3 = df3[~df3['author'].isin(EXCLUDED_AUTHORS)]
fig = px.line(
    df3, x='date', y='score', color='author', markers=True,
    title='Upvotes by author',
    labels={'date': 'Date', 'score': 'Upvotes', 'author': 'User'},
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

df2 = df.groupby(['date', 'author'])['subreddit'].count().reset_index()
df2 = df2[~df2['author'].isin(EXCLUDED_AUTHORS)]
df2 = df2[df2['author'].isin(top_users.index)]
df2 = df2.rename(columns={'subreddit': 'count'})
fig = px.line(
    df2, x='date', y='count', color='author', markers=True,
    title='Activity by author',
    labels={'date': 'Date', 'count': 'Activity count', 'author': 'User'},
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

df_submissions = df[df['type'] == 'submission']
df_pom = df_submissions.groupby('date').sum(numeric_only=True).reset_index()
fig = px.bar(
    df_pom, x='date', y='score',
    title='Total submission score over time',
    labels={'date': 'Date', 'score': 'Total upvotes'},
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

mean_score = round(df_submissions['score'].mean(), 1)
mean_num_comments = round(df_submissions['replies'].mean(), 1)
mean_upvote_ratio = round(df_submissions['ratio'].mean(), 3)
mean_activity = int(df['subreddit'].value_counts().mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric('Avg upvotes / post', mean_score)
col2.metric('Avg comments / post', mean_num_comments)
col3.metric('Avg upvote ratio', mean_upvote_ratio)
col4.metric('Avg activity / day', mean_activity)
