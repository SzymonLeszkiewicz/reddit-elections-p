import pandas as pd
from pprint import pprint
import networkx as nx

df_topics = pd.read_csv('../data/network_topic3.csv')
df_topics['topic'] = df_topics['topic'].astype(str)

G = nx.read_gml('../data/reddit_network_communities.gml')

df_graph = {}
for node in G.nodes(data=True):
    df_graph[node[0]] = node[1]['community']

df_graph = pd.DataFrame.from_dict(df_graph, orient='index')
df_graph.reset_index(inplace=True)
df_graph.columns = ['author', 'community']
df_graph['community'] = df_graph['community'].astype(str)

print(df_topics.shape, df_graph.shape)

df_topics.drop_duplicates(subset=['author'], inplace=True)
df_graph.drop_duplicates(subset=['author'], inplace=True)

print(df_topics.shape, df_graph.shape)

df_topics = df_topics.merge(df_graph, on='author', how='left')
df_topics.dropna(inplace=True)
