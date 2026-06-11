import pandas as pd
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt
from time import sleep

df_topics = pd.read_csv('../data/network_topic3.csv')
G = nx.read_gml('../data/reddit_network_communities.gml')

print(df_topics.head())
print(df_topics.shape)

# convert graph to dataframe
df_graph = {}
for node in G.nodes(data=True):
    df_graph[node[0]] = node[1]['community']

df_graph = pd.DataFrame.from_dict(df_graph, orient='index')
df_graph.reset_index(inplace=True)
df_graph.columns = ['author', 'community']
print(df_graph.head())
print(df_graph.shape)

# merge dataframes
df_topics.drop_duplicates(subset=['author'], inplace=True)
df_graph.drop_duplicates(subset=['author'], inplace=True)

df_topics = df_topics.merge(df_graph, on='author', how='left')
df_topics.dropna(inplace=True)

#topic and community to str
df_topics['topic'] = df_topics['topic'].astype(str)
df_topics['community'] = df_topics['community'].astype(str)


print(df_topics.head())
print(df_topics.shape)

for i in df_topics['topic'].unique():
    df_topics = df_topics[df_topics['community'] != i]

    # plot histogram of topics
    df_topics['topic'].hist(bins=100)
    plt.show()
    sleep(1)

