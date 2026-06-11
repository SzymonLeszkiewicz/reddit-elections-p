import os
from pprint import pprint
import networkx as nx
import pandas as pd
from pyvis import network as net

'''
script for building network from reddit data

input: csv file with reddit data

output: networkx graph: gml file
output: network vizL: pyvis html file
'''

FILE_NAME = '../data/network_activity.csv'
SAMPLING = False

df = pd.read_csv(FILE_NAME)
print('starting shape', df.shape)

if SAMPLING:
    df = df.sample(n=1000, random_state=1)
    print('sampled shape', df.shape)

df_submissions = df[df['type'] == 'submission']
df_comments = df[df['type'] == 'comment']
network = []
nodes_to_delete = []

for id in df_submissions['id'].unique():
    author = df_submissions[df_submissions['id'] == id]['author'].values[0]
    for com in df_comments[df_comments['id'] == id]['author'].unique():
        network.append((author, com))

G = nx.Graph()
G.add_edges_from(network)

# get rid of nodes with degree < 8
for node in G.nodes():
    if G.degree(node) < 8:
        nodes_to_delete.append(node)



# delete authors from df
df = df[~df['author'].isin(nodes_to_delete)]
print('delete nodes with no content', df.shape)
print('mum, of uniqe authors', df['author'].nunique())




pprint("Deleting nodes with degree < 8")
G.remove_nodes_from(nodes_to_delete)
print(G.number_of_nodes())

# make df with only one row per author
df_uniq = df.drop_duplicates(subset=['author'], keep='first')
print('delete nodes with no content', df_uniq.shape)
print('mum, of uniqe authors', df_uniq['author'].nunique())
# head
print(df_uniq.head())
# save df_uniq
df_uniq.to_csv('../output/network_author_uniq.csv', index=False)


nt = net.Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
nt.from_nx(G)
nt.show_buttons(filter_=['physics'])
nt.show("../output/reddit_network.html")

# save graph
nx.write_gml(G, "../data/reddit_network.gml")
