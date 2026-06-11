import streamlit as st
import pandas as pd
import networkx as nx
import community
from pyvis import network as net
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import numpy as np
import json

path_network = '../data/reddit_network.gml'

path_com = '../data/reddit_network_communities.gml'
com_html = "../output/reddit_network_communities.html"
network_html = "../output/reddit_network.html"


def convert_to_counts(degrees):
    counts = dict.fromkeys(range(0, max(degrees) + 1), 0)
    for degree in degrees:
        counts[degree] += 1
    return counts


st.header('Network analysis')

G = nx.read_gml(path_network)
st.write("Number of nodes: ", G.number_of_nodes())
st.write("Number of edges: ", G.number_of_edges())
#
nt = net.Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
nt.from_nx(G)
HtmlFile = open(network_html, 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=800, width=1000, scrolling=True)

# write number of nodes and edges


degrees = [degree for node, degree in G.degree()]
counts = convert_to_counts(degrees)
st.write("Average degree: ", np.mean(degrees))

st.header("Degree distribution")

st.bar_chart(counts)




# Topic modeling
st.header("Topic modeling")
st.write("Analysis of 3 topics in comments")
ldaHtml = open("../output/lda_topics_3.html", 'r', encoding='utf-8')
components.html(ldaHtml.read(), height=800, width=1000, scrolling=True)

st.write("Analysis of 6 topics in comments")
ldaHtml = open("../output/lda_topics.html", 'r', encoding='utf-8')
components.html(ldaHtml.read(), height=800, width=1000, scrolling=True)
