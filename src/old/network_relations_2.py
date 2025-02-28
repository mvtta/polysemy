import os
import re
import spacy
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# Load spaCy model for NER and keyword extraction
nlp = spacy.load("en_core_web_sm")

# Load sentence similarity model
sentence_similarity_model = SentenceTransformer("annakotarba/sentence-similarity")

def load_data(data_file):
    data_headers = ['Author', 'Definition', 'Interpretation', 'Origin', 'Discipline', 'Year', 'Title', 'Doi', 'Search Keys', 'Url']
    data = pd.read_csv(data_file, names=data_headers, delimiter=';')
    return data

def combine_sentences(definitions, interpretations):
    if not definitions or not interpretations:
        return []
    max_length = max(len(definitions), len(interpretations))
    padded_definitions = definitions + [''] * (max_length - len(definitions))
    padded_interpretations = interpretations + [''] * (max_length - len(interpretations))
    combined = [f"{definition} {interpretation}" for definition, interpretation in zip(padded_definitions, padded_interpretations)]
    return combined

def extract_keywords(text):
    text = re.sub(r'http\S+', '', text)
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha and len(token) > 2]
    return keywords

data_file = '/Users/mvtta/Desktop/polysemy/data_in/compiled.csv'
data = load_data(data_file)
combined_sentences = combine_sentences(data.Definition.tolist(), data.Interpretation.tolist())

# Encode sentences
sentence_embeddings = sentence_similarity_model.encode(combined_sentences)

## Create a graph
G = nx.random_geometric_graph(n=100, radius=0.125)

# Add nodes
for i, sentence in enumerate(combined_sentences):
    G.add_node(i, text=sentence[:50], discipline=data.Discipline.iloc[i], year=data.Year.iloc[i])

# Add edges based on similarity
similarity_threshold = 0.7
for i in range(len(sentence_embeddings)):
    for j in range(i+1, len(sentence_embeddings)):
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j]))
        if similarity > similarity_threshold:
            G.add_edge(i, j, weight=similarity)

# Calculate node properties
centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Use force-directed layout
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Create edge traces with reduced opacity
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create node trace
node_x, node_y = [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

# Adjust node sizes based on centrality
node_sizes = [np.log(1 + centrality[node] * 1000) * 10 for node in G.nodes()]

# Create a more diverse color map for disciplines
import colorsys
disciplines = list(set(data.Discipline))
n_colors = len(disciplines)
HSV_tuples = [(x*1.0/n_colors, 0.5, 0.5) for x in range(n_colors)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
color_map = {discipline: f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
             for discipline, (r,g,b) in zip(disciplines, RGB_tuples)}

node_colors = [color_map[G.nodes[node]['discipline']] for node in G.nodes()]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Color node points by the number of connections
for node, adjacencies in enumerate(G.adjacency()):
    node_trace.marker.color += tuple([len(adjacencies[1])])

# Prepare node text
node_hover_text = []
for node in G.nodes():
    node_info = (f"Node: {node}<br>"
                 f"Discipline: {G.nodes[node]['discipline']}<br>"
                 f"Sentence: {G.nodes[node]['text']}...<br>"
                 f"Connections: {len(list(G.neighbors(node)))}<br>"
                 f"Degree Centrality: {centrality[node]:.3f}")
    node_hover_text.append(node_info)

for i, sentence in enumerate(combined_sentences):
    G.add_node(i, text=sentence[:100], discipline=data.Discipline.iloc[i], year=data.Year.iloc[i])
    
node_trace.text = node_hover_text

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Network Graph", "Circular Layout"))

# Add traces to subplots
fig.add_trace(edge_trace, row=1, col=1)
fig.add_trace(node_trace, row=1, col=1)

# Create circular layout
pos_circular = nx.circular_layout(G)
edge_x_circular, edge_y_circular = [], []
for edge in G.edges():
    x0, y0 = pos_circular[edge[0]]
    x1, y1 = pos_circular[edge[1]]
    edge_x_circular.extend([x0, x1, None])
    edge_y_circular.extend([y0, y1, None])

edge_trace_circular = go.Scatter(
    x=edge_x_circular, y=edge_y_circular,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x_circular, node_y_circular = [], []
for node in G.nodes():
    x, y = pos_circular[node]
    node_x_circular.append(x)
    node_y_circular.append(y)

node_trace_circular = go.Scatter(
    x=node_x_circular, y=node_y_circular,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        reversescale=True,
        color=node_trace.marker.color,
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

node_trace_circular.text = node_hover_text

fig.add_trace(edge_trace_circular, row=1, col=2)
fig.add_trace(node_trace_circular, row=1, col=2)

# Update layout
fig.update_layout(
    title='Network Graph of Agency Across Disciplines',
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    annotations=[
        dict(text="Network graph", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002),
        dict(text="Circular layout", showarrow=False, xref="paper", yref="paper", x=0.505, y=-0.002)
    ],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    xaxis2=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis2=dict(showgrid=False, zeroline=False, showticklabels=False)
)

fig.show()