import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
import plotly.graph_objects as go

def load_data(file_path):
    """Load data from the CSV file and parse the frames column."""
    df = pd.read_csv(file_path)
    df['frames'] = df['frames'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    return df

def create_cooccurrence_network(df):
    """
    Create a co-occurrence network of frames.
    Frames are connected if they appear in the same sentence.
    """
    G = nx.Graph()
    
    # Count co-occurrences of frames within the same sentence
    cooccurrence_counter = Counter()
    for frames in df['frames']:
        for pair in combinations(frames, 2):
            cooccurrence_counter[tuple(sorted(pair))] += 1
    
    # Add nodes and edges to the graph
    for (frame1, frame2), weight in cooccurrence_counter.items():
        if not G.has_node(frame1):
            G.add_node(frame1)
        if not G.has_node(frame2):
            G.add_node(frame2)
        G.add_edge(frame1, frame2, weight=weight)
    
    return G

def apply_girvan_newman(G, level=1):
    """Apply Girvan-Newman algorithm and return communities at a specified level."""
    communities_generator = girvan_newman(G)
    for i in range(level):
        top_level_communities = next(communities_generator)
    community_map = {}
    for i, community in enumerate(top_level_communities):
        for node in community:
            community_map[node] = i
    return community_map


def apply_louvain(G):
    """Apply Louvain algorithm for community detection."""
    communities = greedy_modularity_communities(G)
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    return community_map


def visualize_communities_3d(G, pos_3d, community_map):
    """Visualize the graph with communities in 3D using Plotly."""
    
    # Extract node positions and community assignments
    x_nodes = [pos_3d[node][0] for node in G.nodes()]
    y_nodes = [pos_3d[node][1] for node in G.nodes()]
    z_nodes = [pos_3d[node][2] for node in G.nodes()]
    
    # Assign colors based on community
    node_colors = [community_map[node] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=1),
        hoverinfo='none'
    )
    
    # Create node traces
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=dict(
            size=[5 + len(list(G.neighbors(node))) for node in G.nodes()],  # Node size based on degree
            color=node_colors,
            colorscale='Viridis',
            colorbar=dict(title="Community"),
            line_width=0.5
        ),
        text=[f"{node}<br>Community: {community_map[node]}" for node in G.nodes()],
        hoverinfo='text'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Semantic Frame Co-occurrence Network with Communities (Girvan-Newman)",
                        titlefont_size=16,
                        showlegend=False,
                        margin=dict(l=0, r=0, b=0, t=40),
                        scene=dict(
                            xaxis=dict(showbackground=False),
                            yaxis=dict(showbackground=False),
                            zaxis=dict(showbackground=False)
                        )
                    ))
    
    return fig

def main():
    input_file = 'data_out/agency_analysis_results.csv'  # Input CSV file
    
    print("Loading data...")
    df = load_data(input_file)
    
    print("Creating co-occurrence network...")
    G = create_cooccurrence_network(df)
    
    # print("Applying Girvan-Newman algorithm...")
    # community_map = apply_girvan_newman(G)
 
    print("Applying Louvain algorithm...")
    community_map = apply_louvain(G)
    
    print("Calculating 3D positions...")
    pos_3d = nx.spring_layout(G, dim=3, seed=42)  # 3D layout
    
    print("Visualizing communities...")
    fig = visualize_communities_3d(G, pos_3d, community_map)
    
    print("Displaying the plot...")
    fig.show()

if __name__ == "__main__":
    main()
