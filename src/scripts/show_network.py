import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations
from collections import Counter


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


def get_3d_positions(G):
    """Compute 3D positions for nodes using NetworkX's spring layout."""
    pos_3d = nx.spring_layout(G, dim=3, seed=42)  # 3D layout
    return pos_3d


def create_3d_plot(G, pos_3d):
    """Create a 3D network graph using Plotly."""
    
    # Extract node positions
    x_nodes = [pos_3d[node][0] for node in G.nodes()]
    y_nodes = [pos_3d[node][1] for node in G.nodes()]
    z_nodes = [pos_3d[node][2] for node in G.nodes()]
    
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
            color=[len(list(G.neighbors(node))) for node in G.nodes()],  # Color based on degree
            colorscale='Viridis',
            colorbar=dict(title="Node Connections"),
            line_width=0.5
        ),
        text=[f"{node}<br>Connections: {len(list(G.neighbors(node)))}" for node in G.nodes()],
        hoverinfo='text'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Semantic Frame Co-occurrence Network (3D)",
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
    
    print("Calculating 3D positions...")
    pos_3d = get_3d_positions(G)
    
    print("Generating 3D visualization...")
    fig = create_3d_plot(G, pos_3d)
    
    print("Displaying the plot...")
    fig.show()


if __name__ == "__main__":
    main()
