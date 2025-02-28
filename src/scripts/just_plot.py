import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
import umap

# Loading sentence-transformers model for generating embeddings
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(definitions):
    """Generate embeddings for definitions."""
    print("Generating embeddings...")
    return st_model.encode(definitions)

def reduce_dimensions(embeddings, n_components=3, random_state=42):
    """Reduce dimensions using UMAP."""
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(embeddings)

def truncate_text(text, max_length=50):
    """Truncate text to a specified maximum length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def generate_hover_text(row):
    """Generate hover text for each point."""
    sentence = truncate_text(row['Original Definition'])
    author = row['Author'] if 'Author' in row else "Unknown"
    year = row['Year'] if 'Year' in row else "Unknown"

    hover_text = f"<b>Sentence:</b> {sentence}<br>"
    hover_text += f"<b>Author:</b> {author}<br>"
    hover_text += f"<b>Year:</b> {year}"

    return hover_text

def visualize_semantic_space(data, reduced_embeddings, disciplines):
    """Visualize semantic space in 3D with hover text using Plotly."""
    print("Creating 3D scatter plot...")

    # Adding hover text column to the DataFrame
    data['hover_text'] = data.apply(generate_hover_text, axis=1)

    # Making a 3D scatter plot using Plotly
    fig = px.scatter_3d(
        data,
        x='embedding_x',  # X-coordinate column
        y='embedding_y',  # Y-coordinate column
        z='embedding_z',  # Z-coordinate column
        color="Discipline",
        hover_name='hover_text',
        title="3D Semantic Space of Agency Definitions",
        labels={'embedding_x': 'Operability', 'embedding_y': 'Abstraction', 'embedding_z': 'Temporality'}
    )

    # Updating layout for better visualization
    fig.update_layout(
        legend=dict(title="Disciplines", orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def main():
    # File path needs fixing
    input_file = 'data_out/extractions.csv'

    print("Loading processed data...")

    try:
        # Attempt to load data while skipping bad lines
        data = pd.read_csv(input_file, on_bad_lines='skip', delimiter=';')

        # Clean column headers by stripping leading/trailing spaces
        data.columns = data.columns.str.strip()

        print("Data columns:", data.columns)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if embedding columns exist; if not, generate them
    if 'embedding_x' not in data.columns or 'embedding_y' not in data.columns or 'embedding_z' not in data.columns:
        print("Embeddings not found. Generating embeddings...")

        if 'Original Definition' not in data.columns:
            print("Error: 'Original Definition' column is missing from the data.")
            return

        definitions = data['Original Definition'].tolist()
        embeddings = generate_embeddings(definitions)
        reduced_embeddings = reduce_dimensions(embeddings)

        # Add reduced dimensions back to the DataFrame
        data['embedding_x'] = reduced_embeddings[:, 0]
        data['embedding_y'] = reduced_embeddings[:, 1]
        data['embedding_z'] = reduced_embeddings[:, 2]

        # Save updated DataFrame for future use
        data.to_csv(input_file, index=False)
    else:
        # Use existing embedding columns
        reduced_embeddings = data[['embedding_x', 'embedding_y', 'embedding_z']].values

    # Ensure Discipline column exists and is properly populated
    if 'Discipline' not in data.columns or data['Discipline'].isnull().all():
        print("Error: 'Discipline' column is missing or empty.")
        return

    # Trim discipline names to ensure clean data
    data['Discipline'] = data['Discipline'].str.strip()

    disciplines = data['Discipline'].tolist()

    print("Generating visualization...")
    # Use existing embedding columns
    reduced_embeddings = data[['embedding_x', 'embedding_y', 'embedding_z']].values
    visualize_semantic_space(data, reduced_embeddings, disciplines)


if __name__ == "__main__":
    main()
