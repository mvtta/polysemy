
# Polysemy Analysis: Agency

This project analyzes the polysemy of agency definitions across various disciplines using natural language processing and machine learning techniques.

## Main Components

1. Data Processing:
   * Loads exerpts and metadata from a CSV file
   * Cleans and preprocesses the data
2. Text Analysis:
   * Generates definitions using NLP for unstructured exerpts (Normalizing Data - may be a bad idea)
   * Extracts deep structures, scientific terms, feedback loops, and processes (Because of this)
   * Calculates similarity to a WORKING definition (needs to be revised, may be changed at: cybernetics_def = "We are agents, acting upon the world.")
3. Semantic Frame Extraction:
   * Uses FrameSemanticTransformer to extract semantic frames from definitions
4. Embedding Generation:
   * Creates embeddings for various aspects of each definition using SentenceTransformer models
5. Dimensionality Reduction:
   * Applies UMAP to reduce embeddings to 3D for visualization
6. Semiotic Freedom Index:
   * Calculates a Semiotic Freedom Index for each discipline
7. Visualization:
   * Creates a 3D scatter plot using Plotly
   * Points represent definitions, color-coded by discipline
   * Axes represent Operability, Abstraction, and Temporality
   * Includes hover text with definition details
8. Output:
   * Combines all processed data and saves to 'agency_analysis_results.csv' (Need to change for extractions.csv)
   * Generates a 3D visualization of the semantic space (Needs to be updated!)

## Usage

1. Ensure input data is in 'data_in/compiled.csv'
2. Run the main script to perform the analysis
3. View the generated 3D visualization
4. Find detailed results in 'data_out/extractions.csv'
5. Using just_plot.py to run just the visualization untilll stable

### Citations:
