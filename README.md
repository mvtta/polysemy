
# Polysemy Analysis: Agency

This project analyzes the polysemy of agency definitions across various disciplines using natural language processing and machine learning techniques.

## run_temporal 

1. Data Processing:
   * Loads excerpts and metadata from a CSV file
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
   * Calculates a Semiotic Freedom Index for each discipline (Not used!)
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
5. Using just_plot.py to run just the visualization until stable

## run_relational

1. **Data Loading:**
    *   Loads definitions of agency from a CSV file (`data_in/compiled.csv`).
    *   The CSV file should have columns: `Author`, `Definition`, `Interpretation`, `Origin`, `Discipline`, `Year`, `Title`, `Doi`, `Search Keys`, `Url`.
2.  **Data Processing:**
    *   **Semantic Extraction:**  Uses spaCy and sentence-transformers to extract deep structure, scientific terms, feedback loops, processes, PoV, reference context, context implications, modifiers, and output from each definition.
    *   **Embedding Generation:** Generates embeddings for various parts of the definition using `sentence-transformers/all-MiniLM-L6-v2`.
    *   **Similarity Calculation:** Calculates the cosine similarity of each definition to a general definition of cybernetics ("We are agents, acting upon the world.").
    *   **Frame Extraction:**  Uses `frame_semantic_transformer` to identify semantic frames and frame elements within each definition.
3.  **Semiotic Freedom Index Calculation:**
    *   Calculates a semiotic freedom index for each discipline, based on the number of unique frames and frame elements.
4.  **Visualization:**
    *   Generates an interactive 3D scatter plot using Plotly Express to visualize the semantic space of agency definitions.
    *   The plot's axes represent the first three dimensions of the sentence embeddings (`embedding_x`, `embedding_y`, `embedding_z`).
    *   Points are colored by discipline.
    *   Hover text provides detailed information about each definition (sentence, author, year, implicators, modifiers, and output).
5.  **Output:**
    *   Saves the processed data, including extracted features and calculated metrics, to a CSV file (`data_out/agency_analysis_results.csv`).
    *   Displays the interactive 3D scatter plot in a web browser.

## Implementation Details

*   **Libraries:**
    *   `pandas`: For data loading and manipulation.
    *   `torch`: For PyTorch (used by transformers).
    *   `numpy`: For numerical operations.
    *   `transformers`: For loading and using pre-trained transformer models.
    *   `frame_semantic_transformer`: For semantic frame extraction.
    *   `sklearn`: For stop words.
    *   `umap`: For dimensionality reduction (UMAP).
    *   `matplotlib`:  Potentially imported but not directly used (leftover from previous versions).
    *   `tqdm`: For progress bars.
    *   `spacy`: For natural language processing.
    *   `sentence_transformers`: For generating sentence embeddings.
    *   `plotly.express`: For generating interactive plots.
    *   `os`: For file system operations.
    *   `ast`: For safely evaluating string representations of Python data structures.
*   **Models:**
    *   `en_core_web_sm` (spaCy): For NLP tasks.
    *   `all-MiniLM-L6-v2` (SentenceTransformer): For generating sentence embeddings.
*   **Data Structures:**
    *   Uses Pandas DataFrames to store and process the data.
    *   Uses dictionaries to store extracted features.

## Usage

1.  **Install Dependencies:**
    ```
    pip install pandas torch transformers frame_semantic_transformer scikit-learn umap-learn spacy sentence-transformers plotly
    python -m spacy download en_core_web_sm #this may not do the trick
    ```
2.  **Prepare Input Data:**
    *   Create a CSV file named `compiled.csv` and place it in the `data_in` directory.
    *   Ensure the CSV file has the correct columns (see "Data Loading" above).
3.  **Run the Script:**
    ```
    python your_script_name.py
    ```
4.  **View Results:**
    *   The processed data will be saved to `data_out/agency_analysis_results.csv`.
    *   The 3D scatter plot will be displayed in your web browser.

## Notes

*   The script automatically creates the `data_out` directory if it doesn't exist.
*   The `frame_semantic_transformer` library may require additional setup or dependencies.
*   The script handles missing or unparsable values in the input data by skipping the extraction of certain features.
*   This script performs both data processing and visualization.  To separate these steps, see the `visualization_script.py`.

### show_relational
## Functionality

1.  **Data Loading:**
    *   Loads processed data from a CSV file (`data_out/agency_analysis_results.csv`).
    *   This file is expected to be generated by the data processing script.
    *   The CSV should include `embedding_x`, `embedding_y`, `embedding_z`, `Discipline`, and other columns used to generate hover text.
2.  **Hover Text Generation:**
    *   Creates informative hover text for each data point in the scatter plot.
    *   Hover text includes the original sentence, author, year, implicators, modifiers, and output.
    *   Handles missing values gracefully.
3.  **Visualization:**
    *   Generates an interactive 3D scatter plot using Plotly Express.
    *   The plot's axes represent the first three dimensions of sentence embeddings (`embedding_x`, `embedding_y`, `embedding_z`).
    *   Points are colored by discipline.
    *   Hover text provides detailed information about each definition.
4.  **Error Handling:**
    * Checks if the input CSV file exists.
    * Gracefully handles malformed data in the CSV.

## Implementation Details

*   **Libraries:**
    *   `pandas`: For data loading and manipulation.
    *   `plotly.express`: For generating interactive plots.
    *   `ast`: For safely evaluating string representations of Python data structures.
    *   `os`: For file system operations.
*   **Data Structures:**
    *   Uses Pandas DataFrames to store and process the data.

## Usage

1.  **Ensure Data Processing is Complete:**
    *   **Important:**  You *must* run the data processing script first to generate the `data_out/agency_analysis_results.csv` file.  This script *only* visualizes data; it does not perform any data processing.
2.  **Install Dependencies:**
    ```
    pip install pandas plotly
    ```
3.  **Run the Script:**
    ```
    python your_visualization_script_name.py
    ```
4.  **View Results:**
    *   The 3D scatter plot will be displayed in your web browser.

## Notes

*   This script *requires* the output from the data processing script.
*   The `data_out/agency_analysis_results.csv` file must exist in the correct location.
*   This script separates the visualization step from the data processing pipeline, allowing for more modularity and flexibility.
*   The script handles missing or unparsable values in the input data by skipping the extraction of certain features.
  
### Citations:
