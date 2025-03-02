import pandas as pd
import plotly.express as px
import ast
import os

def truncate_text(text, max_length=50):
    """Truncates text to a maximum length, adding '...' if needed."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def generate_hover_text(row):
    """Generates hover text for each data point in the scatter plot."""
    sentence = truncate_text(row['Original Definition'])
    author = row['Author'] if 'Author' in row else "Unknown"
    year = row['Year'] if 'Year' in row else "Unknown"

    implicators_str = ""
    if isinstance(row.get('Context Implications'), dict):
        implicators = row.get('Context Implications', {}).get('co_occurrences', [])
        implicators_str = ', '.join([f"{pair[0]}-{pair[1]}" for pair in implicators[:5]])

    modifiers_str = ""
    if isinstance(row.get('Modifiers'), dict):
        modifiers_adj = row.get('Modifiers', {}).get('adjectives', [])
        modifiers_adv = row.get('Modifiers', {}).get('adverbs', [])
        modifiers_str = ', '.join(modifiers_adj + modifiers_adv)

    output_str = ""
    if isinstance(row.get('Output'), dict):
        output_str = ', '.join(row.get('Output', {}).get('resulting_actions', []))

    hover_text = f"<b>Sentence:</b> {sentence}<br>"
    hover_text += f"<b>Author:</b> {author}<br>"
    hover_text += f"<b>Year:</b> {year}<br>"
    hover_text += f"<b>Implicators:</b> {implicators_str}<br>"
    hover_text += f"<b>Modifiers:</b> {modifiers_str}<br>"
    hover_text += f"<b>Output:</b> {output_str}"

    return hover_text

def visualize_semantic_space(data):
    """Creates and displays a 3D scatter plot of the semantic space."""
    print("Creating 3D scatter plot...")

    # Parse string representations of dictionaries (if any)
    cols_to_parse = ['Context Implications', 'Modifiers', 'Output']
    for col in cols_to_parse:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except (ValueError, SyntaxError) as e:
                print(f"Could not parse column {col} correctly: {e}")

    data['hover_text'] = data.apply(generate_hover_text, axis=1)

    fig = px.scatter_3d(
        data,
        x='embedding_x',
        y='embedding_y',
        z='embedding_z',
        color="Discipline",
        hover_name='hover_text',
        title="3D Semantic Space of Agency Definitions",
        labels={'embedding_x': 'Implicators', 'embedding_y': 'Modifiers', 'embedding_z': 'Output'}
    )

    fig.update_layout(
        legend=dict(title="Disciplines", orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def main():
    """Main function to load data and visualize the semantic space."""
    input_file = 'data_out/agency_analysis_results.csv'  # Expects the processed data

    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.  "
              f"Make sure you run the data processing script first to generate the results file.")
        return

    print(f"Loading processed data from {input_file}...")
    try:
        results = pd.read_csv(input_file, on_bad_lines='skip', delimiter=',')
    except FileNotFoundError:
        print(f"Error: Could not find the file {input_file}")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Parse string representations of dictionaries (if any)
    cols_to_parse = ['Context Implications', 'Modifiers', 'Output']
    for col in cols_to_parse:
        if col in results.columns and results[col].dtype == 'object':
            try:
                results[col] = results[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except (ValueError, SyntaxError) as e:
                print(f"Could not parse column {col} correctly: {e}")

    print("Visualizing semantic space...")
    visualize_semantic_space(results)

if __name__ == "__main__":
    main()
