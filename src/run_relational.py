import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from frame_semantic_transformer import FrameSemanticTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer
import plotly.express as px
import os
import ast

# Load spaCy and sentence-transformers models
nlp = spacy.load("en_core_web_sm")
st_model = SentenceTransformer('all-MiniLM-L6-v2')

class AbstractExcerptProcessor:
    def __init__(self, excerpt):
        self.excerpt = excerpt
        self.doc = nlp(excerpt)
        
    def generate_definition(self):
        sentences = list(self.doc.sents)
        return sentences[0].text if sentences else ""
    
    def extract_deep_structure(self):
        return [(token.dep_, token.text) for token in self.doc if token.dep_ in ['nsubj', 'ROOT', 'dobj']]
    
    def extract_scientific_terms(self):
        return [chunk.text for chunk in self.doc.noun_chunks]
    
    def identify_feedback_loops(self):
        return [token.text for token in self.doc if token.pos_ == 'VERB']
    
    def extract_processes(self):
        return [(token.text, [child.text for child in token.children if child.dep_ == 'dobj']) 
                for token in self.doc if token.pos_ == 'VERB']
    
    def generate_embeddings(self):
        full_embedding = st_model.encode(self.excerpt)
        definition_embedding = st_model.encode(self.generate_definition())
        structure_embedding = st_model.encode(' '.join([t[1] for t in self.extract_deep_structure()]))
        adj_embedding = st_model.encode(' '.join([token.text for token in self.doc if token.pos_ == 'ADJ']))
        terms_embedding = st_model.encode(' '.join(self.extract_scientific_terms()))
        loops_embedding = st_model.encode(' '.join(self.identify_feedback_loops()))
        
        return {
            'full': full_embedding,
            'definition': definition_embedding,
            'structure': structure_embedding,
            'adjectives': adj_embedding,
            'terms': terms_embedding,
            'loops': loops_embedding
        }
    
    def similarity_to_cybernetics(self, cybernetics_def):
        excerpt_embedding = st_model.encode(self.excerpt)
        cybernetics_embedding = st_model.encode(cybernetics_def)
        return np.dot(excerpt_embedding, cybernetics_embedding) / (np.linalg.norm(excerpt_embedding) * np.linalg.norm(cybernetics_embedding))

    def extract_pov(self):
        pronouns = [token.text for token in self.doc if token.pos_ == 'PRON']
        perspective_indicators = [token.text for token in self.doc if token.lemma_ in ['think', 'believe', 'feel', 'perceive']]
        return {'pronouns': pronouns, 'perspective_indicators': perspective_indicators}
    
    def extract_reference_context(self):
        subject_object_pairs = []
        for sent in self.doc.sents:
            for token in sent:
                if token.dep_ == 'nsubj':
                    obj = [child for child in token.head.children if child.dep_ in ['dobj', 'pobj']]
                    if obj:
                        subject_object_pairs.append((token.text, obj[0].text))
        return subject_object_pairs
    
    def extract_context_implications(self):
        co_occurrences = []
        for sent in self.doc.sents:
            words = [token.text for token in sent if not token.is_stop and token.is_alpha]
            co_occurrences.extend([(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))])
        
        semantic_similarity = st_model.encode(self.excerpt)
        
        return {'co_occurrences': co_occurrences, 'semantic_similarity': semantic_similarity.tolist()}
    
    def extract_modifiers(self):
        adjectives = [token.text for token in self.doc if token.pos_ == 'ADJ']
        adverbs = [token.text for token in self.doc if token.pos_ == 'ADV']
        return {'adjectives': adjectives, 'adverbs': adverbs}
    
    def extract_output(self):
        verb_phrases = []
        resulting_actions = []
        for chunk in self.doc.noun_chunks:
            if chunk.root.head.pos_ == 'VERB':
                verb_phrases.append(chunk.root.head.text)
                resulting_actions.extend([child.text for child in chunk.root.head.children if child.dep_ in ['dobj', 'pobj']])
        return {'verb_phrases': verb_phrases, 'resulting_actions': resulting_actions}

    def extract_agency_sentences(self):
        return [sent for sent in self.doc.sents if any(token.text.lower() in ["agency", "agent"] for token in sent)]

def load_data(data_file):
    data_headers = ['Author', 'Definition', 'Interpretation', 'Origin', 'Discipline', 'Year', 'Title', 'Doi', 'Search Keys', 'Url']
    data = pd.read_csv(data_file, names=data_headers, delimiter=',')
    return data

def process_data(data):
    cybernetics_def = "We are agents, acting upon the world."
    
    results = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing data"):
        excerpt = row['Definition']
        processor = AbstractExcerptProcessor(excerpt)
        agency_sentences = processor.extract_agency_sentences()
        
        for sentence in agency_sentences:
            embeddings = processor.generate_embeddings()
            result = {
                'Author': row['Author'],
                'Original Definition': str(sentence),
                'Deep Structure': processor.extract_deep_structure(),
                'Scientific Terms': processor.extract_scientific_terms(),
                'Feedback Loops': processor.identify_feedback_loops(),
                'Processes': processor.extract_processes(),
                'Embeddings': embeddings,
                'Similarity to Cybernetics': processor.similarity_to_cybernetics(cybernetics_def),
                'PoV': processor.extract_pov(),
                'Reference X Context': processor.extract_reference_context(),
                'Context Implications': processor.extract_context_implications(),
                'Modifiers': processor.extract_modifiers(),
                'Output': processor.extract_output(),
                'Discipline': row['Discipline'],
                'Year': row['Year'],
                'embedding_x': embeddings['full'][0],
                'embedding_y': embeddings['full'][1],
                'embedding_z': embeddings['full'][2]
            }
            results.append(result)
    
    return pd.DataFrame(results)

def extract_frames(definitions):
    frame_transformer = FrameSemanticTransformer()
    frames_data = []
    
    for definition in tqdm(definitions, desc="Extracting frames"):
        try:
            result = frame_transformer.detect_frames(definition)
            frames = [frame.name for frame in result.frames]
            elements = []
            for frame in result.frames:
                for element in frame.frame_elements:
                    elements.append(f"{frame.name}:{element.name}")
            
            frames_data.append({
                'definition': definition,
                'frames': frames,
                'frame_elements': elements
            })
        except Exception as e:
            print(f"Error processing: {definition[:50]}... - {str(e)}")
            frames_data.append({
                'definition': definition,
                'frames': [],
                'frame_elements': []
            })
    
    return pd.DataFrame(frames_data)

def get_embeddings(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    batch_size = 8
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings.extend(batch_embeddings.numpy())
    
    return np.array(embeddings)

def reduce_dimensions(embeddings, n_components=3):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def calculate_semiotic_freedom(frames_df, discipline_col='Discipline'):
    sf_scores = {}
    
    for discipline in frames_df[discipline_col].unique():
        discipline_frames = frames_df[frames_df[discipline_col] == discipline]
        
        unique_frames = set()
        for frames in discipline_frames['frames']:
            unique_frames.update(frames)
        
        unique_elements = set()
        for elements in discipline_frames['frame_elements']:
            unique_elements.update(elements)
        
        if len(unique_frames) > 0:
            sf = np.log(len(unique_elements) / len(unique_frames))
            sf_scores[discipline] = max(0.1, min(1.0, sf))
        else:
            sf_scores[discipline] = 0.1
    
    return sf_scores

def truncate_text(text, max_length=50):
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def generate_hover_text(row):
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
    print("Creating 3D scatter plot...")

    # Parse string representations of dictionaries (if any)
    cols_to_parse = ['Context Implications', 'Modifiers', 'Output']
    for col in cols_to_parse:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except (ValueError, SyntaxError) as e:
                print(f"Could not parse column {col} correctly.")

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

def analyze_agency_polysemy(input_file):
    print("Loading data...")
    df = load_data(input_file)
    
    print("Data columns:", df.columns)  # Print column names to verify
    
    print("Processing data...")
    processed_data = process_data(df)
    
    print("Extracting semantic frames...")
    frames_df = extract_frames(df['Definition'].tolist())
    frames_df['Discipline'] = df['Discipline'].tolist()
    frames_df['Origin'] = df['Origin'].tolist()
    frames_df['Year'] = df['Year'].tolist()
    
    print("Calculating Semiotic Freedom Index...")
    sf_scores = calculate_semiotic_freedom(frames_df)
    print("Semiotic Freedom by discipline:", sf_scores)
    
    results = pd.concat([processed_data, frames_df], axis=1)
    
    results.to_csv('agency_analysis_results.csv', index=False)
    
    print("Analysis complete! Results saved to 'agency_analysis_results.csv'")
    return results, sf_scores

def main():
    input_file = 'data_in/compiled.csv'
    output_file = 'data_out/agency_analysis_results.csv'

    if not os.path.exists(output_file):
        print("Starting agency polysemy analysis...")
        results, sf_scores = analyze_agency_polysemy(input_file)
        print(f"Analysis complete. Results saved to {output_file}")
    else:
        print(f"Loading processed data from {output_file}...")
        results = pd.read_csv(output_file, on_bad_lines='skip', delimiter=',')
        
        # Parse string representations of dictionaries (if any)
        cols_to_parse = ['Context Implications', 'Modifiers', 'Output']
        for col in cols_to_parse:
            if results[col].dtype == 'object':
                try:
                    results[col] = results[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except (ValueError, SyntaxError) as e:
                    print(f"Could not parse column {col} correctly: {e}")

    print("Visualizing semantic space...")
    visualize_semantic_space(results)

if __name__ == "__main__":
    main()
