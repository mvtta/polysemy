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

def load_data(data_file):
    data_headers = ['Author', 'Definition', 'Interpretation', 'Origin', 'Discipline', 'Year', 'Title', 'Doi', 'Search Keys', 'Url']
    data = pd.read_csv(data_file, names=data_headers, delimiter=';')
    return data

def process_data(data):
    cybernetics_def = "We are agents, acting upon the world."
    
    results = []
    for _, row in data.iterrows():
        excerpt = row['Definition']
        processor = AbstractExcerptProcessor(excerpt)
        
        result = {
            'Author': row['Author'],
            'Original Definition': excerpt,
            'Generated Definition': processor.generate_definition(),
            'Deep Structure': processor.extract_deep_structure(),
            'Scientific Terms': processor.extract_scientific_terms(),
            'Feedback Loops': processor.identify_feedback_loops(),
            'Processes': processor.extract_processes(),
            'Embeddings': processor.generate_embeddings(),
            'Similarity to Cybernetics': processor.similarity_to_cybernetics(cybernetics_def)
        }
        results.append(result)
    
    return pd.DataFrame(results)

# Add the new functions from the annex
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

# Todo: Take this away or update with just_plot.py version
def visualize_semantic_space(reduced_embeddings, disciplines, sf_scores=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_disciplines = list(set(disciplines))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_disciplines)))
    
    for i, discipline in enumerate(unique_disciplines):
        indices = [j for j, d in enumerate(disciplines) if d == discipline]
        points = reduced_embeddings[indices]
        
        size = 50
        if sf_scores and discipline in sf_scores:
            size = sf_scores[discipline] * 100
        
        ax.scatter(
            points[:, 0], 
            points[:, 1], 
            points[:, 2],
            color=colors[i],
            label=discipline,
            s=size
        )
    
    ax.set_xlabel('Operability')
    ax.set_ylabel('Abstraction')
    ax.set_zlabel('Temporality')
    ax.set_title('3D Semantic Space of Agency Definitions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('agency_semantic_space.png', dpi=300)
    plt.show()

def analyze_agency_polysemy(file_path):
    print("Loading data...")
    df = load_data(file_path)
    
    print("Processing data...")
    processed_data = process_data(df)
    
    print("Extracting semantic frames...")
    frames_df = extract_frames(df['Definition'].tolist())
    frames_df['Discipline'] = df['Discipline'].tolist()
    frames_df['Origin'] = df['Origin'].tolist()
    frames_df['Year'] = df['Year'].tolist()
    
    print("Generating embeddings...")
    embeddings = get_embeddings(df['Definition'].tolist())
    
    print("Reducing dimensions...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    print("Calculating Semiotic Freedom Index...")
    sf_scores = calculate_semiotic_freedom(frames_df)
    print("Semiotic Freedom by discipline:", sf_scores)
    
    print("Visualizing semantic space...")
    visualize_semantic_space(reduced_embeddings, df['Discipline'].tolist(), sf_scores)
    
    # Combine results
    results = pd.concat([processed_data, frames_df], axis=1)
    results['embedding_x'] = reduced_embeddings[:, 0]
    results['embedding_y'] = reduced_embeddings[:, 1]
    results['embedding_z'] = reduced_embeddings[:, 2]
    
    results.to_csv('agency_analysis_results.csv', index=False)
    
    print("Analysis complete! Results saved to 'agency_analysis_results.csv'")
    return results, reduced_embeddings, sf_scores

def main():
    input_file = 'data_in/compiled.csv'
    output_file = 'data_out/extractions.csv'

    print("Starting agency polysemy analysis...")
    results, reduced_embeddings, sf_scores = analyze_agency_polysemy(input_file)
    
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
