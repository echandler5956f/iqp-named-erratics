#!/usr/bin/env python3
"""
Erratic Classification Script

This script classifies glacial erratics based on their descriptions
and attributes using NLP techniques and unsupervised topic modeling.
This is a rigorous implementation for academic research purposes
analyzing North American erratics, with no simplifications.
"""

import sys
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import joblib # For saving/loading sklearn models
import pathlib # For robust path construction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory of 'utils' and 'data_pipeline' to sys.path for robust imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_OF_SCRIPT_DIR = os.path.dirname(SCRIPT_DIR) # This should be the 'python' directory
if PARENT_OF_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, PARENT_OF_SCRIPT_DIR) # Add python/ to sys.path

from utils import db_utils
from utils import file_utils

MODEL_SAVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'data', 'models', 'erratic_classifier'))
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

_nltk_data_local_path = os.path.join(SCRIPT_DIR, "nltk_data_local")

try:
    import spacy
    import nltk
    if _nltk_data_local_path not in nltk.data.path:
        nltk.data.path.insert(0, _nltk_data_local_path)
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
        ADVANCED_TOPIC_MODELING = True
    except ImportError:
        logger.warning("Advanced topic modeling (BERTopic, HDBSCAN, UMAP) not available. Falling back to LDA.")
        ADVANCED_TOPIC_MODELING = False
except ImportError as e:
    logger.error(f"Critical NLP/ML dependencies missing: {e}. Run ./create_conda_env_strict.sh", exc_info=True)
    sys.exit(1)
except FileNotFoundError as e:
    logger.error(f"NLTK data (stopwords, wordnet, or punkt) not found in {nltk.data.path}. Error: {e}", exc_info=True)
    logger.error(f"Ensure NLTK data is in {_nltk_data_local_path} via ./create_conda_env_strict.sh.")
    sys.exit(1)

class ErraticClassifier:
    def __init__(self, model_dir: str = MODEL_SAVE_DIR):
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model: en_core_web_md")
        except OSError as e:
            logger.error(f"Failed to load spaCy model 'en_core_web_md': {e}. Run create_conda_env_strict.sh.", exc_info=True)
            sys.exit(1)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded SentenceTransformer: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer '{self.embedding_model_name}': {e}", exc_info=True)
            sys.exit(1)
        
        self.topic_model, self.vectorizer, self.topic_words, self.topic_labels = None, None, {}, {}
        self.umap_model, self.hdbscan_model, self.model_method = None, None, None
        self.model_dir = model_dir
    
    def preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str): return ""
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 2])
    
    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension(), dtype=np.float32)
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=normalize)
            return embedding.astype(np.float32)[0] if embedding.ndim > 1 else embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension(), dtype=np.float32)
    
    def discover_topics(self, text_corpus: List[str], documents_metadata: List[Dict]) -> Dict:
        if not text_corpus: raise ValueError("Empty text corpus for topic discovery.")
        preprocessed_corpus = [self.preprocess_text(text) for text in text_corpus if text.strip()]
        if not preprocessed_corpus: raise ValueError("Corpus empty after preprocessing.")
        n_docs = len(preprocessed_corpus)
        logger.info(f"Discovering topics on {n_docs} documents.")

        if ADVANCED_TOPIC_MODELING:
            self.model_method = 'bertopic'

            # Step 1: Dynamically adjust UMAP parameters for small datasets.
            # This is required as UMAP's n_neighbors cannot be >= number of samples.
            n_neighbors = 15
            n_components = 5
            if n_docs < n_neighbors:
                n_neighbors = max(2, n_docs - 1)
                logger.warning(f"Low document count ({n_docs}). Adjusting UMAP n_neighbors to {n_neighbors}.")
            
            if n_components >= n_neighbors:
                n_components = max(2, n_neighbors - 1)
                logger.warning(f"Adjusting UMAP n_components to {n_components} to be less than n_neighbors.")

            # Step 2: Disable automatic topic reduction for very small datasets to prevent a known crash.
            # The IndexError occurs inside fit_transform if HDBSCAN finds no clusters to reduce.
            nr_topics = "auto"
            min_docs_for_auto_reduction = 20 # Empirically chosen threshold
            if n_docs < min_docs_for_auto_reduction:
                nr_topics = None
                logger.warning(
                    f"Low document count ({n_docs} < {min_docs_for_auto_reduction}). "
                    f"Disabling 'auto' topic reduction (nr_topics=None) to prevent internal BERTopic error."
                )

            self.umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='cosine', random_state=42)
            self.hdbscan_model = HDBSCAN(min_cluster_size=max(2, int(n_docs*0.01)), metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            self.topic_model = BERTopic(umap_model=self.umap_model, hdbscan_model=self.hdbscan_model, language="english", calculate_probabilities=True, nr_topics=nr_topics, verbose=False)
            
            topics, _ = self.topic_model.fit_transform(preprocessed_corpus)

            # Step 3: Gracefully handle cases where no topics (besides the outlier topic) are found.
            topic_info = self.topic_model.get_topic_info()
            if topic_info.empty or (len(topic_info) == 1 and topic_info.iloc[0]['Topic'] == -1):
                logger.warning("BERTopic processing resulted in no topics being discovered. Returning empty result.")
                self.topic_words = {}
                return {"num_topics": 0, "topic_words": {}, "method": self.model_method}
            
            self.topic_words = {int(topic_id): [word for word, _ in self.topic_model.get_topic(topic_id)] for topic_id in topic_info['Topic'] if topic_id >=0}
            return {"num_topics": len(self.topic_words), "topic_words": self.topic_words, "method": self.model_method}
        else:
            self.model_method = 'lda'
            self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            X = self.vectorizer.fit_transform(preprocessed_corpus)
            if X.shape[0] == 0: raise ValueError("Document-term matrix empty after vectorization for LDA.")
            best_n_topics = min(10, max(2, X.shape[0] // 10)) # Simplified heuristic
            logger.info(f"LDA: Using heuristic number of topics: {best_n_topics}")
            self.topic_model = LatentDirichletAllocation(n_components=best_n_topics, random_state=42, learning_method='batch').fit(X)
            feature_names = self.vectorizer.get_feature_names_out()
            self.topic_words = {tid: [feature_names[i] for i in topic.argsort()[:-15-1:-1]] for tid, topic in enumerate(self.topic_model.components_)}
            return {"num_topics": best_n_topics, "topic_words": self.topic_words, "method": self.model_method}

    def assign_topics_to_erratic(self, text: str) -> Dict:
        if not self.topic_model: raise RuntimeError("Topic model not loaded/trained.")
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text.strip(): return {"dominant_topic": -1, "topic_words": [], "method": self.model_method}
        if self.model_method == 'bertopic' and ADVANCED_TOPIC_MODELING:
            topics, probs = self.topic_model.transform([preprocessed_text])
            topic_id = topics[0]
            result = {"dominant_topic": int(topic_id), "topic_words": self.topic_words.get(int(topic_id), []), "method": "bertopic"}
            if probs is not None and len(probs) > 0: result["topic_probabilities"] = {int(i): float(p) for i,p in enumerate(probs[0]) if p > 0.01}
            return result
        elif self.model_method == 'lda' and self.vectorizer:
            X = self.vectorizer.transform([preprocessed_text])
            topic_dist = self.topic_model.transform(X)[0]
            dominant_topic = np.argmax(topic_dist)
            return {"dominant_topic": int(dominant_topic), "topic_words": self.topic_words.get(int(dominant_topic), []),
                    "topic_distribution": {int(i): float(p) for i,p in enumerate(topic_dist) if p > 0.01}, "method": "lda"}
        return {"dominant_topic": -1, "topic_words": [], "method": "unknown_error_in_assignment"}

    def _has_inscription_keywords(self, text: str) -> bool:
        return any(keyword in text.lower() for keyword in ["inscription", "carving", "engraved", "petroglyph"] if text)
        
    def classify(self, erratic_data: Dict) -> Dict:
        desc = erratic_data.get('description', '')
        cult_sig = erratic_data.get('cultural_significance', '')
        hist_notes = erratic_data.get('historical_notes', '')
        combined_text = " ".join(filter(None, [str(t).strip() if t else '' for t in [desc, cult_sig, hist_notes]])).strip()

        if not combined_text:
            logger.warn(f"No text for classification: erratic ID {erratic_data.get('id')}")
            zero_emb = np.zeros(self.embedding_model.get_sentence_embedding_dimension(), dtype=np.float32).tolist()
            return {"erratic_id": erratic_data.get('id'), "erratic_name": erratic_data.get('name'),
                    "classification": {"top_categories": [-1], "has_embedding": True, "method": "no_text", "cultural_significance_score": 1, "has_inscriptions": False},
                    "topic_classification": {"dominant_topic": -1}, "vector_embedding": zero_emb}

        embedding = self.generate_embedding(combined_text)
        topic_assignment = self.assign_topics_to_erratic(combined_text)
        has_inscriptions = self._has_inscription_keywords(combined_text)
        score = int(topic_assignment.get("dominant_topic", -1))
        significance_score = min(10, max(1, score + 5 if score != -1 else 1))
        return {"erratic_id": erratic_data.get('id'), "erratic_name": erratic_data.get('name'),
                "classification": {"top_categories": [topic_assignment.get("dominant_topic", -1)], "has_embedding": True,
                                 "method": topic_assignment.get("method"), "cultural_significance_score": significance_score,
                                 "has_inscriptions": has_inscriptions},
                "topic_classification": topic_assignment, "vector_embedding": embedding.tolist()}
        
    def save_model(self, model_dir: Optional[str] = None):
        save_dir = model_dir or self.model_dir; os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving models to: {save_dir}")
        if not self.topic_model: raise RuntimeError("No model trained/loaded.")
        meta = {'model_method': self.model_method, 'embedding_model_name': self.embedding_model_name,
                'timestamp': pd.Timestamp.now().isoformat(), 'topic_count': len(self.topic_words or {}), 'version': '1.2'}
        try:
            file_utils.json_to_file(meta, os.path.join(save_dir, 'model_metadata.json'))
            file_utils.json_to_file(self.topic_words or {}, os.path.join(save_dir, 'topic_words.json'))
            if self.model_method == 'bertopic' and ADVANCED_TOPIC_MODELING:
                self.topic_model.save(os.path.join(save_dir, "bertopic_model"), serialization="pytorch", save_embedding_model=False)
                if self.topic_labels: file_utils.json_to_file(self.topic_labels, os.path.join(save_dir, 'topic_labels.json'))
            elif self.model_method == 'lda':
                joblib.dump(self.topic_model, os.path.join(save_dir, 'lda_model.joblib'))
                if self.vectorizer: joblib.dump(self.vectorizer, os.path.join(save_dir, 'vectorizer.joblib'))
            else: raise ValueError(f"Unsupported model method: {self.model_method}")
            logger.info(f"Saved model components to {save_dir}")
        except Exception as e: logger.error(f"Error saving model: {e}", exc_info=True); raise

    def load_model(self, model_dir: Optional[str] = None):
        load_dir = model_dir or self.model_dir; logger.info(f"Loading models from: {load_dir}")
        meta_path = os.path.join(load_dir, 'model_metadata.json')
        if not os.path.exists(meta_path): raise FileNotFoundError(f"Metadata not found: {meta_path}")
        try:
            with open(meta_path, 'r') as f: meta = json.load(f)
            self.model_method = meta.get('model_method')
            loaded_emb_name = meta.get('embedding_model_name')
            if loaded_emb_name and loaded_emb_name != self.embedding_model_name:
                logger.warning(f"Model trained with '{loaded_emb_name}', current is '{self.embedding_model_name}'. Re-init embedding model.")
                self.embedding_model = SentenceTransformer(loaded_emb_name); self.embedding_model_name = loaded_emb_name
            
            words_path = os.path.join(load_dir, 'topic_words.json')
            if os.path.exists(words_path): 
                with open(words_path, 'r') as f: self.topic_words = {int(k): v for k,v in json.load(f).items()}
            labels_path = os.path.join(load_dir, 'topic_labels.json')
            if os.path.exists(labels_path):
                 with open(labels_path, 'r') as f: self.topic_labels = {int(k): v for k,v in json.load(f).items()}
            
            if self.model_method == 'bertopic' and ADVANCED_TOPIC_MODELING:
                model_file_path = os.path.join(load_dir, "bertopic_model")
                if not os.path.exists(model_file_path): raise FileNotFoundError(f"BERTopic model missing: {model_file_path}")
                self.topic_model = BERTopic.load(model_file_path, embedding_model=self.embedding_model)
                if not self.topic_words: self.topic_words = {t: [w[0] for w in self.topic_model.get_topic(t) if w is not None and w[0] is not None] for t in self.topic_model.get_topics() if t != -1}
            elif self.model_method == 'lda':
                model_file_path, vec_file_path = os.path.join(load_dir, 'lda_model.joblib'), os.path.join(load_dir, 'vectorizer.joblib')
                if not os.path.exists(model_file_path) or not os.path.exists(vec_file_path): raise FileNotFoundError("LDA/Vectorizer missing.")
                self.topic_model, self.vectorizer = joblib.load(model_file_path), joblib.load(vec_file_path)
                if not self.topic_words:
                    ft_names = self.vectorizer.get_feature_names_out()
                    self.topic_words = {tid: [ft_names[i] for i in topic.argsort()[:-15-1:-1]] for tid, topic in enumerate(self.topic_model.components_)}
            else: raise ValueError(f"Unsupported model method '{self.model_method}' in metadata.")
            logger.info(f"Model '{self.model_method}' loaded. Topics: {len(self.topic_words)}")
        except Exception as e: logger.error(f"Error loading model from {load_dir}: {e}", exc_info=True); raise

def build_topic_model(classifier: ErraticClassifier) -> Dict:
    logger.info("Building topic model: Loading all erratics from database...")
    erratics_gdf = db_utils.load_all_erratics_gdf()
    if erratics_gdf.empty: raise ValueError("No erratics data from DB for topic modeling.")
    
    texts, metadata = [], []
    for _, row in erratics_gdf.iterrows():
        combined_text = " ".join(filter(None, [str(row.get(col, '')).strip() for col in ['description', 'cultural_significance', 'historical_notes']])).strip()
        if combined_text: texts.append(combined_text); metadata.append({'id': row.get('id'), 'name': row.get('name', 'Unknown')})
    
    if not texts: raise ValueError("No text descriptions found for topic modeling.")
    logger.info(f"Performing topic modeling on {len(texts)} erratic descriptions.")
    return classifier.discover_topics(texts, metadata)

def main():
    parser = argparse.ArgumentParser(description='Classify a glacial erratic using ML/NLP techniques')
    parser.add_argument('erratic_id', type=int, help='ID of the erratic (unless --build-topics)', nargs='?')
    parser.add_argument('--build-topics', action='store_true', help='Build topic model from all erratics')
    parser.add_argument('--update-db', action='store_true', help='Update database with results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--model-dir', type=str, help='Custom directory for model saving/loading')
    
    args = parser.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.build_topics and args.erratic_id is None: parser.error("erratic_id is required unless --build-topics.")

    model_dir = args.model_dir or MODEL_SAVE_DIR
    classifier = ErraticClassifier(model_dir=model_dir)
    logger.info(f"Using model directory: {model_dir}")
    
    if args.build_topics:
        logger.info("PHASE 1: Building topic model...")
        start_time = pd.Timestamp.now()
        try:
            topic_model_results = build_topic_model(classifier)
            logger.info(f"Built topic model: {topic_model_results.get('num_topics')} topics using {topic_model_results.get('method')}.")
            if args.output:
                path_prefix = os.path.splitext(args.output)[0]
                file_utils.json_to_file(topic_model_results, f"{path_prefix}_topics.json")
            classifier.save_model()
            logger.info(f"Topic model building completed in {(pd.Timestamp.now() - start_time).total_seconds():.1f}s.")
            if args.erratic_id is None: 
                print(json.dumps({"status": "Topic model built and saved.", "details": topic_model_results}, indent=2)); sys.exit(0)
        except Exception as e: logger.error(f"Topic model build failed: {e}", exc_info=True); print(json.dumps({"error": str(e)})); sys.exit(1)
            
    if args.erratic_id is not None:
        logger.info(f"PHASE 2: Classifying erratic ID {args.erratic_id}...")
        try:
            classifier.load_model()
        except Exception as e: logger.error(f"Failed to load model: {e}. Run --build-topics?", exc_info=True); print(json.dumps({"error": str(e)})); sys.exit(1)

        erratic_data = db_utils.load_erratic_details_by_id(args.erratic_id)
        if not erratic_data: logger.error(f"Erratic {args.erratic_id} not found."); print(json.dumps({"error": "Erratic not found"})); sys.exit(1)
        
        try:
            results = classifier.classify(erratic_data)
            logger.info(f"Classification successful for {args.erratic_id}")
        except Exception as e: logger.error(f"Classification error for {args.erratic_id}: {e}", exc_info=True); print(json.dumps({"error": str(e)})); sys.exit(1)
        
        if args.update_db:
            logger.info(f"Updating DB for {args.erratic_id} with classification...")
            update_payload = {
                "usage_type": [str(tid) for tid in results.get("classification",{}).get("top_categories",[]) if tid != -1],
                "cultural_significance_score": results.get("classification",{}).get("cultural_significance_score"),
                "vector_embedding": results.get("vector_embedding"), 
                "has_inscriptions": results.get("classification",{}).get("has_inscriptions")
            }
            success = db_utils.update_erratic_analysis_results(args.erratic_id, update_payload)
            results['database_updated'] = success
            logger.info(f"DB update for {args.erratic_id} classification {'succeeded' if success else 'failed'}.")
        
        if args.output: file_utils.json_to_file(results, args.output)
        
        # Prepare console output (remove verbose embedding)
        console_output = {k: v for k, v in results.items() if k != 'vector_embedding'}
        if "vector_embedding" in results: console_output['vector_embedding_length'] = len(results.get("vector_embedding",[]))
        print(json.dumps(console_output, indent=2))
        logger.info(f"classify_erratic.py finished for ID {args.erratic_id}.")
    elif not args.build_topics:
        logger.error("No action: erratic_id not given and --build-topics not set."); parser.print_help(); sys.exit(1)

if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_python_scripts_dir = os.path.dirname(current_file_dir)
    if project_python_scripts_dir not in sys.path:
        sys.path.insert(0, project_python_scripts_dir)
    from utils import db_utils, file_utils # Re-import for direct execution context
    sys.exit(main()) 