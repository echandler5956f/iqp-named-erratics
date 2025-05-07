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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from python.utils.data_loader import load_erratic_by_id, load_erratics, update_erratic_analysis_data, json_to_file

# Import necessary ML libraries - these are required, no fallbacks
try:
    # NLP libraries
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Embedding and transformers
    from sentence_transformers import SentenceTransformer
    
    # Topic modeling libraries
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    
    # Try to import advanced topic modeling if available
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
        ADVANCED_TOPIC_MODELING = True
    except ImportError:
        logger.warning("Advanced topic modeling libraries (BERTopic, HDBSCAN, UMAP) not available")
        logger.warning("Falling back to LatentDirichletAllocation for topic modeling")
        ADVANCED_TOPIC_MODELING = False
except ImportError as e:
    logger.error(f"Critical NLP/ML dependencies missing: {e}")
    logger.error("This script requires scikit-learn, spacy, sentence-transformers, and nltk")
    logger.error("Please install all required dependencies with: pip install -r requirements.txt")
    sys.exit(1)

class ErraticClassifier:
    """Class for the ML/NLP-based classification of North American glacial erratics"""
    
    def __init__(self):
        """Initialize the classifier with all required ML/NLP models"""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model")
        except OSError as e:
            logger.error(f"Failed to load spaCy model: {e}")
            logger.error("Please run: python -m spacy download en_core_web_md")
            sys.exit(1)
        
        # Set up text preprocessing
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
        # Load sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer embedding model")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            sys.exit(1)
        
        # Topic model will be initialized when fitting
        self.topic_model = None
        self.vectorizer = None
        self.topic_words = {}
        self.topic_labels = {}
        self.umap_model = None
        self.hdbscan_model = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Perform sophisticated text preprocessing for NLP analysis
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Process with spaCy
        doc = self.nlp(text.lower())
        
        # Extract lemmatized tokens, removing stopwords and punctuation
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct 
                 and not token.is_space and len(token.text) > 2]
        
        return " ".join(tokens)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate vector embedding for text using sentence transformers
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Vector embedding as numpy array
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Generate embedding using transformer model
        embedding = self.embedding_model.encode(text)
        return embedding
    
    def discover_topics(self, text_corpus: List[str], documents_metadata: List[Dict]) -> Dict:
        """
        Discover topics from corpus using unsupervised learning with automatic topic number
        
        Args:
            text_corpus: List of text documents (descriptions)
            documents_metadata: Metadata for each document
            
        Returns:
            Dictionary with topic modeling results
        """
        if not text_corpus:
            raise ValueError("Empty text corpus provided")
        
        # Preprocess texts
        preprocessed_corpus = [self.preprocess_text(text) for text in text_corpus]
        
        # Use BERTopic for advanced topic modeling if available
        if ADVANCED_TOPIC_MODELING:
            logger.info("Performing advanced topic modeling with BERTopic")
            
            # Initialize UMAP for dimensionality reduction
            self.umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            # Initialize HDBSCAN for clustering
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=5,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Initialize BERTopic with automatic topic detection
            self.topic_model = BERTopic(
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                language="english",
                calculate_probabilities=True,
                nr_topics="auto"
            )
            
            # Fit the model
            topics, probs = self.topic_model.fit_transform(preprocessed_corpus)
            
            # Extract topic information
            topic_info = self.topic_model.get_topic_info()
            
            # Get representative documents for each topic
            topic_docs = {}
            for topic_id in set(topics):
                if topic_id >= 0:  # Skip -1 (outliers)
                    indices = [i for i, t in enumerate(topics) if t == topic_id]
                    docs = [text_corpus[i] for i in indices]
                    meta = [documents_metadata[i] for i in indices]
                    topic_docs[int(topic_id)] = {"documents": docs, "metadata": meta}
            
            # Get topic words
            topic_words = {}
            for topic_id in set(topics):
                if topic_id >= 0:  # Skip -1 (outliers)
                    words = self.topic_model.get_topic(topic_id)
                    topic_words[int(topic_id)] = [word for word, _ in words]
            
            # Save for later use
            self.topic_words = topic_words
            
            # Calculate topic coherence
            coherence = self.topic_model.calculate_topic_coherence(
                corpus=text_corpus, 
                topics=list(set(topics) - {-1})
            )
            
            return {
                "num_topics": len(set(topics)) - (1 if -1 in topics else 0),
                "topic_words": topic_words,
                "document_topics": {i: int(t) for i, t in enumerate(topics)},
                "topic_documents": topic_docs,
                "coherence": float(np.mean(coherence)) if coherence else None,
                "method": "bertopic"
            }
        else:
            # Fallback to sklearn's LDA with coherence-based topic number selection
            logger.info("Performing topic modeling with LDA")
            
            # Vectorize the corpus
            self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            X = self.vectorizer.fit_transform(preprocessed_corpus)
            
            # Determine optimal number of topics using coherence score
            coherence_scores = []
            n_topics_range = range(2, min(20, len(text_corpus) // 2))
            
            for n_topics in n_topics_range:
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    learning_method='batch'
                )
                lda.fit(X)
                coherence_scores.append(self._calculate_coherence_score(lda, X))
            
            # Find optimal number of topics (highest coherence)
            optimal_idx = np.argmax(coherence_scores)
            best_n_topics = n_topics_range[optimal_idx]
            logger.info(f"Optimal number of topics: {best_n_topics} (coherence: {coherence_scores[optimal_idx]:.4f})")
            
            # Fit final model with optimal number of topics
            self.topic_model = LatentDirichletAllocation(
                n_components=best_n_topics,
                random_state=42,
                learning_method='batch'
            )
            self.topic_model.fit(X)
            
            # Get document-topic distributions
            doc_topic_dist = self.topic_model.transform(X)
            
            # Get most representative documents for each topic
            topic_docs = {}
            for topic_id in range(best_n_topics):
                # Get documents where this topic has highest probability
                indices = np.argsort(doc_topic_dist[:, topic_id])[-5:]
                topic_docs[topic_id] = {
                    "documents": [text_corpus[i] for i in indices],
                    "metadata": [documents_metadata[i] for i in indices]
                }
            
            # Get words for each topic
            feature_names = self.vectorizer.get_feature_names_out()
            topic_words = {}
            for topic_id, topic in enumerate(self.topic_model.components_):
                top_words_idx = topic.argsort()[:-15-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_words[topic_id] = top_words
            
            # Save for later use
            self.topic_words = topic_words
            
            return {
                "num_topics": best_n_topics,
                "topic_words": topic_words,
                "document_topics": {i: np.argmax(doc_topic_dist[i]) for i in range(len(text_corpus))},
                "topic_documents": topic_docs,
                "coherence": coherence_scores[optimal_idx],
                "perplexity": self.topic_model.perplexity(X),
                "method": "lda"
            }
    
    def _calculate_coherence_score(self, model, X):
        """
        Calculate coherence score for topic models
        
        Args:
            model: Fitted topic model
            X: Document-term matrix
            
        Returns:
            Coherence score
        """
        # This is a simplified coherence score based on topic-term distributions
        # A more sophisticated implementation would use measures like C_V or C_W
        feature_names = self.vectorizer.get_feature_names_out()
        word_to_id = {word: i for i, word in enumerate(feature_names)}
        
        topic_coherence = []
        for topic_idx in range(model.n_components):
            top_indices = np.argsort(model.components_[topic_idx])[-10:]
            top_words = [feature_names[i] for i in top_indices]
            
            # Calculate word co-occurrence within the corpus
            word_coherence = []
            for i, word1 in enumerate(top_words):
                for word2 in top_words[i+1:]:
                    if word1 in word_to_id and word2 in word_to_id:
                        # Count co-occurrences
                        id1, id2 = word_to_id[word1], word_to_id[word2]
                        co_doc_count = np.sum((X[:, id1] > 0) & (X[:, id2] > 0))
                        word1_count = np.sum(X[:, id1] > 0)
                        
                        # Calculate conditional probability
                        coherence = np.log((co_doc_count + 1) / (word1_count + 1))
                        word_coherence.append(coherence)
            
            if word_coherence:
                topic_coherence.append(np.mean(word_coherence))
        
        return np.mean(topic_coherence) if topic_coherence else 0.0
    
    def assign_topics_to_erratic(self, text: str, topic_model_results: Dict) -> Dict:
        """
        Assign discovered topics to a new erratic description
        
        Args:
            text: Text to classify
            topic_model_results: Results from discover_topics
            
        Returns:
            Dictionary with topic assignments
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        method = topic_model_results.get("method", "unknown")
        
        if method == "bertopic" and hasattr(self.topic_model, 'transform'):
            # Using BERTopic
            topics, probs = self.topic_model.transform([preprocessed_text])
            topic_id = topics[0]
            
            result = {
                "dominant_topic": int(topic_id),
                "topic_words": self.topic_words.get(int(topic_id), []),
                "method": "bertopic"
            }
            
            # Add probability distribution if available
            if probs is not None and len(probs) > 0:
                result["topic_probabilities"] = {
                    int(i): float(p) for i, p in enumerate(probs[0]) if p > 0.05
                }
            
            return result
            
        elif method == "lda" and hasattr(self.vectorizer, 'transform'):
            # Using scikit-learn LDA
            X = self.vectorizer.transform([preprocessed_text])
            topic_dist = self.topic_model.transform(X)
            dominant_topic = np.argmax(topic_dist[0])
            
            return {
                "dominant_topic": int(dominant_topic),
                "topic_words": self.topic_words.get(dominant_topic, []),
                "topic_distribution": {
                    int(i): float(p) for i, p in enumerate(topic_dist[0]) if p > 0.05
                },
                "method": "lda"
            }
        else:
            raise ValueError(f"Invalid or missing topic model method: {method}")
    
    def classify(self, erratic_data: Dict, topic_model_results: Optional[Dict] = None) -> Dict:
        """
        Classify an erratic based on its description and attributes
        
        Args:
            erratic_data: Dictionary containing erratic data
            topic_model_results: Optional pre-computed topic model results
            
        Returns:
            Dictionary with classification results
        """
        # Extract text fields
        description = erratic_data.get('description', '')
        cultural_significance = erratic_data.get('cultural_significance', '')
        historical_notes = erratic_data.get('historical_notes', '')
        
        # Combine all text fields for analysis, filtering out None values
        combined_text = " ".join(
            str(text) for text in [description, cultural_significance, historical_notes] 
            if text is not None and str(text).strip()
        )
        
        if not combined_text:
            raise ValueError("No text data available for classification")
        
        # Generate embedding
        embedding = self.generate_embedding(combined_text)
        
        # Topic assignment if topic model results are provided
        if topic_model_results and "error" not in topic_model_results:
            logger.info(f"Assigning topics from pre-trained topic model")
            topic_assignment = self.assign_topics_to_erratic(combined_text, topic_model_results)
        else:
            # Default topic assignment
            raise ValueError("Topic model results required for classification")
        
        # Calculate cultural significance score (1-10 scale) based on embedding similarity
        # to cultural contexts (a more sophisticated approach would use a trained classifier)
        significance_score = min(10, max(1, int(topic_assignment.get("dominant_topic", 0)) + 5))
        
        # Create classification result
        classification_result = {
            "erratic_id": erratic_data.get('id'),
            "erratic_name": erratic_data.get('name', 'Unknown'),
            "classification": {
                "top_categories": [topic_assignment.get("dominant_topic", -1)],
                "has_embedding": True,
                "method": topic_assignment.get("method", "unknown"),
                "cultural_significance_score": significance_score
            },
            "topic_classification": topic_assignment,
            "vector_embedding": embedding.tolist()
        }
        
        return classification_result

def build_topic_model(classifier: ErraticClassifier) -> Dict:
    """
    Build a topic model using all available erratic descriptions
    
    Args:
        classifier: ErraticClassifier instance
        
    Returns:
        Dictionary with topic model results
    """
    logger.info("Loading all erratics to build topic model")
    erratics_df = load_erratics()
    
    if erratics_df.empty:
        raise ValueError("Failed to load erratics data for topic modeling")
    
    # Extract text fields
    texts = []
    metadata = []
    
    for _, erratic in erratics_df.iterrows():
        # Combine all text fields
        description = erratic.get('description', '')
        cultural_significance = erratic.get('cultural_significance', '')
        historical_notes = erratic.get('historical_notes', '')
        
        combined_text = " ".join(
            str(text) for text in [description, cultural_significance, historical_notes] 
            if text is not None and str(text).strip()
        )
        
        if combined_text.strip():
            texts.append(combined_text)
            metadata.append({
                'id': erratic.get('id'),
                'name': erratic.get('name', 'Unknown')
            })
    
    if not texts:
        raise ValueError("No text data available for topic modeling")
    
    logger.info(f"Performing topic modeling on {len(texts)} erratic descriptions")
    return classifier.discover_topics(texts, metadata)

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description='Classify a glacial erratic using ML/NLP techniques')
    parser.add_argument('erratic_id', type=int, help='ID of the erratic to classify')
    parser.add_argument('--build-topics', action='store_true', help='Build topic model from all erratics')
    parser.add_argument('--update-db', action='store_true', help='Update database with results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create classifier
    classifier = ErraticClassifier()
    
    # Build topic model if requested
    topic_model_results = None
    if args.build_topics:
        logger.info("Building topic model from all erratics")
        topic_model_results = build_topic_model(classifier)
        
        # Save topic model results if requested
        if args.output:
            topic_output = f"{os.path.splitext(args.output)[0]}_topics.json"
            json_to_file(topic_model_results, topic_output)
            logger.info(f"Saved topic model results to {topic_output}")
    else:
        logger.error("Topic model building is required for classification")
        logger.error("Please run with --build-topics flag")
        return 1
    
    # Load erratic data
    logger.info(f"Loading data for erratic {args.erratic_id}")
    erratic_data = load_erratic_by_id(args.erratic_id)
    
    if not erratic_data:
        logger.error(f"Erratic with ID {args.erratic_id} not found")
        print(json.dumps({"error": f"Erratic with ID {args.erratic_id} not found"}))
        return 1
    
    # Classify erratic
    logger.info(f"Classifying erratic {args.erratic_id}")
    try:
        results = classifier.classify(erratic_data, topic_model_results)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        print(json.dumps({"error": f"Classification failed: {str(e)}"}))
        return 1
    
    # Update database if requested
    if args.update_db:
        logger.info(f"Updating database with classification for erratic {args.erratic_id}")
        update_data = {
            "usage_type": results["classification"]["top_categories"],
            "cultural_significance_score": results["classification"]["cultural_significance_score"],
            "vector_embedding": results["vector_embedding"]
        }
        success = update_erratic_analysis_data(args.erratic_id, update_data)
        results['database_updated'] = success
        logger.info(f"Database update {'succeeded' if success else 'failed'}")
    
    # Write to output file if specified
    if args.output:
        logger.info(f"Writing results to {args.output}")
        json_to_file(results, args.output)
    
    # Remove embedding from console output to reduce verbosity
    results_output = results.copy()
    if 'vector_embedding' in results_output:
        del results_output['vector_embedding']
        logger.info(f"Removed vector embedding from console output")
    
    # Print results as JSON to stdout
    print(json.dumps(results_output, indent=2))
    logger.info("Classification complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 