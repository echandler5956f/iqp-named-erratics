#!/usr/bin/env python3
"""
Environment Test Script for Glacial Erratics Spatial Analysis

This script tests whether all required packages are properly installed and
validates that core functionality works correctly. Run this after setting
up the conda environment to ensure everything is working as expected.
"""

import os
import sys
import json
import logging
import importlib
from typing import Dict, List
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of required packages to test - name is the import name, package is the pip/conda package name
REQUIRED_PACKAGES = [
    # Core data processing
    {'name': 'numpy', 'package': 'numpy'},
    {'name': 'pandas', 'package': 'pandas'},
    {'name': 'geopandas', 'package': 'geopandas'},
    {'name': 'shapely', 'package': 'shapely'},
    
    # ML/NLP packages
    {'name': 'sklearn', 'package': 'scikit-learn'},
    {'name': 'nltk', 'package': 'nltk'},
    {'name': 'spacy', 'package': 'spacy'},
    {'name': 'sentence_transformers', 'package': 'sentence-transformers'},
    {'name': 'joblib', 'package': 'joblib'},
    {'name': 'bertopic', 'package': 'bertopic'},
    {'name': 'hdbscan', 'package': 'hdbscan'},
    {'name': 'umap', 'package': 'umap-learn'},
    
    # Specialized packages
    {'name': 'kneed', 'package': 'kneed'},
    {'name': 'plotly', 'package': 'plotly'}, # Needed for bertopic
    {'name': 'psycopg2', 'package': 'psycopg2-binary'},
    {'name': 'sqlalchemy', 'package': 'sqlalchemy'},
    {'name': 'pyproj', 'package': 'pyproj'},
    {'name': 'rtree', 'package': 'rtree'},
]

def check_packages() -> Dict[str, bool]:
    """Check if required packages can be imported."""
    results = {}
    
    for pkg in REQUIRED_PACKAGES:
        package_name = pkg['name']
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            results[package_name] = True
            logger.info(f"✓ Successfully imported {package_name} {getattr(module, '__version__', 'unknown version')}")
        except ImportError as e:
            results[package_name] = False
            logger.error(f"✗ Failed to import {package_name}: {e}")
        except Exception as e:
            results[package_name] = False
            logger.error(f"✗ Error with {package_name}: {e}")
            if "numpy.ufunc" in str(e) and package_name in ['scipy', 'sklearn', 'nltk']:
                logger.error("  This looks like a scipy/numpy version compatibility issue.")
                logger.error("  Try reinstalling scipy and numpy with compatible versions:")
                logger.error("  conda install -y numpy=1.24.3 scipy=1.11.4")
    
    return results

def safe_import(module_name):
    """Safely import a module, catching any exceptions."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.error(f"Error importing {module_name}: {e}")
        return None

def test_spacy_models() -> Dict[str, bool]:
    """Test if spaCy models can be loaded."""
    models = ['en_core_web_md', 'en_core_web_lg']
    results = {}
    
    spacy = safe_import('spacy')
    if not spacy:
        logger.error("✗ Failed to import spacy")
        for model in models:
            results[model] = False
        return results
    
    logger.info(f"Testing spaCy models (version {spacy.__version__})...")
    
    for model in models:
        try:
            nlp = spacy.load(model)
            # Test basic NLP functionality
            doc = nlp("This is a test sentence about glacial erratics in North America.")
            results[model] = True
            logger.info(f"✓ Successfully loaded spaCy model {model}")
        except Exception as e:
            results[model] = False
            logger.error(f"✗ Failed to load spaCy model {model}: {e}")
            logger.error(f"  To install this model, run: python -m spacy download {model}")
    
    return results

def test_sentence_transformers() -> bool:
    """Test if sentence-transformers can create embeddings."""
    st = safe_import('sentence_transformers')
    if not st:
        return False
    
    try:
        logger.info("Testing sentence-transformers...")
        
        model = st.SentenceTransformer('all-MiniLM-L6-v2')
        sentences = ["This is a glacial erratic.", "Native Americans used this boulder as a meeting place."]
        embeddings = model.encode(sentences)
        
        logger.info(f"✓ Successfully created embeddings with shape {embeddings.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create embeddings: {e}")
        return False

def test_geospatial() -> bool:
    """Test basic geospatial functionality."""
    gpd = safe_import('geopandas')
    shapely = safe_import('shapely')
    if not gpd or not shapely:
        return False
    
    try:
        from shapely.geometry import Point
        logger.info("Testing geospatial functionality...")
        
        # Create a simple GeoDataFrame
        points = [Point(-74.0, 40.7), Point(-118.2, 34.0)]  # NYC and LA
        data = {'name': ['New York', 'Los Angeles'], 'geometry': points}
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        # Test basic spatial operations
        buffer_gdf = gdf.buffer(0.1)  # ~11km buffer
        
        logger.info(f"✓ Successfully created and manipulated a GeoDataFrame")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to test geospatial functionality: {e}")
        return False

def test_sklearn() -> bool:
    """Test basic scikit-learn functionality."""
    sklearn = safe_import('sklearn')
    numpy = safe_import('numpy')
    if not sklearn or not numpy:
        return False
    
    try:
        from sklearn.cluster import DBSCAN
        logger.info("Testing scikit-learn...")
        
        # Create some sample data
        X = numpy.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [9, 7]])
        
        # Test clustering
        clustering = DBSCAN(eps=1.5, min_samples=2).fit(X)
        labels = clustering.labels_
        
        logger.info(f"✓ Successfully ran DBSCAN clustering, found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to test scikit-learn: {e}")
        return False

def test_nltk() -> bool:
    """Test basic NLTK functionality."""
    nltk = safe_import('nltk')
    if not nltk:
        return False
    
    try:
        logger.info("Testing NLTK...")
        
        # Try to ensure the required data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt data...")
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords data...")
            nltk.download('stopwords')
        
        # Test tokenization and stopword removal - using a simpler approach to avoid scipy
        try:
            from nltk.tokenize import word_tokenize
            sentence = "This large glacial erratic was transported here during the last ice age."
            tokens = word_tokenize(sentence)
            logger.info(f"✓ Successfully tokenized text with NLTK")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to tokenize text: {e}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to test NLTK: {e}")
        return False

def main():
    """Main function to test the environment."""
    print("\n" + "="*80)
    print("GLACIAL ERRATICS SPATIAL ANALYSIS - ENVIRONMENT TEST")
    print("="*80 + "\n")
    
    logger.info(f"Python version: {sys.version}")
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    logger.info(f"Conda environment: {conda_env or 'Not in a conda environment'}")
    
    if conda_env != 'iqp-py310':
        logger.warning("Not running in the expected 'iqp-py310' conda environment!")
    
    # Test package imports
    logger.info("\nTesting package imports...")
    package_results = check_packages()
    
    # Only continue with functionality tests if core packages are available
    core_packages_ok = all(package_results.get(pkg['name'], False) for pkg in REQUIRED_PACKAGES 
                         if pkg['name'] in ['numpy', 'pandas', 'sklearn', 'spacy'])
    
    if not core_packages_ok:
        logger.error("\nCritical packages are missing. Skipping functionality tests.")
        logger.error("Please fix the package installation issues first:")
        logger.error("1. Run: conda install -y numpy=1.24.3 scipy=1.11.4")
        logger.error("2. Run: conda install -y scikit-learn nltk spacy")
        logger.error("3. Run: python -m spacy download en_core_web_md")
        package_success = sum(package_results.values()) / len(package_results) if package_results else 0
        
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        print(f"Package imports: {package_success:.1%} ({sum(package_results.values())}/{len(package_results)} packages)")
        print(f"Functionality tests: Not run due to missing core packages")
        print(f"Overall score: {package_success/3:.1%}")
        
        print("\n✗ Several tests failed. The environment needs fixing before spatial analysis will work.")
        print("\n" + "="*80)
        
        return 1
    
    # Test spaCy models
    logger.info("\nTesting spaCy models...")
    spacy_results = test_spacy_models()
    
    # Test sentence transformers
    logger.info("\nTesting sentence-transformers...")
    transformers_result = test_sentence_transformers()
    
    # Test geospatial functionality
    logger.info("\nTesting geospatial functionality...")
    geospatial_result = test_geospatial()
    
    # Test scikit-learn functionality
    logger.info("\nTesting scikit-learn...")
    sklearn_result = test_sklearn()
    
    # Test NLTK functionality
    logger.info("\nTesting NLTK...")
    nltk_result = test_nltk()
    
    # Calculate overall results
    package_success = sum(package_results.values()) / len(package_results) if package_results else 0
    spacy_success = sum(spacy_results.values()) / len(spacy_results) if spacy_results else 0
    
    functionality_tests = [
        transformers_result, 
        geospatial_result,
        sklearn_result,
        nltk_result
    ]
    functionality_success = sum(1 for res in functionality_tests if res) / len(functionality_tests) if functionality_tests else 0
    
    overall_score = (package_success + spacy_success + functionality_success) / 3
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Package imports: {package_success:.1%} ({sum(package_results.values())}/{len(package_results)} packages)")
    print(f"spaCy models: {spacy_success:.1%} ({sum(spacy_results.values())}/{len(spacy_results)} models)")
    print(f"Functionality tests: {functionality_success:.1%} ({sum(1 for res in functionality_tests if res)}/{len(functionality_tests)} tests)")
    print(f"Overall score: {overall_score:.1%}")
    
    if overall_score == 1.0:
        print("\n✓ All tests passed! The environment is ready for spatial analysis.")
    elif overall_score >= 0.8:
        print("\n⚠ Most tests passed. The environment may work with some limitations.")
    else:
        print("\n✗ Several tests failed. The environment needs fixing before spatial analysis will work.")
        
        # Provide more specific recommendations
        print("\nRecommended fixes:")
        if package_success < 0.9:
            print("- Run setup_env.py to install missing packages:")
            print("  python setup_env.py")
        
        if "scipy" in str(package_results) and package_success < 1.0:
            print("- Fix numpy/scipy compatibility:")
            print("  conda install -y numpy=1.24.3 scipy=1.11.4")
            
        if spacy_success < 1.0:
            print("- Install missing spaCy models:")
            print("  python -m spacy download en_core_web_md")
            print("  python -m spacy download en_core_web_lg")
    
    print("\n" + "="*80)
    
    # Return exit code based on success
    return 0 if overall_score >= 0.8 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        print("\n✗ Test failed due to unexpected error. See log for details.")
        sys.exit(1) 