#!/bin/bash
# Create a robust conda environment for the glacial erratics spatial analysis
# This script prioritizes conda-forge for better dependency handling.

# Parse command line args first
FIX_MODE=false
if [ "$1" == "-f" ]; then
  FIX_MODE=true
  echo "Running in fix mode - will only fix existing environment"
fi

echo "Glacial Erratics Python Environment Setup Script (conda-forge priority)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
  echo "Error: conda is not installed or not in PATH."
  echo "Please install conda from https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

ENV_NAME="iqp-py310"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
NLTK_DATA_DIR="$SCRIPT_DIR/nltk_data_local"

# Handle environment existence check differently based on mode
if [ "$FIX_MODE" = true ]; then
  # In fix mode, just check that the environment exists
  if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Error: $ENV_NAME environment not found but fix mode was specified."
    echo "Please create the environment first with: ./create_conda_env_strict.sh"
    exit 1
  fi
  echo "Fixing existing $ENV_NAME environment..."
else
  # In create mode, check if environment exists and ask to replace it
  if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    echo "Do you want to remove it and create a fresh environment? (y/n)"
    read answer
    if [ "$answer" == "y" ]; then
      echo "Removing existing environment..."
      conda remove --name $ENV_NAME --all -y
    else
      echo "Using existing environment. To fix issues, run this script with -f flag:"
      echo "  ./create_conda_env_strict.sh -f"
      echo "You can activate the environment with: conda activate $ENV_NAME"
      exit 0
    fi
  fi
  
  # Clean conda cache to ensure fresh package installs
  conda clean --all -y

  # Create environment with Python 3.10 and pip
  echo "Creating new conda environment $ENV_NAME with Python 3.10..."
  conda create --name $ENV_NAME python=3.10 pip -y
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# --- Conda Installations (prioritize conda-forge) ---
if [ "$FIX_MODE" = true ]; then
  echo "Ensuring pinned directory is removed before conda installs..."
  rm -rf $CONDA_PREFIX/conda-meta/pinned
fi

echo "Installing build tools (Cython, Compilers)..."
conda install -y -c conda-forge cython gcc_linux-64 gxx_linux-64

echo "Installing core scientific stack with conda (conda-forge priority)..."
conda install -y -c conda-forge joblib>=1.1.1 numpy=1.24.3 scipy=1.11.4 scikit-learn=1.2.2 pandas

echo "Installing geospatial packages with conda-forge..."
conda install -y -c conda-forge geopandas=0.10.2 shapely=1.8.0 pyproj=3.3.0 rtree=1.0.0 pyarrow>=8.0.0 requests>=2.27.0 rasterio>=1.2.10 matplotlib>=3.5.0

echo "Installing NLP/ML stack with conda-forge..."
conda install -y -c conda-forge nltk=3.7.0 kneed=0.8.5 sentence-transformers=2.2.2 hdbscan=0.8.29 umap-learn=0.5.3 spacy=3.4.0

echo "Installing Plotly (needed for bertopic)..."
conda install -y -c conda-forge plotly

echo "Installing DB packages with conda-forge..."
conda install -y -c conda-forge psycopg2 sqlalchemy

# --- Pinning and Pip Installations ---
echo "Pinning core conda packages..."
PINNED_FILE=$CONDA_PREFIX/conda-meta/pinned
rm -rf "$PINNED_FILE"
cat > "$PINNED_FILE" << EOL
numpy 1.24.*
scipy 1.11.*
scikit-learn 1.2.*
EOL

echo "Installing specific packages via pip (bertopic, python-dotenv, pgvector)..."
pip install --no-deps "bertopic[visualization]==0.15.0" # Added [visualization] for plotly dep if needed by bertopic itself
pip install "python-dotenv>=0.20.0"
pip install "pgvector>=0.2.0"

python -c "import bertopic; print(f'✓ BERTopic imported successfully (version {bertopic.__version__})')" || echo "✗ BERTopic import failed!"
python -c "import dotenv; print(f'✓ python-dotenv imported successfully')" || echo "✗ python-dotenv import failed!"
python -c "import pgvector.psycopg2; print(f'✓ pgvector.psycopg2 imported successfully')" || echo "✗ pgvector.psycopg2 import failed!"

echo "Installing spaCy models..."
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg

echo "Installing essential NLTK data to $NLTK_DATA_DIR ..."
mkdir -p "$NLTK_DATA_DIR"
python -c "
import nltk
import ssl

custom_nltk_path = '$NLTK_DATA_DIR'
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)

print(f'NLTK data path configured to: {nltk.data.path}')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

essential_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for pkg in essential_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}') # Check for tokenizers
        print(f'NLTK package {pkg} already found.')
    except nltk.downloader.DownloadError:
        try:
            nltk.data.find(f'corpora/{pkg}') # Check for corpora
            print(f'NLTK package {pkg} already found.')
        except nltk.downloader.DownloadError:
            print(f'Downloading NLTK package: {pkg} to {custom_nltk_path}')
            nltk.download(pkg, download_dir=custom_nltk_path, quiet=True)
print('NLTK essential data download process completed.')
" 

echo "Verifying NLTK tokenization works with custom path..."
python -c "
import nltk
custom_nltk_path = '$NLTK_DATA_DIR'
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)
try:
    tokens = nltk.word_tokenize(\"This is a test sentence.\")
    print(f'NLTK tokenization successful: {tokens}')
except Exception as e:
    print(f'NLTK tokenization failed: {e}')
    print(f'NLTK data path was: {nltk.data.path}')
    print(f'Please check {custom_nltk_path} for NLTK data.')
"

echo -e "\nConda environment '$ENV_NAME' has been set up."
echo "You can activate it with: conda activate $ENV_NAME"

echo -e "\nRunning environment test to verify setup..."
if [ -f "test_environment.py" ]; then
  python test_environment.py
else
  python -c "
import sys
print('Python version:', sys.version)
print('Testing critical imports:')
modules = ['numpy', 'pandas', 'geopandas', 'sklearn', 'spacy', 'bertopic', 'nltk', 'sentence_transformers', 'kneed', 'dotenv', 'pgvector']
import nltk
custom_nltk_path = '$NLTK_DATA_DIR'
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)
all_good = True
for m in modules:
    try:
        __import__(m)
        print(f'✓ {m} imported successfully')
    except ImportError as e:
        print(f'✗ {m} import failed: {e}')
        all_good = False
    except Exception as e:
        print(f'✗ Error importing {m}: {e}')
        all_good = False

try:
    tokens = nltk.word_tokenize('This large glacial erratic was transported here during the last ice age.')
    print(f'✓ NLTK tokenization works: {tokens[:3]}...')
except Exception as e:
    print(f'✗ NLTK tokenization failed: {e}')
    all_good = False

print('\\nBasic environment check ' + ('passed' if all_good else 'failed'))
"
fi

echo -e "\nTo fix issues in this environment, run again with -f flag:"
echo "./create_conda_env_strict.sh -f" 