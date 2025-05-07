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

# Handle environment existence check differently based on mode
if [ "$FIX_MODE" = true ]; then
  # In fix mode, just check that the environment exists
  if ! conda env list | grep -q "iqp-py310"; then
    echo "Error: iqp-py310 environment not found but fix mode was specified."
    echo "Please create the environment first with: ./create_conda_env_strict.sh"
    exit 1
  fi
  echo "Fixing existing iqp-py310 environment..."
else
  # In create mode, check if environment exists and ask to replace it
  if conda env list | grep -q "iqp-py310"; then
    echo "Environment 'iqp-py310' already exists."
    echo "Do you want to remove it and create a fresh environment? (y/n)"
    read answer
    if [ "$answer" == "y" ]; then
      echo "Removing existing environment..."
      conda remove --name iqp-py310 --all -y
    else
      echo "Using existing environment. To fix issues, run this script with -f flag:"
      echo "  ./create_conda_env_strict.sh -f"
      echo "You can activate the environment with: conda activate iqp-py310"
      exit 0
    fi
  fi
  
  # Clean conda cache to ensure fresh package installs
  conda clean --all -y

  # Create environment with Python 3.10 and pip
  echo "Creating new conda environment with Python 3.10..."
  conda create --name iqp-py310 python=3.10 pip -y
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate iqp-py310

# --- Conda Installations (prioritize conda-forge) ---
# Ensure pinned directory does not exist before running conda install commands in fix mode
if [ "$FIX_MODE" = true ]; then
  echo "Ensuring pinned directory is removed before conda installs..."
  rm -rf $CONDA_PREFIX/conda-meta/pinned
fi

# Install build tools first
echo "Installing build tools (Cython, Compilers)..."
conda install -y -c conda-forge cython gcc_linux-64 gxx_linux-64

# Install core scientific stack with specific versions via conda-forge where possible
echo "Installing core scientific stack with conda (conda-forge priority)..."
conda install -y -c conda-forge joblib>=1.1.1 numpy=1.24.3 scipy=1.11.4 scikit-learn=1.2.2 pandas

# Install geospatial packages via conda-forge
echo "Installing geospatial packages with conda-forge..."
conda install -y -c conda-forge geopandas=0.10.2 shapely=1.8.0 pyproj=3.3.0 rtree=1.0.0

# Install NLP/ML packages via conda-forge where possible
echo "Installing NLP/ML stack with conda-forge..."
conda install -y -c conda-forge nltk=3.7.0 kneed=0.8.5
# Install sentence-transformers from conda-forge, hoping it pulls compatible deps
conda install -y -c conda-forge sentence-transformers=2.2.2
conda install -y -c conda-forge hdbscan=0.8.29 umap-learn=0.5.3
conda install -y -c conda-forge spacy=3.4.0

# Plotly is needed for bertopic
conda install -y -c conda-forge plotly

# Install database packages via conda-forge (split for solver)
echo "Installing DB packages with conda-forge..."
conda install -y -c conda-forge psycopg2 sqlalchemy

# --- Pinning and Pip Installations (Minimal) ---

# Create the conda pinned file correctly (still useful for stability)
echo "Pinning core conda packages..."
PINNED_FILE=$CONDA_PREFIX/conda-meta/pinned
# Remove existing directory/file first (important in fix mode)
rm -rf "$PINNED_FILE"
# Create the pinned file with specifications
cat > "$PINNED_FILE" << EOL
numpy 1.24.*
scipy 1.11.*
scikit-learn 1.2.*
EOL

# Install bertopic via pip as conda version might be tricky
echo "Installing bertopic via pip..."
pip install --no-deps "bertopic==0.15.0"

# Verify bertopic import (basic check)
python -c "import bertopic; print(f'✓ BERTopic imported successfully (version {bertopic.__version__})')" || echo "✗ BERTopic import failed!"

# Install spaCy models
echo "Installing spaCy models..."
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg

# Install NLTK data - essentials only
echo "Installing essential NLTK data..."
python -c '
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Essential data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
print("NLTK essential data downloaded.")
'

# Verify NLTK punkt is working
echo "Verifying NLTK tokenization works..."
python -c '
import nltk
try:
    tokens = nltk.word_tokenize("This is a test sentence.")
    print(f"NLTK tokenization successful: {tokens}")
except Exception as e:
    print(f"NLTK tokenization failed: {e}")
'

# Success message
echo -e "\nConda environment 'iqp-py310' has been set up with conda-forge priority."
echo "You can activate it with: conda activate iqp-py310"

# Add environment variable to bashrc (optional, less critical if pip use is minimal)
echo "Setting up environment for future use..."
PIP_CONSTRAINTS_FILE=$HOME/.pip/constraints.txt # Still create constraints file for manual pip installs
mkdir -p $(dirname $PIP_CONSTRAINTS_FILE)
cat > $PIP_CONSTRAINTS_FILE << EOL
numpy==1.24.3
scipy==1.11.4
sklearn==1.2.2
EOL
if ! grep -q "export PIP_CONSTRAINT" ~/.bashrc; then
  echo "" >> ~/.bashrc
  echo "# Added by glacial erratics setup - ensures package compatibility for pip" >> ~/.bashrc
  echo "export PIP_CONSTRAINT=$PIP_CONSTRAINTS_FILE" >> ~/.bashrc
  echo -e "\nAdded environment variable to ~/.bashrc to maintain compatibility for pip."
  echo "You may need to restart your shell or run: source ~/.bashrc"
fi

# Run comprehensive test script
echo -e "\nRunning environment test to verify setup..."
if [ -f "test_environment.py" ]; then
  python test_environment.py
else
  # Basic test if test_environment.py isn't available
  python -c "
import sys
print('Python version:', sys.version)
print('Testing critical imports:')
modules = ['numpy', 'pandas', 'geopandas', 'sklearn', 'spacy', 'bertopic', 'nltk', 'sentence_transformers', 'kneed']
import nltk
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

# Test NLTK tokenization
try:
    tokens = nltk.word_tokenize('This large glacial erratic was transported here during the last ice age.')
    print(f'✓ NLTK tokenization works: {tokens[:3]}...')
except Exception as e:
    print(f'✗ NLTK tokenization failed: {e}')
    all_good = False

print('\\nBasic environment check ' + ('passed' if all_good else 'failed'))
"
fi

echo -e "\nFor future pip installs, if needed, consider using:"
echo "pip install --no-deps <package>"
echo -e "\nTo fix issues in this environment, run:"
echo "./create_conda_env_strict.sh -f" 