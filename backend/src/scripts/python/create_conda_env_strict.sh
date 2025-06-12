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

ENV_NAME="glacial-erratics"
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
conda install -y -c conda-forge "joblib>=1.1.1" numpy=1.24.3 scipy=1.11.4 scikit-learn=1.2.2 pandas

echo "Installing geospatial packages with conda-forge..."
conda install -y -c conda-forge geopandas shapely pyproj rtree "pyarrow>=8.0.0" "requests>=2.27.0" rasterio matplotlib

echo "Installing NLP/ML stack with conda-forge..."
conda install -y -c conda-forge nltk=3.7.0 kneed=0.8.5 hdbscan=0.8.29 umap-learn=0.5.3 spacy=3.4.0 pyyaml

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

echo "Installing specific packages via pip (bertopic, python-dotenv, pgvector, pytest)..."
pip install "sentence-transformers==2.2.2"
pip install "transformers>=4.17.0,<4.18.0" "huggingface-hub>=0.11.0,<0.12.0"
pip install "bertopic[visualization]==0.15.0"
pip install "python-dotenv>=0.20.0"
pip install "pgvector>=0.2.0"
pip install "pytest>=7.0.0"
pip install "portalocker==3.1.1"
pip install "pyyaml==6.0.2"

python -c "import bertopic; print(f'BERTopic imported successfully (version {bertopic.__version__})')" || echo "BERTopic import failed!"
python -c "import dotenv; print(f'python-dotenv imported successfully')" || echo "python-dotenv import failed!"
python -c "import pgvector.psycopg2; print(f'pgvector.psycopg2 imported successfully')" || echo "pgvector.psycopg2 import failed!"

echo "Installing spaCy models..."
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg

echo "Installing essential NLTK data to $NLTK_DATA_DIR ..."
mkdir -p "$NLTK_DATA_DIR"
python -c "
import nltk
import ssl
import os
import time
import shutil

custom_nltk_path = '$NLTK_DATA_DIR'
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)

print(f\"NLTK data path configured to: {nltk.data.path}\")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

essential_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for pkg_name in essential_packages:
    download_successful_for_pkg = False
    resource_paths_to_check = []
    if pkg_name == 'punkt':
        resource_paths_to_check.append('tokenizers/punkt')
    elif pkg_name == 'stopwords':
        resource_paths_to_check.append('corpora/stopwords')
    elif pkg_name == 'wordnet':
        wordnet_zip_path = os.path.join(custom_nltk_path, 'corpora', 'wordnet.zip')
        wordnet_dir_path = os.path.join(custom_nltk_path, 'corpora', 'wordnet')

        if os.path.exists(wordnet_zip_path):
            print(f\"Removing existing {wordnet_zip_path}\")
            os.remove(wordnet_zip_path)
        if os.path.exists(wordnet_dir_path):
            print(f\"Removing existing directory {wordnet_dir_path}\")
            shutil.rmtree(wordnet_dir_path)
        
        os.makedirs(os.path.join(custom_nltk_path, 'corpora'), exist_ok=True)

        download_attempts = 1 
        wordnet_download_succeeded_according_to_nltk = False
        for attempt in range(download_attempts):
            try:
                print(f\"Attempting to download NLTK package: {pkg_name} to {custom_nltk_path}\")
                nltk.download(pkg_name, download_dir=custom_nltk_path, quiet=False, force=True, raise_on_error=True)
                wordnet_download_succeeded_according_to_nltk = True
                break
            except Exception as e_download:
                print(f\"Error during nltk.download for {pkg_name}: {e_download}\")

        if wordnet_download_succeeded_according_to_nltk or os.path.exists(wordnet_zip_path):
            if os.path.exists(wordnet_zip_path):
                print(f\"Found {wordnet_zip_path}. Attempting manual unzip.\")
                try:
                    import zipfile
                    with zipfile.ZipFile(wordnet_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(custom_nltk_path, 'corpora'))
                    print(f\"Manually unzipped {wordnet_zip_path} to {os.path.join(custom_nltk_path, 'corpora')}\")
                    if os.path.exists(wordnet_dir_path):
                        print(f\"Directory {wordnet_dir_path} exists after manual unzip.\")
                        download_successful_for_pkg = True 
                    else:
                        print(f\"Directory {wordnet_dir_path} DOES NOT exist after manual unzip.\")
                except Exception as e_unzip:
                    print(f\"Error manually unzipping {wordnet_zip_path}: {e_unzip}\")
            else:
                print(f\"{wordnet_zip_path} not found even after nltk.download claimed success or was up-to-date.\")
        else:
            print(f\"NLTK download for {pkg_name} did not result in a zip file and did not claim success.\")

        if download_successful_for_pkg: 
            try:
                nltk.data.find('corpora/wordnet')
                print(f\"Successfully verified NLTK resource for {pkg_name} (corpora/wordnet) after potential manual unzip.\")
            except LookupError:
                print(f\"Verification for {pkg_name} (corpora/wordnet) FAILED even after potential manual unzip.\")
                download_successful_for_pkg = False
        else:
            print(f\"Skipping final verification for {pkg_name} as download/unzip stage indicated failure.\")
            
    elif pkg_name == 'averaged_perceptron_tagger':
        resource_paths_to_check.append('taggers/averaged_perceptron_tagger')
    else:
        resource_paths_to_check.append(f\"corpora/{pkg_name}\")
        resource_paths_to_check.append(f\"tokenizers/{pkg_name}\")
        resource_paths_to_check.append(f\"taggers/{pkg_name}\")

    found_locally = False
    if resource_paths_to_check:
        try:
            nltk.data.find(resource_paths_to_check[0])
            print(f\"NLTK resource for {pkg_name} ({resource_paths_to_check[0]}) already found.\")
            found_locally = True
            download_successful_for_pkg = True
        except LookupError:
            print(f\"NLTK resource for {pkg_name} not found locally via {resource_paths_to_check[0]}. Will attempt download.\")
        except Exception as e_find:
            print(f\"Unexpected error checking for NLTK resource {resource_paths_to_check[0]} for {pkg_name}: {e_find}\")

    if not found_locally:
        download_attempts = 3
        for attempt in range(download_attempts):
            try:
                print(f\"Attempt {attempt + 1} of {download_attempts} to download NLTK package: {pkg_name} to {custom_nltk_path}\")
                nltk.download(pkg_name, download_dir=custom_nltk_path, quiet=False, force=False, raise_on_error=True)
                if resource_paths_to_check:
                    nltk.data.find(resource_paths_to_check[0]) 
                    print(f\"Successfully downloaded and verified NLTK resource for {pkg_name} ({resource_paths_to_check[0]}).\")
                else:
                    print(f\"Successfully downloaded NLTK package {pkg_name} (no specific resource path to verify).\")
                download_successful_for_pkg = True
                break 
            except LookupError as le_verify: 
                print(f\"Verification failed for {pkg_name} after download attempt {attempt + 1}: {le_verify}\")
                if attempt < download_attempts - 1:
                    print(f\"Retrying download for {pkg_name}...\")
                    time.sleep(2)
                else:
                    print(f\"Failed to download/verify {pkg_name} after {download_attempts} attempts due to verification LookupError.\")
            except Exception as e_download: 
                print(f\"An error occurred during NLTK download attempt {attempt + 1} for {pkg_name}: {e_download}\")
                if attempt < download_attempts - 1:
                    print(f\"Retrying download for {pkg_name}...\")
                    time.sleep(2) 
                else:
                    print(f\"Failed to download {pkg_name} after {download_attempts} attempts due to error: {e_download}.\")
        
        if not download_successful_for_pkg:
            print(f\"Warning: NLTK package {pkg_name} could not be successfully downloaded and verified after multiple attempts.\")

print(\"NLTK essential data download process completed.\")

print(\"\\nVerifying WordNet installation specifically (with custom path prioritized)...\")
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)
try:
    nltk.data.find('corpora/wordnet')
    print(\"WordNet is available to NLTK.\")
    wordnet_in_custom_path = False
    for p in nltk.data.path:
        if os.path.exists(os.path.join(p, 'corpora', 'wordnet')) or os.path.exists(os.path.join(p, 'corpora', 'wordnet.zip')):
            if custom_nltk_path in p:
                 wordnet_in_custom_path = True
            print(f\"  Found 'corpora/wordnet' in: {p}\")
    if wordnet_in_custom_path:
        print(f\"  WordNet confirmed in custom download path: {custom_nltk_path}\")
    else:
        print(f\"  Warning: WordNet found, but not in the expected custom path: {custom_nltk_path}\")
except LookupError: 
    print(\"WordNet still not found by NLTK.\")
    print(f\"  Please check the directory structure, e.g., {custom_nltk_path}/corpora/wordnet or {custom_nltk_path}/corpora/wordnet.zip\")
    print(f\"  NLTK search paths were: {nltk.data.path}\")
    print(f'  Consider running: python -m nltk.downloader -d \"{custom_nltk_path}\" wordnet')
"

echo "Verifying NLTK tokenization works with custom path..."
python -c "
import nltk
custom_nltk_path = '$NLTK_DATA_DIR'
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)
try:
    tokens = nltk.word_tokenize(\"This is a test sentence.\")
    print(f\"NLTK tokenization successful: {tokens}\")
except Exception as e:
    print(f\"NLTK tokenization failed: {e}\")
    print(f\"NLTK data path was: {nltk.data.path}\")
    print(f\"Please check {custom_nltk_path} for NLTK data.\")
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
print(\"Testing critical imports:\")
modules = ['numpy', 'pandas', 'geopandas', 'sklearn', 'spacy', 'bertopic', 'nltk', 'sentence_transformers', 'kneed', 'dotenv', 'pgvector', 'pytest', 'huggingface_hub']
import nltk
custom_nltk_path = '$NLTK_DATA_DIR'
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)
all_good = True
for m in modules:
    try:
        if m == 'huggingface_hub':
            from huggingface_hub import HfApi # Try importing something common
            print(f\"{m} (HfApi) imported successfully\")
        elif m == 'sentence_transformers':
            import sentence_transformers
            print(f\"{m} imported successfully (version {sentence_transformers.__version__})\")
        else:
            __import__(m)
            print(f\"{m} imported successfully\")
    except ImportError as e:
        print(f\"{m} import failed: {e}\")
        all_good = False
    except Exception as e:
        print(f\"Error importing {m}: {e}\")
        all_good = False

try:
    tokens = nltk.word_tokenize('This large glacial erratic was transported here during the last ice age.')
    print(f\"NLTK tokenization works: {tokens[:3]}...\")
except Exception as e:
    print(f\"NLTK tokenization failed: {e}\")
    all_good = False

print('\nBasic environment check ' + ('passed' if all_good else 'failed'))
"
fi

echo -e "\nEnvironment setup complete."
echo -e "\nTo fix issues in this environment, run again with -f flag:"
echo "./create_conda_env_strict.sh -f" 