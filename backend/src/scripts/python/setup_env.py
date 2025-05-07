#!/usr/bin/env python3
"""
Environment setup script for the Python spatial analysis components.
This script verifies the required packages are installed and sets up the environment.
It's designed to work with the 'iqp-py310' conda environment.
"""

import sys
import subprocess
import os
import pkg_resources
import platform

# Core packages with specific versions for compatibility
CORE_PACKAGES = [
    'numpy==1.24.3',
    'scipy==1.11.4',
    'scikit-learn==1.2.2'
]

# All required packages for the spatial analysis and ML components
REQUIRED_PACKAGES = [
    # Core ML/NLP packages (versions are specified in CORE_PACKAGES)
    'scikit-learn>=1.2.0',
    'sentence-transformers>=2.2.0',
    'joblib>=1.1.0',
    'bertopic>=0.15.0',
    'hdbscan>=0.8.29',
    'umap-learn>=0.5.3',
    'nltk>=3.7.0',
    'spacy>=3.4.0',
    
    # Data processing and analysis
    'numpy>=1.24.0',
    'pandas>=1.4.0',
    'kneed>=0.8.0',  # For finding knee points in curves

    'plotly>=5.7.0', # Needed for bertopic
    
    # Database connectivity
    'psycopg2-binary>=2.9.3',
    'SQLAlchemy>=1.4.0',
    
    # Geographic and spatial processing
    'geopandas>=0.10.0',
    'shapely>=1.8.0',
    'pyproj>=3.3.0',
    'rtree>=1.0.0',
]

# Required spaCy models
SPACY_MODELS = ['en_core_web_md', 'en_core_web_lg']

CONDA_ENV_NAME = 'iqp-py310'

def check_conda_env():
    """Check if running in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != CONDA_ENV_NAME:
        print(f"Warning: This script should be run in the '{CONDA_ENV_NAME}' conda environment.")
        print(f"Current environment: {conda_env or 'Not in a conda environment'}")
        print(f"Please activate the correct environment with: conda activate {CONDA_ENV_NAME}")
        return False
    return True

def check_conda_executable():
    """Check if conda is available and return the path to the executable."""
    try:
        if platform.system() == 'Windows':
            # On Windows, try to find conda in standard locations
            paths_to_check = [
                os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3', 'Scripts', 'conda.exe'),
                os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'Scripts', 'conda.exe'),
                os.path.join(os.environ.get('PROGRAMFILES', ''), 'miniconda3', 'Scripts', 'conda.exe'),
                os.path.join(os.environ.get('PROGRAMFILES', ''), 'anaconda3', 'Scripts', 'conda.exe')
            ]
            for path in paths_to_check:
                if os.path.exists(path):
                    return path
            
            # Try to get from where clause
            try:
                result = subprocess.run(['where', 'conda'], capture_output=True, text=True, check=True)
                return result.stdout.strip().split('\n')[0]
            except:
                pass
        else:
            # On Unix, use which
            try:
                result = subprocess.run(['which', 'conda'], capture_output=True, text=True, check=True)
                return result.stdout.strip()
            except:
                pass
            
            # Try standard locations
            paths_to_check = [
                os.path.join(os.environ.get('HOME', ''), 'miniconda3', 'bin', 'conda'),
                os.path.join(os.environ.get('HOME', ''), 'anaconda3', 'bin', 'conda'),
                '/opt/conda/bin/conda',
                '/usr/local/bin/conda'
            ]
            for path in paths_to_check:
                if os.path.exists(path):
                    return path
    except Exception as e:
        print(f"Error checking for conda: {e}")
    
    return None

def check_packages():
    """Check if required packages are installed."""
    missing = []
    for package_req in REQUIRED_PACKAGES:
        # Extract package name from requirement string (remove version)
        package = package_req.split('>=')[0].split('==')[0].strip()
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    return missing

def check_package_versions():
    """Check if installed packages have compatible versions."""
    incompatible = []
    
    try:
        # Check numpy/scipy compatibility
        numpy_ver = pkg_resources.get_distribution('numpy').version
        scipy_ver = pkg_resources.get_distribution('scipy').version
        
        print(f"Installed numpy version: {numpy_ver}")
        print(f"Installed scipy version: {scipy_ver}")
        
        # Check scikit-learn
        if pkg_resources.get_distribution('scikit-learn').version.startswith('1.3'):
            incompatible.append('scikit-learn')
        
        # Add specific version checks here based on known compatibility issues
        if numpy_ver.startswith('1.26') and scipy_ver.startswith('1.11'):
            incompatible.append('numpy')
            incompatible.append('scipy')
            
    except Exception as e:
        print(f"Warning: Could not check package versions: {e}")
    
    return incompatible

def check_spacy_models():
    """Check if required spaCy models are installed."""
    missing_models = []
    try:
        import spacy
        for model in SPACY_MODELS:
            try:
                spacy.load(model)
            except OSError:
                missing_models.append(model)
    except ImportError:
        # spaCy itself is missing, which will be handled by check_packages
        pass
    
    return missing_models

def install_core_packages():
    """Install core packages with specific versions for compatibility."""
    print("Installing core packages with specific versions for compatibility...")
    conda_path = check_conda_executable()
    
    if conda_path:
        # First try to install numpy and scipy using conda
        try:
            print("Installing numpy and scipy with conda...")
            subprocess.check_call([conda_path, 'install', '-y', 'numpy=1.24.3', 'scipy=1.11.4', 'scikit-learn=1.2.2'])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install core packages with conda: {e}")
    
    # If conda failed or isn't available, try pip
    try:
        print("Installing core packages with pip...")
        for package in CORE_PACKAGES:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install core packages: {e}")
        return False

def install_packages(packages):
    """Install missing packages using conda or pip as appropriate."""
    conda_path = check_conda_executable()
    
    for package in packages:
        print(f"Installing {package}...")
        # Try conda first, then pip if conda fails
        if conda_path:
            try:
                subprocess.check_call([conda_path, 'install', '-y', package])
                continue  # Skip pip if conda succeeds
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Conda install failed ({e}), trying pip for {package}...")
        
        # Try pip if conda failed or isn't available
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

def install_spacy_models(models):
    """Install required spaCy models."""
    for model in models:
        print(f"Downloading spaCy model: {model}...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        except subprocess.CalledProcessError as e:
            print(f"Failed to download spaCy model {model}: {e}")

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        print("Downloading NLTK data...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def fix_environment_issues():
    """Fix common environment issues."""
    issues_fixed = False
    
    # Check for incompatible versions
    incompatible = check_package_versions()
    if incompatible:
        print(f"Found incompatible package versions: {', '.join(incompatible)}")
        if install_core_packages():
            print("Successfully fixed package version compatibility issues")
            issues_fixed = True
        else:
            print("Failed to fix package version compatibility issues")
    
    return issues_fixed

def verify_imports():
    """Verify that all critical packages can be imported."""
    print("Testing imports of critical packages...")
    
    critical_packages = ['numpy', 'scipy', 'sklearn', 'spacy', 'pandas', 'geopandas']
    all_ok = True
    
    for package in critical_packages:
        try:
            if package == 'sklearn':
                # sklearn is the import name for scikit-learn
                __import__('sklearn')
                print(f"✓ Successfully imported {package}")
            else:
                __import__(package)
                print(f"✓ Successfully imported {package}")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Main function to set up the environment."""
    print("Checking Python environment for spatial analysis components...")
    
    # Check if in the right conda environment
    if not check_conda_env():
        print("Please activate the correct conda environment and run this script again.")
        return
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print(f"Warning: Python 3.10+ is recommended. You are using Python {python_version.major}.{python_version.minor}")
    
    # Fix any environment issues first
    fixed_issues = fix_environment_issues()
    
    # Check packages
    missing_packages = check_packages()
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        install_packages(missing_packages)
    else:
        print("All required Python packages are installed.")
    
    # Check spaCy models
    missing_models = check_spacy_models()
    if missing_models:
        print(f"Missing spaCy models: {', '.join(missing_models)}")
        install_spacy_models(missing_models)
    else:
        print("All required spaCy models are installed.")
    
    # Download NLTK data
    download_nltk_data()
    
    # Verify imports
    if verify_imports():
        print("All critical packages can be imported.")
    else:
        print("Warning: Some critical packages could not be imported.")
        print("You may need to run 'conda install -y numpy=1.24.3 scipy=1.11.4 scikit-learn=1.2.2'")
    
    print("Environment setup complete.")

if __name__ == "__main__":
    main() 