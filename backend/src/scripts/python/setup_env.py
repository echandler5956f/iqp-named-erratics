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

REQUIRED_PACKAGES = [
    'scikit-learn',
    'geopandas',
    'spacy',
    'pandas',
    'numpy',
    'shapely',
    'rtree',
    'matplotlib',
    'tensorflow'
]

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

def check_packages():
    """Check if required packages are installed."""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    return missing

def install_packages(packages):
    """Install missing packages using conda or pip as appropriate."""
    for package in packages:
        print(f"Installing {package}...")
        # Try conda first, then pip if conda fails
        try:
            subprocess.check_call(['conda', 'install', '-y', package])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Conda install failed, trying pip for {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Download spaCy model
    if 'spacy' in packages:
        print("Downloading spaCy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])

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
    
    # Check packages
    missing_packages = check_packages()
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        install_packages(missing_packages)
    else:
        print("All required packages are installed.")
    
    print("Environment setup complete.")

if __name__ == "__main__":
    main() 