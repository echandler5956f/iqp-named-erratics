# Glacial Erratics Spatial Analysis Tools

This directory contains the Python-based spatial analysis and machine learning components for the North American Named Glacial Erratics project. These tools provide advanced geospatial analysis, classification, and clustering capabilities.

## Environment Setup

The spatial analysis tools require Python 3.10+ and numerous specialized libraries for geospatial analysis, machine learning, and natural language processing. We use a conda environment to manage these dependencies.

### Setting Up the Environment

1. **Install Conda**:
   If you don't have conda installed, download and install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Environment**:
   Run the provided script to create the conda environment:

   ```bash
   chmod +x create_conda_env.sh
   ./create_conda_env.sh
   ```

   This will create a new conda environment named `iqp-py310` and install all required packages.

3. **Activate the Environment**:
   ```bash
   conda activate iqp-py310
   ```

4. **Verify the Setup**:
   Run the environment test to verify that all dependencies are installed correctly:

   ```bash
   python test_environment.py
   ```

### Fixing Environment Issues

If you encounter compatibility errors, especially between numpy and scipy, use the provided fix script:

```bash
chmod +x fix_env.sh
./fix_env.sh
```

This script reinstalls compatible versions of critical packages and resolves common issues.

Common issues and their fixes:

1. **numpy/scipy compatibility error** (`ValueError: All ufuncs must have type numpy.ufunc`):
   ```bash
   conda install -y numpy=1.24.3 scipy=1.11.4 scikit-learn=1.2.2
   ```

2. **Missing spaCy models**:
   ```bash
   python -m spacy download en_core_web_md
   python -m spacy download en_core_web_lg
   ```

## Using the Analysis Tools

The `run_analysis.py` script provides a unified interface for running the various analysis tools. Here are some common use cases:

### Proximity Analysis

Calculate distances from an erratic to various geographic features:

```bash
python run_analysis.py proximity 123 --update-db
```

Where `123` is the ID of the erratic to analyze.

### Classification

Classify an erratic based on its description and attributes:

```bash
python run_analysis.py classify 123 --update-db
```

### Clustering Analysis

Perform spatial clustering of all erratics:

```bash
python run_analysis.py cluster --output cluster_results.json
```

### Testing the Environment

Run a comprehensive test of the Python environment:

```bash
python run_analysis.py test-env
```

## Individual Scripts

If needed, you can also run the individual scripts directly:

- `proximity_analysis.py`: Calculate distances to geographic features
- `classify_erratic.py`: Classify erratics using NLP and ML
- `setup_env.py`: Check and install required packages
- `test_environment.py`: Test that all dependencies are working
- `fix_env.sh`: Fix common environment issues

## Integration with Node.js Backend

These Python scripts are called from the Node.js backend using the `pythonService.js` service, which handles spawning Python processes with the correct environment.

## Data Storage

The scripts use the following data directories:

- `data/`: Main data directory
- `data/gis/`: GIS datasets
- `data/cache/`: Cached analysis results

## Common Issues

1. **Missing conda environment**: Make sure to run `./create_conda_env.sh` first
2. **Package compatibility errors**: Run `./fix_env.sh` to install compatible versions
3. **Missing spaCy models**: Run `python -m spacy download en_core_web_lg`
4. **Database connection issues**: Ensure the database credentials are set in environment variables

## Further Documentation

- See `SpatialAnalysisMetaPrompt.md` for detailed information about the analysis approach
- Each script includes comprehensive documentation and help text (use `--help` flag) 