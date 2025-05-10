# Glacial Erratics Spatial Analysis Tools

This directory contains the Python-based spatial analysis and machine learning components for the North American Named Glacial Erratics project. These scripts provide advanced geospatial analysis, classification, and clustering capabilities and are primarily designed to be invoked by the Node.js backend.

## Environment Setup

The spatial analysis tools require Python 3.10+ and a number of specialized libraries. Dependencies are managed via `requirements.txt`.

1.  **Create a Python Environment**: It's highly recommended to use a virtual environment (e.g., venv, conda).
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary packages, including `geopandas`, `scikit-learn`, `sentence-transformers`, `psycopg2-binary`, and `pgvector` (for PostgreSQL vector type support).

3.  **Download NLP Models**:
    ```bash
    python -m spacy download en_core_web_md
    # python -m spacy download en_core_web_lg # Optional, larger model
    # NLTK data (if not already present from previous project setup)
    # import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')
    ```
4.  **GDAL/OGR**: Ensure `ogr2ogr` (part of GDAL) is installed and in your system PATH for processing OpenStreetMap PBF data. See main project `README.md` for installation guidance.

## Script Usage (from Node.js via `pythonService.js`)

The Node.js backend service (`pythonService.js`) invokes these scripts directly with specific arguments.

### `proximity_analysis.py`
Calculates proximity to various features and other contextual metrics.
-   **Node.js Call Signature (via `pythonService.runProximityAnalysis`)**:
    -   `erraticId` (integer, required): The ID of the erratic.
    -   `featureLayers` (array of strings, optional): Specific feature layers to analyze.
    -   `updateDb` (boolean, optional): If true, script attempts to update `ErraticAnalyses` table via `data_loader.py`.
-   **Direct Command-Line Example**:
    ```bash
    python proximity_analysis.py <erratic_id> [--features layer1 layer2] [--update-db] [--output results.json]
    ```

### `classify_erratic.py`
Handles NLP tasks: text preprocessing, generating sentence embeddings, training/loading topic models, and classifying erratics.

1.  **Classification Mode**:
    -   **Node.js Call Signature (via `pythonService.runClassification`)**:
        -   `erraticId` (integer, required): The ID of the erratic to classify.
        -   `updateDb` (boolean, optional): If true, script attempts to update `ErraticAnalyses` table.
    -   **Direct Command-Line Example**:
        ```bash
        python classify_erratic.py <erratic_id> [--update-db] [--output classification.json]
        ```

2.  **Topic Model Building Mode**:
    -   **Node.js Call Signature (via `pythonService.runBuildTopicModels`)**:
        -   `outputPath` (string, optional): Path to save the output/log of the build process (e.g., 'build_topics_result.json').
    -   **Direct Command-Line Example (as used by `pythonService.js`)**:
        ```bash
        # <placeholder_id> is required by script structure but not used for building.
        python classify_erratic.py 1 --build-topics --output <outputPath>
        ```

### `clustering.py`
Performs spatial clustering on erratics.
-   **Node.js Call Signature (via `pythonService.runClusteringAnalysis`)**:
    -   `algorithm` (string, required): e.g., 'dbscan', 'kmeans', 'hierarchical'.
    -   `featuresToCluster` (array of strings, optional): Features to use (e.g., ['latitude', 'longitude']).
    -   `algoParams` (object, optional): Algorithm-specific parameters as a JSON string (e.g., `JSON.stringify({ eps: 0.5 })`).
    -   `outputToFile` (boolean, optional): Passed as `--output <filename>` if true.
    -   `outputFilename` (string, optional): Filename for output if `outputToFile` is true.
-   **Direct Command-Line Example**:
    ```bash
    python clustering.py --algorithm dbscan [--features lat lon] [--algo_params '{"eps":0.5}'] [--output results.json]
    ```

### Utility Scripts
-   `utils/data_loader.py`: Manages database connections (using `psycopg2` and `pgvector` for vector types), data fetching from PostgreSQL, downloading/caching external datasets, and updating the `ErraticAnalyses` table.
-   `utils/geo_utils.py`: Provides core geospatial functions.

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