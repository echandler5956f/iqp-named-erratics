# Glacial Erratics Spatial Analysis Tools

This directory contains the Python-based spatial analysis and machine learning components for the North American Named Glacial Erratics project. These scripts provide advanced geospatial analysis, classification, and clustering capabilities and are primarily designed to be invoked by the Node.js backend.

## Environment Setup

The spatial analysis tools require a specific Python 3.10+ environment managed by Conda. All dependencies, including core libraries, NLP models (spaCy), and NLTK data, are installed and configured by a dedicated shell script.

**Prerequisites:**
-   **Conda/Miniconda:** You must have Conda installed. If not, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
-   **GDAL/OGR (External Tool):** For processing OpenStreetMap PBF data, the `ogr2ogr` command-line tool (part of GDAL) must be installed and available in your system PATH. Installation methods vary by OS (e.g., `sudo apt install gdal-bin` on Debian/Ubuntu, `brew install gdal` on macOS, or `conda install -c conda-forge gdal` *before* running the environment script if you prefer to manage it with Conda globally or in the base environment).

**Setup Steps:**

1.  **Navigate to this directory** (`backend/src/scripts/python/`).
2.  **Run the Conda Environment Creation Script:**
    ```bash
    ./create_conda_env_strict.sh
    ```
    This script will:
    *   Create a Conda environment named `iqp-py310` (or attempt to fix it if it exists and you use the `-f` flag).
    *   Install all required Python packages using `conda` (primarily from `conda-forge`) and `pip` for specific packages like `bertopic`, `python-dotenv`, and `pgvector`.
    *   Download necessary spaCy models (`en_core_web_md`, `en_core_web_lg`).
    *   Download required NLTK data (`punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`) into a local directory (`./nltk_data_local/`) within this Python scripts directory to avoid user-level or system-wide NLTK data installations.
    *   Run a basic verification test at the end.

3.  **Activate the Conda Environment:**
    ```bash
    conda activate iqp-py310
    ```
    You must activate this environment in any terminal session where you intend to run these Python scripts directly or when running the Node.js backend that calls these scripts.

**Note on `requirements.txt`:**
While a `requirements.txt` file exists in this directory, it primarily serves as a reference or for specific `pip` installations handled *within* the `create_conda_env_strict.sh` script. It now also includes `pytest` for the testing framework. It's recommended to use the shell script for full environment setup.

## Testing
A test suite using the `pytest` framework is located in the `tests/` subdirectory. To run the tests, activate the Conda environment and run `pytest` from this `python/` directory:

```bash
conda activate iqp-py310
pytest
```

## Script Usage (from Node.js via `pythonService.js`)

The Node.js backend service (`pythonService.js`) invokes these scripts directly with specific arguments. Ensure the `iqp-py310` Conda environment is activated when running the Node.js backend.

### `proximity_analysis.py`
Calculates proximity to various features (water bodies, OSM settlements, NATD roads, Native territories, forest trails) and other contextual metrics using GMTED DEM data (via `geo_utils.py` and `data_pipeline`).
-   **Direct Command-Line Example (ensure `iqp-py310` is active):**
    ```bash
    python proximity_analysis.py <erratic_id> [--update-db] [--output results.json]
    ```

### `classify_erratic.py`
Handles NLP tasks: text preprocessing, generating sentence embeddings, training/loading topic models, and classifying erratics.

1.  **Classification Mode (ensure `iqp-py310` is active):**
    ```bash
    python classify_erratic.py <erratic_id> [--update-db] [--output classification.json]
    ```

2.  **Topic Model Building Mode (ensure `iqp-py310` is active):**
    ```bash
    # <placeholder_id> is required by script structure but not used for building.
    python classify_erratic.py 1 --build-topics --output <outputPath>
    ```

### `clustering.py`
Performs spatial clustering on erratics.
-   **Direct Command-Line Example (ensure `iqp-py310` is active):**
    ```bash
    python clustering.py --algorithm dbscan [--features lat lon] [--algo_params '{"eps":0.5}'] [--output results.json]
    ```

### Utility and Data Pipeline Modules
-   `data_pipeline/` (directory): This module is a complete replacement for the old `utils/data_loader.py`. It provides a modular system for defining data sources (`data_sources.py`), loading data from various origins (`loaders.py`), processing different GIS formats (`processors.py`), managing a cache (`cache.py`), and orchestrating the overall data acquisition and preparation (`pipeline.py`). See `data_pipeline/README.md` for more details.
-   `utils/geo_utils.py`: Provides core geospatial functions. DEM data loading is now specifically handled for the tiled GMTED `DataSource` defined in the `data_pipeline`.
-   `utils/db_utils.py`: Manages database connections (using `psycopg2` and `pgvector` for vector types) and all database read/write operations for the Python scripts. Loads database credentials from the project root `.env` file.
-   `utils/file_utils.py`: Basic file utilities (e.g., for JSON handling).

## Data Storage

The scripts use the following data directories relative to `backend/src/scripts/python/`:

- `data/`: Main data directory.
    - `data/gis/`: Base directory for GIS datasets. The `data_pipeline` manages downloads and organization within subdirectories here based on `DataSource` configurations (e.g., `data/gis/hydro/`, `data/gis/osm/`, `data/gis/elevation/`). Manually acquired datasets (like specific GMTED tiles, NATD roads, Forest Trails, Native Land GeoJSONs) must be placed in the correct subdirectories as expected by `data_pipeline/data_sources.py`.
    - `data/cache/`: Cached intermediate results (e.g., processed GeoDataFrames), managed by `data_pipeline.cache.CacheManager`.
    - `data/models/`: Saved machine learning models (e.g., for `classify_erratic.py`).
- `nltk_data_local/`: Locally downloaded NLTK data packages (this directory is in `.gitignore`).

## Common Issues

1.  **Conda Not Found:** Ensure Conda is installed and its `bin` directory is in your system `PATH`.
2.  **Environment Activation:** Always activate the `iqp-py310` environment (`conda activate iqp-py310`) before running scripts or the Node.js backend.
3.  **Script Execution Errors:** If `create_conda_env_strict.sh` completed without errors but scripts fail, ensure the environment is active. Check script-specific logs for more details. Consider re-running `./create_conda_env_strict.sh -f` to attempt to fix the environment.
4.  **Database Connection Issues:** Ensure PostgreSQL is running and accessible. Database credentials are loaded from the project root `.env` file by `utils/db_utils.py`. Verify this `.env` file is correctly populated.
5.  **`ogr2ogr` Not Found:** If OSM PBF processing fails (when using `data_pipeline` for OSM sources), ensure GDAL/OGR is installed and `ogr2ogr` is in your PATH.
6.  **Missing Local GIS Data:** Many `DataSource` objects defined in `data_pipeline/data_sources.py` are of type `file` and expect data to be manually placed in `data/gis/`. If scripts fail to load these sources, verify the files exist in the correct subpaths.

## Further Documentation

- See `SpatialAnalysisMetaPrompt.md` for detailed information about the analysis approach.
- See `data_pipeline/README.md` for specifics on the data handling system.
- Each script includes comprehensive documentation and help text (use `--help` flag when running from the command line). 