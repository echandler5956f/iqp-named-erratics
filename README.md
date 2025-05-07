# Glacial Erratics Map

A full-stack web application that displays a dynamic, interactive map of named glacial erratics in North America, with detailed information and integrated spatial analysis capabilities.

## Project Overview

This application provides an interactive map interface to explore named glacial erratics (large rocks transported and deposited by glaciers) found across North America (USA and Canada). It focuses on presenting detailed geological, historical, and cultural information for each erratic, augmented by **sophisticated backend spatial analysis and NLP/ML classification**. Features include:

- Multiple toggleable map layers (OpenStreetMap, Satellite, Terrain)
- Interactive markers for named glacial erratics
- Detailed pop-up and sidebar information for each erratic (location, size, rock type, description, cultural significance, historical notes, etc.)
- **Comprehensive Backend Analysis:**
  * Integration with diverse GIS datasets (HydroSHEDS, Native Land Digital, OSM, historical data) providing rich North American context.
  * Calculation of proximity to various features (water bodies, modern/colonial settlements & roads, Native territories).
  * Contextual characterization based on elevation, accessibility, size, estimated displacement, geomorphology, and terrain metrics (ruggedness, slope position).
  * NLP-based classification using descriptions and notes to uncover thematic categories (via Sentence Transformers and Topic Modeling like BERTopic/LDA), including identifying potential inscriptions.
  * Spatial relationship analysis between erratics using various clustering algorithms (DBSCAN, KMeans, Hierarchical) and nearest neighbor distances.
- User location tracking for finding nearby erratics.
- Admin interface for managing erratic data.
- Responsive design for desktop and mobile viewing.

## Technical Stack

### Frontend
- React (via Vite)
- Leaflet.js (via React-Leaflet) for interactive mapping
- React Router for navigation
- Axios for API requests
- CSS for styling

### Backend
- Node.js with Express.js
- PostgreSQL with PostGIS extension for spatial data storage and querying
- Sequelize ORM for database interactions
- Optional: pgvector extension for efficient similarity search on vector embeddings
- JWT for admin authentication
- RESTful API architecture
- Integration with Python scripts (`child_process`) for specific analysis tasks

### Analysis (Backend/Python Scripts)
- **Core Libraries:** Python 3.10+, `geopandas`, `shapely`, `pandas`, `numpy`, `psycopg2`, `rasterio` (for DEM processing), `joblib` (for model saving).
- **NLP/ML:** `spacy` (en_core_web_md), `nltk`, `sentence-transformers` (all-MiniLM-L6-v2), `scikit-learn` (for LDA, DBSCAN, K-Means, Hierarchical Clustering, utilities), `kneed` (for DBSCAN eps estimation), potentially `bertopic`, `hdbscan`, `umap` if installed (for advanced topic modeling).
- **External Tools:** Requires `ogr2ogr` (part of GDAL) to be installed and available in the system PATH for processing OpenStreetMap PBF data.
- **Data Handling:** Loads data from PostgreSQL/PostGIS and external authoritative GIS sources (HydroSHEDS, Native Land Digital, OSM extracts via Geofabrik, NHGIS, DAART). Downloads and caches external datasets, including tiled SRTM 90m DEM data via FTP.
- **Scripts:**
    - `proximity_analysis.py`: Calculates proximity to various features (water, roads, settlements, native lands, colonial sites) and basic context features (elevation category, accessibility, estimated displacement). Integrates terrain analysis results from `geo_utils.py`.
    - `clustering.py`: Performs spatial clustering on erratics using DBSCAN, K-Means, or Hierarchical methods based on specified features (coordinates, attributes).
    - `classify_erratic.py`: Handles NLP tasks: text preprocessing, generating sentence embeddings, training/loading topic models (BERTopic or LDA), classifying erratic usage/significance, and detecting potential inscriptions. Includes separate modes for building/saving models and for classifying individual erratics.
    - `utils/data_loader.py`: Manages database connections, data fetching from PostgreSQL, downloading/caching external datasets (including OSM PBF processing via `ogr2ogr` and SRTM 90m FTP downloads), and updating the `ErraticAnalyses` table.
    - `utils/geo_utils.py`: Provides core geospatial functions like distance calculations (Haversine), nearest neighbor search (in GeoDataFrames or via PostGIS), DEM data loading (`rasterio`), elevation/slope/aspect/ruggedness calculations from DEM.
- **Integration:** Python scripts in `backend/src/scripts/python/` executed by the Node.js backend (`backend/src/services/pythonService.js`) via `child_process`, exchanging data via command-line arguments and JSON output.

## Setup & Installation

### Prerequisites
- Node.js v20.19.0 and npm v10.8.2 (or compatible versions)
- PostgreSQL 14+ with PostGIS extension enabled (`CREATE EXTENSION postgis;`)
- **Python 3.10+**
- **GDAL (with `ogr2ogr` command-line tool):** Needs to be installed on the system and accessible in the PATH. Installation varies by OS (e.g., `sudo apt install gdal-bin` on Ubuntu/Debian, OSGeo4W on Windows, Homebrew on macOS).
- **Python Packages:** Install via pip:
  ```bash
  pip install -r backend/src/scripts/python/requirements.txt
  ```
  This includes `geopandas`, `rasterio`, `scikit-learn`, `sentence-transformers`, `bertopic`, etc. See the file for the full list.
  - *Note:* Installing `geopandas` and `rasterio` can sometimes be complex due to underlying C library dependencies (like GDAL itself). Using Conda (`conda install -c conda-forge geopandas rasterio`) is often recommended.
  - Download spaCy models: `python -m spacy download en_core_web_md`
  - Download NLTK data: Run python and execute `import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')`

### Database Setup
1. Create a PostgreSQL database (e.g., `glacial_erratics`).
2. **Enable Extensions:** Connect to the database (e.g., using `psql`) and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   -- Optional but recommended for vector similarity search:
   CREATE EXTENSION IF NOT EXISTS vector; 
   ```
3. Configure connection details in the `.env` file in the `backend/` directory (copy from `.env.example`).

### Installation Steps
1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project root: `cd iqp-named-erratics`
3. Install Node.js dependencies for both frontend and backend:
   ```bash
   npm run install:all
   ```
4. Navigate to the backend directory:
   ```bash
   cd backend
   ```
5. Run database migrations to create tables and add columns:
   ```bash
   npm run db:migrate 
   ``` 
   *(This is crucial for creating the `Erratics` and `ErraticAnalyses` tables with the correct schema)*
6. Seed the database with initial data (if seeders are available - check `src/seeders/`):
   ```bash
   # Example command, adjust if necessary
   # npx sequelize-cli db:seed:all
   ```
   (Alternatively, use `npm run db:init` if it handles seeding).
7. **Optional:** Download large datasets manually if needed (e.g., OSM PBF):
    - Download `north-america-latest.osm.pbf` (or relevant regional extract) from a source like Geofabrik.
    - Place it in `backend/src/scripts/python/data/gis/settlements/osm_north_america/`.
8. **Build Topic Model (One-time or when data updates):** Run the classification script with the `--build-topics` flag to train and save the NLP models. This can take time.
   ```bash
   # From the project root directory:
   python backend/src/scripts/python/classify_erratic.py 1 --build-topics --output temp_build_log.json
   ```
   *(Note: The erratic ID '1' is just a placeholder and not used for building the model, but required by the current script argument structure. The output file is temporary)*
9. Start the development servers (from the project root):
   ```bash
   npm run dev
   ```
   This should start both the backend server and the frontend dev server concurrently.

## Database Structure

The application uses PostgreSQL with PostGIS, managed by Sequelize. The core data is split into two main related tables: `Erratics` for intrinsic properties and `ErraticAnalyses` for computed analytical data, linked by a one-to-one relationship (`ErraticAnalyses.erraticId` references `Erratics.id`).

### Erratic Model (`Erratics` table)

Stores the fundamental, directly observed, or historically recorded information about each glacial erratic.

-   **Core Fields:**
    -   `id`: `INTEGER` (Primary Key, Auto-incrementing)
    -   `name`: `STRING` (Name of the erratic)
    -   `location`: `GEOMETRY('POINT', 4326)` (Geographic coordinates, SRID 4326)
    -   `elevation`: `FLOAT` (Elevation in meters - often from initial data source)
    -   `size_meters`: `FLOAT` (Approximate size in meters)
    -   `rock_type`: `STRING` (Primary rock composition)
    -   `estimated_age`: `STRING` (Geological age estimate)
    -   `discovery_date`: `DATE` (Date of formal discovery or recording)
    -   `description`: `TEXT` (General description)
    -   `cultural_significance`: `TEXT` (Notes on cultural importance)
    -   `historical_notes`: `TEXT` (Historical accounts or events)
    -   `image_url`: `STRING` (URL to a representative image)
-   **Associations:**
    -   Has many `ErraticMedia`
    -   Has many `ErraticReference`
    -   Has one `ErraticAnalysis`

### ErraticAnalysis Model (`ErraticAnalyses` table)

Stores results computed by the backend Python analysis pipeline. Created/updated by the analysis scripts.

-   **Key Fields:**
    -   `erraticId`: `INTEGER` (Primary Key, Foreign Key referencing `Erratics.id`)
    -   `usage_type`: `ARRAY(STRING)` (Derived from NLP topic modeling, e.g., "Geological Feature", "Cultural Landmark")
    -   `cultural_significance_score`: `INTEGER` (Score derived from NLP classification or other metrics)
    -   `has_inscriptions`: `BOOLEAN` (Derived from keyword analysis of text fields)
    -   `accessibility_score`: `INTEGER` (Score [1-5] based on proximity to roads, settlements)
    -   `size_category`: `STRING` (e.g., "Small", "Medium", "Large", "Monumental", based on `size_meters`)
    -   `nearest_water_body_dist`: `FLOAT` (Distance in meters to nearest lake/river)
    -   `nearest_settlement_dist`: `FLOAT` (Distance in meters to nearest modern settlement - OSM based)
    -   `nearest_colonial_settlement_dist`: `FLOAT` (Distance to nearest historical colonial settlement - NHGIS/Fallback)
    -   `nearest_road_dist`: `FLOAT` (Distance to nearest modern road - OSM based)
    -   `nearest_colonial_road_dist`: `FLOAT` (Distance to nearest historical colonial road - DAART/Fallback)
    -   `nearest_native_territory_dist`: `FLOAT` (Distance to nearest boundary of a Native/Indigenous territory - Native Land Digital)
    -   `elevation_category`: `STRING` (e.g., "Lowland", "Upland", based on DEM elevation)
    -   `geological_type`: `STRING` (Potential future field for detailed geological context)
    -   `estimated_displacement_dist`: `FLOAT` (Simplified estimation of glacial transport distance)
    -   `vector_embedding`: `VECTOR(384)` or `JSONB` (Vector embedding from sentence transformer. Uses `VECTOR` if pgvector is enabled, otherwise `JSONB`).
    -   `vector_embedding_data`: `JSONB` (Legacy/Redundant field if `vector_embedding` is used correctly).
    -   `ruggedness_tri`: `FLOAT` (Terrain Ruggedness Index around the erratic, calculated from DEM)
    -   `terrain_landform`: `STRING` (Simplified landform classification based on DEM, e.g., 'flat_plain', 'hilly_or_mountainous')
    -   `terrain_slope_position`: `STRING` (Simplified slope position based on DEM, e.g., 'level', 'mid-slope')
    -   `createdAt`: `TIMESTAMP WITH TIME ZONE`
    -   `updatedAt`: `TIMESTAMP WITH TIME ZONE`
-   **Associations:**
    -   Belongs to `Erratic`

### Other Models

- **ErraticMedia**: Images and other media associated with erratics.
- **ErraticReference**: Scientific references and sources related to erratics.
- **User**: Admin user accounts for managing data.
- **Migration**: Tracks applied database migrations (managed by Sequelize).

## API Endpoints

### Public Erratic Endpoints
- `GET /api/erratics`: Get all erratics (potentially with joined analysis data).
- `GET /api/erratics/:id`: Get a specific erratic by ID (potentially with joined analysis data).
- `GET /api/erratics/nearby`: Get erratics near a specific location (lat, lng, radius).

### Public Analysis Endpoints (Trigger Python Scripts)
- `GET /api/analysis/proximity/:id`: Calculates/retrieves proximity and context metrics for a specific erratic. **(Implementation Status: Backend script `proximity_analysis.py` exists and performs calculations, including terrain. Needs integration with Node.js endpoint and data persistence via `update_erratic_analysis_data`).**
- `GET /api/analysis/classify/:id`: Classifies an erratic using pre-trained NLP models. **(Implementation Status: Backend script `classify_erratic.py` exists with separate train/classify modes. Needs integration with Node.js endpoint and data persistence via `update_erratic_analysis_data`).**
- `GET /api/analysis/cluster`: Performs spatial clustering on all erratics. **(Implementation Status: Backend script `clustering.py` exists. Needs Node.js endpoint to trigger it and decide how results are stored/returned).**

### Protected Admin Endpoints
- `POST /api/erratics`: Create a new erratic.
- `PUT /api/erratics/:id`: Update an existing erratic.
- `DELETE /api/erratics/:id`: Delete an erratic.
- `POST /api/auth/login`: Admin login.
- `GET /api/auth/profile`: Get admin profile information.
- `POST /api/analysis/proximity/batch`: Trigger proximity analysis for multiple erratic IDs. **(Implementation Status: Requires Node.js endpoint implementation to call `proximity_analysis.py` repeatedly or in batch mode).**
- `POST /api/analysis/classify/batch`: Trigger NLP classification for multiple erratic IDs. **(Implementation Status: Requires Node.js endpoint implementation to call `classify_erratic.py`).**
- **Trigger Topic Model Build:** No dedicated endpoint. Requires running `python backend/src/scripts/python/classify_erratic.py --build-topics` manually or via a custom admin action.

## Frontend Features

### Interactive Map

The application features an interactive map (`ErraticsMap.jsx`) built with React-Leaflet, displaying erratics and allowing user interaction.

### Filtering System

The frontend (`HomePage.jsx`) implements a dynamic filtering system allowing users to customize which erratics are displayed on the map.

-   **Architecture:**
    -   Filter definitions (`GLOBAL_FILTER_DEFINITIONS`) are managed in `HomePage.jsx`. These definitions specify the filter's label, default configuration, and the React component used to render its specific UI controls (e.g., sliders for ranges, dropdowns for categories).
    -   The `FilterPanel.jsx` component consumes these definitions and the current filter state to render the UI for adding, editing, and toggling filters.
    -   Filter state (the list of currently applied filter configurations) is managed in `HomePage.jsx` and passed down to `FilterPanel.jsx`. Changes are propagated back via callback functions.
    -   The actual filtering logic is encapsulated in `filterUtils.js` (`passesAllFilters` function), which processes an erratic against the list of active filters.
-   **Current Filterable Attributes:**
    -   Size (min/max `size_meters`)
    -   Proximity to Water (max `nearest_water_body_dist`)
    -   Rock Type (select from distinct `rock_type` values)
    -   Usage Type (select from distinct `usage_type` tags)
    -   Has Inscriptions (boolean `has_inscriptions`)
-   **Extensibility:**
    -   The system is designed to be extensible. New filters based on other intrinsic erratic properties or computed spatial analysis results (from the `ErraticAnalyses` table) can be added by:
        1.  Ensuring the relevant data fields (including those from `ErraticAnalyses`) are fetched with the erratic data in `HomePage.jsx`.
        2.  Adding a new entry to `GLOBAL_FILTER_DEFINITIONS` in `HomePage.jsx`, including its UI component.
        3.  Updating the `switch` statement in `filterUtils.js` to include the logic for the new filter type.
    -   This will allow users to, for example, filter erratics by `accessibility_score`, `elevation_category`, `nearest_colonial_road_dist`, etc., once these analyses are fully populated and data is consistently available to the frontend.

## Spatial Analysis Features

The backend Python scripts provide significant analytical capabilities:

- **Data Integration:** Automatically downloads and utilizes data from key North American GIS sources:
  * HydroSHEDS (Rivers, Lakes - uses PostGIS or file fallback)
  * Native Land Digital (Territories - currently experiencing access issues, uses API fallback)
  * OpenStreetMap Extracts (Settlements, Modern Roads - requires PBF file download, uses `ogr2ogr` for processing, has fallback data)
  * DAART (Colonial Roads - uses GeoJSON download, has fallback data)
  * NHGIS (Historical Settlements - uses Shapefile download from Zip, has fallback data)
  * SRTM 90m DEM (Downloads required tiles via FTP, uses `rasterio` for processing)
- **Proximity Metrics:** Calculates Haversine distances to the nearest feature in various layers.
- **Contextual Analysis:** Derives attributes based on location and relationships:
  * `elevation_category`: Classifies elevation based on DEM data.
  * `accessibility_score`: Rates 1-5 based on distance to nearest modern road and settlement.
  * `size_category`: Small, Medium, Large, Monumental based on `size_meters`.
  * `estimated_displacement_dist`: Simplified estimation of glacial transport distance based on latitude.
  * Terrain Metrics: Calculates `ruggedness_tri`, `terrain_landform`, `terrain_slope_position` using DEM data via `rasterio`.
- **NLP Classification & Topic Modeling:**
  * Combines `description`, `cultural_significance`, and `historical_notes`.
  * Preprocesses text using spaCy/NLTK.
  * Generates sentence embeddings (`sentence-transformers`).
  * Trains topic models (BERTopic or LDA) via `--build-topics` flag and saves them.
  * Classifies erratics using loaded models, identifying dominant topic (`usage_type`), `cultural_significance_score`, `has_inscriptions` (keyword-based), and `vector_embedding`.
- **Inter-Erratic Clustering:**
  * Script `clustering.py` implements DBSCAN, K-Means, and Hierarchical clustering.
  * Can cluster based on coordinates or other numerical/vector features.
  * Calculates Silhouette Score for evaluating cluster quality.

## Methodological Challenges & Known Issues

Working with named glacial erratics presents unique methodological challenges for spatial analysis, particularly due to their complex historical, cultural, and geological attributes. Our analysis pipeline must confront these issues to provide meaningful results.

### Data Sparsity and Heterogeneity 

Named glacial erratics often have uneven documentation across various attributes:
* **Geological Properties:** Many lack precise measurements (size, composition, displacement distance).
* **Historical Documentation:** Varies dramatically (Plymouth Rock vs. remote erratics with minimal records).
* **Cultural Significance:** Qualitative and varies in depth (indigenous traditions vs. colonial accounts).
* **Temporal Changes:** Many erratics have been moved, fractured, or altered throughout recorded history.

### Case Study Challenges

Our work with nine prominent North American erratics highlights specific methodological issues:

#### Positional Ambiguity and Movement

* **Plymouth Rock** (Massachusetts): Repeatedly moved and fractured since the 17th century. Only represents a portion of the original boulder, challenging our efforts to record its "true" location and size. Its primarily symbolic national significance overshadows its modest geological importance, creating classification challenges.

* **Dighton Rock** (Massachusetts): Famous for inscrutable petroglyphs (variously attributed to Native Americans, Norse explorers, or Portuguese sailors). Moved from original river location to a museum. We faced decisions about which location to use (original vs. current), impacting proximity analyses to waterways.

* **Rollstone Boulder** (Massachusetts): Originally on Rollstone Hill in Fitchburg, it was broken apart and reassembled in a downtown park when threatened by quarrying. Our database must decide whether to represent original or current location, significantly affecting terrain analysis and proximity to settlements.

#### Classification Complexity

* **Willamette Meteorite** (Oregon): Technically a meteorite (extraterrestrial) but transported by glacial ice and/or Missoula Floods, challenging our classification system which assumes standard terrestrial glacial transport. Additionally sacred to Clackamas Chinook people, requiring detailed cultural significance encoding. Its `estimated_displacement_dist` can't be calculated like standard erratics.

* **Judges Cave** (Connecticut): Not a true "erratic" but rather a formation of several glacially-transported boulders creating a natural shelter. Gained significance as a 17th-century hiding place for regicide judges. Challenges our data model, which assumes individual boulders rather than composite features.

#### Multiplicity Problems

* **Babson's Boulders** (Massachusetts): Represents dozens of inscribed glacial boulders spread across Dogtown Common, not a single erratic. Created difficulties in our database schema: Do we create separate entries for each boulder? Use a centroid for the collection? Create a polygon? Different approaches significantly impact clustering analysis results.

#### Scale and Outlier Management

* **Okotoks "Big Rock" Erratic** (Alberta): At ~16,500 tons and 9 meters tall, this enormous quartzite erratic presents an extreme size outlier, potentially skewing our `size_category` classification system. Also deeply sacred to the Blackfoot people (Napi/Old Man mythology), requiring detailed cultural encoding.

* **Madison Boulder** (New Hampshire): One of North America's largest glacial erratics (estimated 4,662 tons), challenges our size classification system and requires special handling to avoid distorting aggregate statistics.

* **Bleasdell Boulder** (Ontario): While well-documented as a local landmark, finding consistent historical information presented challenges compared to more famous erratics, highlighting the uneven documentation across different regions.

### Impact on Spatial Analysis Pipeline

These challenges required specific adaptations to our analysis approach:

1. **Flexible Coordinate System:** Our database accommodates multiple coordinate points for moved erratics (original and current locations).

2. **Classification Nuance:** The `usage_type` array field allows multiple categorizations instead of forcing single classifications.

3. **Confidence Fields:** Where precise measurements are unavailable, we've implemented uncertainty markers.

4. **Historical Context Prioritization:** The analysis pipeline includes toggles to prioritize either historical or current locations for proximity calculations.

5. **Collection Handling:** Special processing for feature collections (like Babson's Boulders) to prevent artificial clustering results.

6. **Cultural Significance Enhancement:** Our NLP topic modeling employs specialized techniques to capture indigenous cultural significance when present in text descriptions.

7. **Methodological Transparency:** Analysis results include flags when calculations are based on estimated or ambiguous data.

### Data Model Adaptations

To address these challenges, our data model incorporates:

* **Temporal Fields:** Capturing location/attribute changes over time
* **Confidence Scoring:** For uncertain attributes
* **Location Type Classification:** Original, current, ceremonial, etc.
* **Relationship Linkages:** Connecting related erratics (e.g., Babson's collection)
* **Special Classification Flags:** For edge cases (meteorite+erratic, composite formations)

These methodological challenges ultimately reveal the rich complexity behind seemingly simple geological features, reinforcing the need for nuanced spatial analysis that respects both precise geospatial relationships and the deep cultural meanings embedded within these natural landmarks.

### Current Known Issues / Areas for Improvement:
*   **Native Land Digital API:** The current API endpoint (https://nativeland.info/api/index.php?maps=territories) returns a 403 Forbidden error. The script currently falls back gracefully, but this data source is unavailable. The API endpoint or access method may need updating.
*   **OSM PBF Data:** The PBF processing logic using `ogr2ogr` is implemented in `data_loader.py`, but requires the relevant `.osm.pbf` file (e.g., `north-america-latest.osm.pbf`) to be manually downloaded and placed in the correct directory (`backend/src/scripts/python/data/gis/settlements/osm_north_america/`). Otherwise, the script uses hardcoded fallback data for modern roads and settlements.
*   **NHGIS Data:** The `load_colonial_settlements` function attempts to find a specific shapefile (`*_place_*.shp`) within the downloaded `nhgis_historical.zip`. This may need adjustment based on the actual contents of the archive to load the correct settlement data instead of the fallback.
*   **Database Schema Synchronization:** While migrations have been created, ensuring the database schema perfectly matches the Sequelize models (`Erratic.js`, `ErraticAnalysis.js`) is crucial. Future model changes must be accompanied by corresponding migration files.
*   **Terrain Data:** Uses SRTM 90m data due to availability constraints, not the originally intended 30m. Coverage is limited to below 60 degrees latitude.
*   **Performance:** Downloading and processing large datasets (PBF, DEM tiles) can be time-consuming on the first run. Caching (using Apache Feather format) mitigates this for subsequent runs. Complex analyses (clustering, topic modeling) can also be computationally intensive.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Glacial erratic data sourced from various geological databases, historical records, and publications.
- Mapping libraries and components from Leaflet.js and React-Leaflet.
- Project goals influenced by the analysis requirements in `SpatialAnalysisMetaPrompt.md`.