# Glacial Erratics Map

A full-stack web application that displays a dynamic, interactive map of named glacial erratics in North America, with detailed information and integrated spatial analysis capabilities.

## Project Overview

This application provides an interactive map interface to explore named glacial erratics (large rocks transported and deposited by glaciers) found across North America (USA and Canada). It focuses on presenting detailed geological, historical, and cultural information for each erratic, augmented by **sophisticated backend spatial analysis and NLP/ML classification**. Features include:

- Multiple toggleable map layers (OpenStreetMap, Satellite, Terrain)
- Interactive markers for named glacial erratics
- Detailed pop-up and sidebar information for each erratic (location, size, rock type, description, cultural significance, historical notes, etc.)
- **Real-time Traveling Salesman Problem (TSP) Route Optimization:**
  * Dynamic calculation of optimal visiting routes for filtered erratics
  * Integration with user's current location via geolocation
  * Visual route display with distance calculations
  * Efficient nearest-neighbor + 2-opt heuristic algorithm
  * Real-time recalculation when filters change
- **Comprehensive Backend Analysis:**
  * Integration with diverse GIS datasets (HydroSHEDS, Native Land Digital GeoJSONs, OSM, local road/trail data) providing rich North American context.
  * Calculation of proximity to various features (water bodies, modern settlements & roads, Native territories, forest trails).
  * Contextual characterization based on elevation (from GMTED DEM tiles), accessibility, size, estimated displacement, geomorphology, and terrain metrics (ruggedness, slope position).
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
- **Real-time TSP Solver:** Pure JavaScript implementation with Haversine distance calculations
- **Geolocation API:** User location integration for route optimization
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
- **Core Libraries:** Python 3.10+, `geopandas`, `shapely`, `pandas`, `numpy`, `psycopg2`, `rasterio` (for DEM processing), `joblib` (for model saving), `pytest` (for testing).
- **NLP/ML:** `spacy` (en_core_web_md), `nltk`, `sentence-transformers` (all-MiniLM-L6-v2), `scikit-learn` (for LDA, DBSCAN, K-Means, Hierarchical Clustering, utilities), `kneed` (for DBSCAN eps estimation), potentially `bertopic`, `hdbscan`, `umap` if installed (for advanced topic modeling).
- **External Tools:** Requires `ogr2ogr` (part of GDAL) to be installed and available in the system PATH for processing OpenStreetMap PBF data.
- **Data Handling:** Utilizes a modular `data_pipeline` system (`backend/src/scripts/python/data_pipeline/`) for loading and processing GIS data. This system manages data from PostgreSQL/PostGIS, external URLs (HydroSHEDS, Geofabrik for OSM PBFs), and locally stored files (Native Land GeoJSONs, GMTED DEM tiles, North American roads, National Forest trails). Caching is handled via Apache Feather.
- **Scripts:**
    - `proximity_analysis.py`: Calculates proximity to various features (water bodies, OSM settlements, NATD roads, Native territories, forest trails) and context features (elevation category, accessibility, estimated displacement). Integrates terrain analysis results from `geo_utils.py` using GMTED DEM data.
    - `clustering.py`: Performs spatial clustering on erratics using DBSCAN, K-Means, or Hierarchical methods based on specified features (coordinates, attributes).
    - `classify_erratic.py`: Handles NLP tasks: text preprocessing, generating sentence embeddings, training/loading topic models (BERTopic or LDA), classifying erratic usage/significance, and detecting potential inscriptions.
    - `data_pipeline/` (directory): Contains modules for defining data sources (`data_sources.py`), loading (`loaders.py`), processing (`processors.py`), caching (`cache.py`), and orchestrating (`pipeline.py`) GIS data. Replaces the old `utils/data_loader.py`.
    - `utils/geo_utils.py`: Provides core geospatial functions including distance calculations, nearest neighbor search, DEM data loading (now specifically for tiled GMTED data via the `data_pipeline`), and terrain metric calculations.
    - `utils/db_utils.py`: Manages database connections and CRUD operations for Python scripts.
- **Integration:** Python scripts in `backend/src/scripts/python/` executed by the Node.js backend (`backend/src/services/pythonService.js`) via `child_process`, exchanging data via command-line arguments and JSON output.

## Setup & Installation

### Prerequisites
- Node.js v20.x and npm v10.x (or compatible versions)
- PostgreSQL 14+ with PostGIS and pgvector extensions enabled
- **Python 3.10+** with Conda
- **GDAL (with `ogr2ogr` command-line tool)**

### Database Setup
1. Create a PostgreSQL database (e.g., `glacial_erratics`).
2. **Enable Extensions:** Connect to the database and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   CREATE EXTENSION IF NOT EXISTS vector; 
   ```
3. Configure connection details in a `.env` file in the `root/` directory.

### Installation and Data Workflow
1. Clone the repository and navigate to the project root.
2. Install all Node.js dependencies for the entire project (root, frontend, backend):
   ```bash
   npm run install:all
   ```
3. Run database migrations from the **project root** to create the required tables:
   ```bash
   npm run db:migrate 
   ``` 
5. **Manually Acquire and Place GIS Datasets:**
    Several GIS datasets are expected to be manually downloaded and placed into the `backend/src/scripts/python/data/gis/` directory structure. Refer to `backend/src/scripts/python/data_pipeline/data_sources.py` for the expected `DataSource` names and their `_local_path()` constructions to determine the correct subdirectories. Key local datasets include:
    *   **Native Land Data:** `territories.geojson`, `languages.geojson`, `treaties.geojson` (in `native/`).
    *   **North American Roads (NATD):** `North_American_Roads.shp` (and associated files) (in `roads/`).
    *   **National Forest System Trails:** `National_Forest_System_Trails_(Feature_Layer).shp` (and associated files) (in `trails/`).
    *   **GMTED2010 DEM Tiles:** All individual `.tif` files for the required coverage (e.g., `30n090w_20101117_gmted_mea300.tif` in `elevation/GMTED2010N30W090_300/`, etc.).
    *   **GLCC Data (Future):** Various GLCC `.tif` files (in `glcc/`). (Note: GLCC data integration is planned for the future).
    *   **OpenStreetMap PBF (Optional but Recommended):** Download `north-america-latest.osm.pbf` (or relevant regional extracts like `us-latest.osm.pbf`, `canada-latest.osm.pbf`) from a source like Geofabrik. Place it in `backend/src/scripts/python/data/gis/osm/north_america_pbf/` (or `us_pbf/`, `canada_pbf/` respectively). If not provided, PBF processing will be skipped by the pipeline for these sources.

6. Import the initial core erratic data from the CSV file. This script must be run from the `backend` directory:
```bash
cd backend
npm run db:import
cd .. 
```
7. **Populate Analysis Data (Choose ONE option):**
    The core erratic data is now loaded, but the associated analysis fields (proximity, classification, etc.) are still empty. You can populate them using one of the following methods.

    - **Option A: Run the All-in-One Python Script (Recommended for initial setup)**
      This script iterates through all erratics and runs the necessary proximity and classification analyses to populate the `ErraticAnalyses` table. Make sure your `glacial-erratics` conda environment is active.
      ```bash
      # From the project root directory:
      python backend/src/scripts/python/run_analysis.py --all
      ```

    - **Option B: Use the API Endpoints (For targeted updates or dynamic analysis)**
      After starting the application, you can use the backend's API endpoints to run analyses for individual erratics or in batches. This is useful for updating specific records or if you prefer a more interactive approach.

8. Start the development servers (from the project root):

## Database Structure

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

Stores results computed by the backend Python analysis pipeline.

-   **Key Fields:**
    -   `erraticId`: `INTEGER` (Primary Key, Foreign Key referencing `Erratics.id`)
    -   `usage_type`: `ARRAY(STRING)`
    -   `cultural_significance_score`: `INTEGER`
    -   `has_inscriptions`: `BOOLEAN`
    -   `accessibility_score`: `INTEGER`
    -   `size_category`: `STRING`
    -   `nearest_water_body_dist`: `FLOAT`
    -   `nearest_settlement_dist`: `FLOAT` (to nearest modern settlement - OSM based)
    -   `nearest_road_dist`: `FLOAT` (Distance to nearest road - NATD based)
    -   `nearest_native_territory_dist`: `FLOAT`
    -   `nearest_natd_road_dist`: `FLOAT` (Distance to nearest NATD road)
    -   `nearest_forest_trail_dist`: `FLOAT` (Distance to nearest National Forest trail)
    -   `elevation_category`: `STRING`
    -   `geological_type`: `STRING`
    -   `estimated_displacement_dist`: `FLOAT`
    -   `vector_embedding`: `VECTOR(384)`
    -   `ruggedness_tri`: `FLOAT`
    -   `terrain_landform`: `STRING`
    -   `terrain_slope_position`: `STRING`
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

The backend provides endpoints to trigger analysis. Many are **asynchronous**, meaning they start a background job and return a `job_id`. You can poll the `GET /api/analysis/jobs/:jobId` endpoint to check the status.

- `GET /api/analysis/proximity/:id`: **Synchronous**. Calculates proximity metrics for a single erratic.
- `POST /api/analysis/proximity/batch`: **Asynchronous**. Runs proximity analysis for a list of erratic IDs.
- `GET /api/analysis/classify/:id`: **Synchronous**. Classifies a single erratic using NLP models.
- `POST /api/analysis/classify/batch`: **Asynchronous**. Runs classification for a list of erratic IDs.
- `GET /api/analysis/cluster`: **Asynchronous**. Triggers spatial clustering on all erratics.
- `POST /api/analysis/build-topics` (Admin): **Asynchronous**. Retrains the NLP topic models.

### Protected Admin Endpoints
- `POST /api/erratics`: Create a new erratic.
- `PUT /api/erratics/:id`: Update an existing erratic.
- `DELETE /api/erratics/:id`: Delete an erratic.
- `POST /api/auth/login`: Admin login.
- `GET /api/auth/profile`: Get admin profile information.

## Frontend Features

### Interactive Map

The application features an interactive map (`ErraticsMap.jsx`) built with React-Leaflet, displaying erratics and allowing user interaction.

### Real-time Route Optimization

The application includes a sophisticated Traveling Salesman Problem (TSP) solver that provides optimal route planning:

-   **Dynamic Route Calculation:**
    -   Real-time computation of optimal visiting routes for currently filtered erratics
    -   Integration with user's geolocation as the starting point
    -   Efficient nearest-neighbor construction followed by 2-opt improvement heuristic
    -   Handles 200+ points comfortably with sub-second response times
-   **Visual Route Display:**
    -   Blue polyline showing the optimal path
    -   Waypoint markers at each erratic location
    -   Total distance calculation in kilometers
    -   Automatic map bounds adjustment to show entire route
-   **Interactive Controls:**
    -   Toggle button to show/hide optimal route
    -   "Locate Me" button for manual geolocation requests
    -   Real-time recalculation when filters change
    -   Loading indicators during route computation
-   **Algorithm:**
    -   Pure JavaScript implementation using Haversine distance calculations
    -   Two-phase heuristic: nearest-neighbor construction + 2-opt improvement
    -   Returns high-quality (often optimal) routes with minimal computation time
    -   Scales efficiently for typical erratic datasets (10-300 points)

### Filtering System

The frontend (`HomePage.jsx`) implements a dynamic filtering system allowing users to customize which erratics are displayed on the map.

-   **Architecture:**
    -   Filter definitions (`GLOBAL_FILTER_DEFINITIONS`) are managed in `HomePage.jsx`. These definitions specify the filter's label, default configuration, and the React component used to render its specific UI controls (e.g., sliders for ranges, dropdowns for categories).
    -   The `FilterPanel.jsx` component consumes these definitions and the current filter state to render the UI for adding, editing, and toggling filters.
    -   Filter state (the list of currently applied filter configurations) is managed in `HomePage.jsx` and passed down to `FilterPanel.jsx`. Changes are propagated back via callback functions.
    -   The actual filtering logic is encapsulated in `filterUtils.js` (`passesAllFilters` function), which processes an erratic against the list of active filters.
    -   **TSP Integration:** The filtering system automatically triggers route recalculation when filter selections change, ensuring the optimal route always reflects the current subset of visible erratics.
-   **Current Filterable Attributes (via `Erratic` and joined `ErraticAnalysis` fields):**
    -   Size (min/max `size_meters`)
    -   Proximity to Water (max `nearest_water_body_dist`)
    -   Rock Type (select from distinct `rock_type` values)
    -   Usage Type (select from distinct `usage_type` tags from `ErraticAnalysis`)
    -   Has Inscriptions (boolean `has_inscriptions` from `ErraticAnalysis`)
    -   Accessibility Score (min/max `accessibility_score` from `ErraticAnalysis`, 1-10)
    -   Terrain Landform (select from distinct `terrain_landform` values from `ErraticAnalysis`)
    -   Proximity to NATD Road (max `nearest_natd_road_dist` from `ErraticAnalysis`)
    -   Proximity to Forest Trail (max `nearest_forest_trail_dist` from `ErraticAnalysis`)
    -   Cultural Significance Score (min/max `cultural_significance_score`)
    -   Discovery Date (year range filtering)
    -   Estimated Age (select from distinct values)
    -   Elevation (min/max `elevation` in meters)
    -   Size Category (select from distinct categories)
    -   Geological Type (select from distinct types)
    -   Displacement Distance (min/max `estimated_displacement_dist`)
    -   Terrain Ruggedness (min/max `ruggedness_tri`)
    -   Slope Position (select from distinct positions)
-   **Extensibility:**
    -   The system is designed to be extensible. New filters based on other intrinsic erratic properties or computed spatial analysis results (from the `ErraticAnalyses` table) can be added by:
        1.  Ensuring the relevant data fields (including those from `ErraticAnalyses`) are fetched with the erratic data in `HomePage.jsx`.
        2.  Adding a new entry to `GLOBAL_FILTER_DEFINITIONS` in `HomePage.jsx`, including its UI component.
        3.  Updating the `switch` statement in `filterUtils.js` to include the logic for the new filter type.
    -   The TSP system automatically adapts to any new filterable attributes without requiring additional configuration.

## Spatial Analysis Features

The backend Python scripts, using the new `data_pipeline`, provide analytical capabilities:

- **Data Integration:** Utilizes data from:
  * HydroSHEDS (Rivers, Lakes - via HTTPS download)
  * Native Land Digital (Territories, Languages, Treaties - from local GeoJSON files)
  * OpenStreetMap Extracts (Settlements - PBFs via HTTPS download from Geofabrik, processed with `ogr2ogr`)
  * North American Roads Database (NATD) (Local Shapefile)
  * National Forest System Trails (Local Shapefile)
  * GMTED2010 DEM (Locally stored GeoTIFF tiles, selected based on erratic location)
- **Proximity Metrics:** Calculates Haversine distances to the nearest feature in various layers.
- **Contextual Analysis:** Derives attributes:
  * `elevation_category`: From GMTED DEM.
  * `accessibility_score`: Based on distance to nearest NATD road and OSM settlement.
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
*   **Native Land Digital Data:** Currently uses local GeoJSON files due to previous API issues. These local files need to be kept up-to-date if the original source changes.
*   **OSM PBF Data for `data_pipeline`:** The `osm_north_america` (and US/Canada variants) `DataSource` in `data_pipeline/data_sources.py` relies on PBF files being manually downloaded and placed in the correct directory (e.g., `backend/src/scripts/python/data/gis/osm/north_america_pbf/`). If not present, the pipeline will skip loading these features.
*   **GMTED DEM Tiles:** These must be manually downloaded and placed into their respective `backend/src/scripts/python/data/gis/elevation/GMTED.../` directories. The `geo_utils.load_dem_data` function selects the appropriate tile based on point location from these local files. Coverage is defined by the available tiles.
*   **GLCC Data:** Integration of GLCC land cover data is planned for future enhancement. The `DataSource` definitions exist but are not yet fully utilized in analysis scripts.
*   **Database Schema Synchronization:** Crucial. The recent migration updated `ErraticAnalyses` for new proximity fields. Future model/schema changes need corresponding migrations.
*   **Performance:** Processing large datasets (PBFs, extensive DEM analysis across many erratics) can be intensive. Caching via `data_pipeline` helps, but initial runs or widespread analyses might be slow.

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