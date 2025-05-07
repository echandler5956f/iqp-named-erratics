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
  * Contextual characterization based on elevation, accessibility, size, estimated displacement, and geomorphology.
  * NLP-based classification using descriptions and notes to uncover thematic categories (via Sentence Transformers and Topic Modeling like BERTopic/LDA).
  * Spatial relationship analysis between erratics (clustering, nearest neighbor distances).
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
- JWT for admin authentication
- RESTful API architecture
- Integration with Python scripts (`child_process`) for specific analysis tasks

### Analysis (Backend/Python Scripts)
- **Core Libraries:** Python 3.10+, `geopandas`, `shapely`, `pandas`, `numpy`, `psycopg2`.
- **NLP/ML:** `spacy` (en_core_web_md), `nltk`, `sentence-transformers` (all-MiniLM-L6-v2), `scikit-learn` (for LDA, DBSCAN), potentially `bertopic`, `hdbscan`, `umap` if installed (for advanced topic modeling/clustering).
- **Data Handling:** Loads data from PostgreSQL/PostGIS and external authoritative GIS sources (HydroSHEDS, Native Land Digital, OSM extracts via Geofabrik, potentially NHGIS, DAART). Downloads and caches external datasets.
- **Integration:** Python scripts in `backend/src/scripts/python/` executed by the Node.js backend (`backend/src/services/pythonService.js`) via `child_process`, exchanging data via command-line arguments and JSON output.

## Setup & Installation

### Prerequisites
- Node.js v20.19.0 and npm v10.8.2 (or compatible versions)
- PostgreSQL 14+ with PostGIS extension enabled
- Python 3.10+ with necessary geospatial packages installed (e.g., `pip install geopandas shapely`)

### Database Setup
1. Create a PostgreSQL database (e.g., `glacial_erratics`).
2. Connect to the database and run `CREATE EXTENSION IF NOT EXISTS postgis;`
3. Configure connection details in the `.env` file in the `backend/` directory.

### Installation Steps
1. Clone the repository
2. Install dependencies:
   ```
   npm run install:all
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env` in the backend directory
   - Update database connection details and JWT secret

4. Initialize the database:
   ```
   cd backend
   npm run db:init
   ```

5. Start the development servers:
   ```
   npm start
   ```

## Database Structure

The application uses PostgreSQL with PostGIS, managed by Sequelize. The core data is split into two main related tables: `Erratics` for intrinsic properties and `ErraticAnalyses` for computed analytical data, linked by a one-to-one relationship.

### Erratic Model (`Erratics` table)

Stores the fundamental, directly observed, or historically recorded information about each glacial erratic.

-   **Core Fields:**
    -   `id`: `INTEGER` (Primary Key, Auto-incrementing)
    -   `name`: `STRING` (Name of the erratic)
    -   `location`: `GEOMETRY('POINT')` (Geographic coordinates)
    -   `elevation`: `FLOAT` (Elevation in meters)
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
    -   Has one `ErraticAnalysis` (links to the computed analysis data)

### ErraticAnalysis Model (`ErraticAnalyses` table)

Stores results computed by the backend Python analysis pipeline. Each record is directly associated with an `Erratic` record.

-   **Key Fields:**
    -   `erraticId`: `INTEGER` (Primary Key, Foreign Key referencing `Erratics.id`)
    -   `usage_type`: `ARRAY(STRING)` (Derived from NLP classification, e.g., "Religious/Ceremonial", "Boundary Marker")
    -   `cultural_significance_score`: `INTEGER` (Score derived from NLP classification or other metrics)
    -   `has_inscriptions`: `BOOLEAN` (Indicates if the erratic has known inscriptions)
    -   `accessibility_score`: `INTEGER` (Score based on proximity to roads, settlements, etc.)
    -   `size_category`: `STRING` (e.g., "Small", "Medium", "Large", "Monumental", based on `size_meters`)
    -   `nearest_water_body_dist`: `FLOAT` (Distance in meters to the nearest significant water body)
    -   `nearest_settlement_dist`: `FLOAT` (Distance in meters to the nearest modern settlement)
    -   `nearest_colonial_settlement_dist`: `FLOAT` (Distance to nearest historical colonial settlement)
    -   `nearest_road_dist`: `FLOAT` (Distance to nearest modern road)
    -   `nearest_colonial_road_dist`: `FLOAT` (Distance to nearest historical colonial road)
    -   `nearest_native_territory_dist`: `FLOAT` (Distance to nearest boundary of a Native/Indigenous territory)
    -   `elevation_category`: `STRING` (e.g., "Lowland", "Upland", based on elevation and surrounding terrain)
    -   `geological_type`: `STRING` (Potentially more detailed geological classification)
    -   `estimated_displacement_dist`: `FLOAT` (Simplified estimation of glacial transport distance)
    -   `vector_embedding`: `JSONB` (Currently stores vector embeddings as JSON. Planned to utilize `pgvector`'s `VECTOR` type for optimized similarity searches.)
    -   `vector_embedding_data`: `JSONB` (Potentially a placeholder or for raw data related to embeddings; its role should be clarified or deprecated if `vector_embedding` is fully implemented with `pgvector`.)
-   **Associations:**
    -   Belongs to `Erratic`

### Other Models

- **ErraticMedia**: Images and other media associated with erratics.
- **ErraticReference**: Scientific references and sources related to erratics.
- **User**: Admin user accounts for managing data.

## API Endpoints

### Public Erratic Endpoints
- `GET /api/erratics`: Get all erratics.
- `GET /api/erratics/:id`: Get a specific erratic by ID.
- `GET /api/erratics/nearby`: Get erratics near a specific location (lat, lng, radius).

### Public Analysis Endpoints (Trigger Python Scripts)
- `GET /api/analysis/proximity/:id`: Calculates detailed proximity metrics for a specific erratic against various feature layers (water, settlements, roads, native territories, etc.) and provides contextual analysis (elevation category, accessibility).
- `GET /api/analysis/classify/:id`: Performs NLP classification on an erratic's text fields, returning topic assignments, significance score, and vector embedding. Requires a pre-built topic model (see admin endpoints).

### Protected Admin Endpoints
- `POST /api/erratics`: Create a new erratic.
- `PUT /api/erratics/:id`: Update an existing erratic.
- `DELETE /api/erratics/:id`: Delete an erratic.
- `POST /api/auth/login`: Admin login.
- `GET /api/auth/profile`: Get admin profile information.
- `POST /api/analysis/proximity/batch`: Run proximity analysis for multiple erratic IDs.
- `POST /api/analysis/classify/batch`: Run NLP classification for multiple erratic IDs (requires pre-built topic model). **Note:** The Python `classify_erratic.py` script includes a `--build-topics` flag, suggesting topic models are built/updated via script execution, likely triggered manually or via an admin process, rather than a dedicated API endpoint for *building* the model itself. Clustering analysis (DBSCAN) seems embedded within `proximity_analysis.py --cluster-analysis` script flag.

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
        1.  Ensuring the relevant data fields are fetched with the erratic data in `HomePage.jsx`.
        2.  Adding a new entry to `GLOBAL_FILTER_DEFINITIONS` in `HomePage.jsx`, including its UI component.
        3.  Updating the `switch` statement in `filterUtils.js` to include the logic for the new filter type.
    -   This will allow users to, for example, filter erratics by `accessibility_score`, `elevation_category`, or proximity to colonial roads once these analyses are fully populated and data is available to the frontend.

## Spatial Analysis Features

The backend Python scripts provide significant analytical capabilities:

- **Data Integration:** Automatically downloads and utilizes data from key North American GIS sources:
  * HydroSHEDS (Rivers, Lakes)
  * Native Land Digital (Territories)
  * OpenStreetMap Extracts (Settlements, Modern Roads via Geofabrik)
  * DAART (Colonial Roads - planned/referenced)
  * NHGIS (Historical Settlements - planned/referenced)
- **Proximity Metrics:** Calculates Haversine distances to the nearest feature in layers like:
  * Lakes (HydroLAKES)
  * Modern Settlements (OSM)
  * Colonial Settlements (NHGIS/Fallback)
  * Modern Roads (OSM/Fallback)
  * Colonial Roads (DAART/Fallback)
  * Native Territories (Native Land Digital)
- **Contextual Analysis:** Derives attributes based on location and relationships:
  * `elevation_category`: Classifies elevation (lowland, upland, mountain, etc.).
  * `accessibility_score`: Rates 1-5 based on distance to nearest road and settlement.
  * `size_category`: Small, Medium, Large, Monumental based on `size_meters`.
  * `estimated_displacement_dist`: Simplified estimation of glacial transport distance.
  * Basic terrain/geomorphological context (landform, slope position - using `geo_utils.py` placeholders).
- **NLP Classification & Topic Modeling:**
  * Combines `description`, `cultural_significance`, and `historical_notes` text fields.
  * Performs preprocessing (lemmatization, stopword removal) using spaCy/NLTK.
  * Generates semantic vector embeddings using `sentence-transformers`.
  * Applies unsupervised topic modeling (BERTopic preferred, fallback to LDA) to discover thematic clusters across the entire dataset's descriptions.
  * Assigns dominant topics and distributions to individual erratics.
  * Calculates a simple `cultural_significance_score` based on topic assignment.
- **Inter-Erratic Clustering:**
  * Calculates an all-pairs distance matrix for erratics.
  * Uses DBSCAN (via scikit-learn, if available) on coordinates with Haversine metric to identify spatial clusters. Triggered via `proximity_analysis.py --cluster-analysis`.

## Methodological Challenges in North American Glacial Erratics Analysis

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