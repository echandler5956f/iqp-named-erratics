# Glacial Erratics Map - Codebase Documentation

## Project Overview

This documentation provides a comprehensive overview of the Glacial Erratics Map application, a full-stack web application that displays a dynamic, interactive map of named glacial erratics located in North America (USA and Canada). The application allows users to explore these geological features, view detailed information, and leverages backend spatial analysis capabilities.

### Key Features

- Interactive map (React-Leaflet) with multiple toggleable base layers (OpenStreetMap, Satellite, Terrain).
- Markers for named glacial erratics with detailed pop-up and sidebar information.
- User location tracking for finding nearby erratics.
- PostgreSQL/PostGIS backend storing comprehensive data, including spatial analysis fields.
- Integration with Python scripts for spatial analysis (proximity, classification) via Node.js backend.
- RESTful API for data retrieval and analysis tasks.
- Admin interface for managing erratic data (CRUD operations).
- JWT-based authentication for admin access.

## Architecture Overview

The application follows a standard client-server architecture:

- **Frontend**: React-based single-page application using React-Leaflet for mapping.
- **Backend**: Node.js/Express API server with PostgreSQL/PostGIS database.
- **Spatial Analysis**: Python scripts executed by the Node.js backend for computationally intensive tasks.
- **Authentication**: JWT-based authentication for admin access.

## Directory Structure

```
/
├── frontend/                  # React frontend application
│   ├── src/
│   │   ├── components/        # Reusable UI components (e.g., Map, Auth, Layout)
│   │   │   ├── Map/           # Primary map component (ErraticsMap.jsx)
│   │   │   └── ...            # Other UI components
│   │   ├── pages/             # Page components (HomePage, AdminPage, etc.)
│   │   ├── contexts/          # React context providers (if any)
│   │   ├── hooks/             # Custom React hooks (currently unused)
│   │   ├── services/          # API service functions (currently unused, logic in components)
│   │   ├── utils/             # Utility functions
│   │   ├── assets/            # Static assets (icons, images)
│   │   ├── App.jsx            # Main application component
│   │   └── main.jsx           # Application entry point
│   └── public/                # Static files (e.g., custom erratic icon)
├── backend/                   # Express backend API server
│   ├── src/
│   │   ├── controllers/       # Request handlers (erraticController, authController, analysisController)
│   │   ├── models/            # Database models (Sequelize definitions - Erratic, User, etc.)
│   │   ├── routes/            # API route definitions (erraticRoutes, authRoutes, analysisRoutes)
│   │   ├── services/          # Business logic services (e.g., pythonService)
│   │   ├── utils/             # Utility functions (e.g., auth middleware)
│   │   ├── data/              # Data files and import scripts (if any)
│   │   ├── scripts/           # Python analysis scripts called by backend
│   │   └── index.js           # Server entry point
│   └── .env                   # Environment configuration
```

## Database Schema

The application uses PostgreSQL with the PostGIS extension, managed via Sequelize. Key models include:

### Erratic

Represents a glacial erratic with its location and comprehensive attributes:

```javascript
{
  // Core fields
  id: Integer (PK),
  name: String,
  location: Point (PostGIS geometry),
  elevation: Float,
  size_meters: Float,
  rock_type: String,
  estimated_age: String,
  discovery_date: Date,
  description: Text,
  cultural_significance: Text,
  historical_notes: Text,
  image_url: String,
  
  // Spatial Analysis / ML Fields (from SpatialAnalysisMetaPrompt.md)
  usage_type: Array<String>,
  cultural_significance_score: Integer,
  has_inscriptions: Boolean,
  accessibility_score: Integer,
  size_category: String,
  nearest_water_body_dist: Float,
  nearest_settlement_dist: Float,
  elevation_category: String,
  geological_type: String,
  estimated_displacement_dist: Float,
  vector_embedding_data: JSONB // Placeholder for vector embeddings
}
```

### ErraticMedia

Stores media files (images, videos, documents) related to erratics:

```javascript
{
  id: Integer (PK),
  erraticId: Integer (FK -> Erratics.id),
  media_type: Enum('image', 'video', 'document', 'other'),
  url: String,
  title: String,
  description: Text,
  credit: String,
  capture_date: Date
}
```

### ErraticReference

Stores scholarly references related to erratics:

```javascript
{
  id: Integer (PK),
  erraticId: Integer (FK -> Erratics.id),
  reference_type: Enum('article', 'book', 'paper', 'website', 'other'),
  title: String,
  authors: String,
  publication: String,
  year: Integer,
  url: String,
  doi: String,
  description: Text
}
```

### User

Admin user accounts for managing the application:

```javascript
{
  id: Integer (PK),
  username: String,
  email: String,
  password: String (hashed),
  is_admin: Boolean,
  last_login: Date
}
```

## Backend API

The backend provides a RESTful API with the following key endpoints:

### Erratic Endpoints

- `GET /api/erratics`: Get all erratics (with basic fields).
- `GET /api/erratics/:id`: Get a specific erratic by ID (includes associations like media/references).
- `GET /api/erratics/nearby`: Get erratics near a specific location (`lat`, `lng`, `radius`).
- `POST /api/erratics`: Create a new erratic (Admin only).
- `PUT /api/erratics/:id`: Update an existing erratic (Admin only).
- `DELETE /api/erratics/:id`: Delete an erratic (Admin only).

### Analysis Endpoints

- `GET /api/analysis/proximity/:id`: Get proximity analysis results for a single erratic. Executes Python script. Query param `?update=true` can trigger DB update. (Public)
- `POST /api/analysis/proximity/batch`: Run batch proximity analysis for multiple erratic IDs. Executes Python script. (Admin only)
- `GET /api/analysis/classify/:id`: Classify a single erratic based on its data. Executes Python script. Query param `?update=true` can trigger DB update. (Public)
- `POST /api/analysis/classify/batch`: Run batch classification for multiple erratic IDs. Executes Python script. (Admin only)

### Authentication Endpoints

- `POST /api/auth/login`: Admin login, returns JWT token.
- `GET /api/auth/profile`: Get authenticated admin's profile information.

## Frontend Components

### Core Components

#### ErraticsMap (`frontend/src/components/Map/ErraticsMap.jsx`)

The central component responsible for the map interface:

- Uses `React-Leaflet` to render the map container, tile layers, markers, and popups.
- Implements `LayersControl` for switching base maps (OpenStreetMap, Satellite, Terrain).
- Fetches erratic data using `axios` directly within the component's `useEffect` hook.
- Displays erratic markers with custom icons (`erratic-icon.png`).
- Shows basic info in marker popups and detailed info in a sidebar when a marker is clicked.
- Includes a `LocationMarker` component to track user's location and find nearby erratics.
- Handles loading and error states for data fetching.

#### Authentication Components (`frontend/src/components/Auth/`)

Likely contains components for the login process (e.g., `LoginForm.jsx`).

#### Admin Components (`frontend/src/pages/AdminPage.jsx`)

Provides the UI for admin users to perform CRUD operations on erratic data, likely including forms and data tables.

### State Management

- Primarily uses React's built-in state management (`useState`, `useEffect`).
- No dedicated state management library (like Redux or Zustand) or extensive use of Context API observed in `ErraticsMap.jsx`.

## Implementation Details

### Map Implementation

- Leverages `React-Leaflet` heavily.
- Base layers provided: OpenStreetMap, Esri World Imagery (Satellite), Stamen Terrain.
- Custom erratic marker icon loaded from `public/erratic-icon.png`.
- Popups provide quick info; a sidebar (`erratic-sidebar`) displays full details for the selected erratic.
- User location is obtained using the Leaflet's `map.locate()` method.

### Backend-Python Interaction

- The Node.js backend uses a service (`backend/src/services/pythonService.js`, not fully inspected but inferred) to execute Python scripts located in `backend/src/scripts/`.
- Data is likely passed as command-line arguments, and results are returned via stdout (parsed as JSON).
- Analysis endpoints (`/api/analysis/...`) trigger these Python scripts.
- Batch processing endpoints exist for running analysis on multiple erratics, likely processed sequentially in the background on the server.

### Authentication Flow

- Standard JWT flow: login endpoint validates credentials, issues token.
- Token stored client-side (likely localStorage or sessionStorage).
- Token sent in `Authorization` header for protected API routes.
- Backend uses middleware (`backend/src/utils/auth.js`) to verify tokens and check admin privileges.

### Data Flow

1. Frontend (`ErraticsMap.jsx`) fetches initial erratic data from `/api/erratics`.
2. Data is stored in component state and rendered as markers.
3. User interaction (clicking marker) updates state to show details in the sidebar.
4. User location tracking triggers fetch to `/api/erratics/nearby`.
5. Admin actions (via `AdminPage.jsx`) call protected API endpoints to modify data.
6. Analysis requests (potentially triggered by user action or admin batch jobs) call `/api/analysis/...` endpoints, which execute Python scripts and potentially update the database.

## Future Enhancements (Based on `SpatialAnalysisMetaPrompt.md`)

The project aims to incorporate more advanced features, building upon the current structure:

1.  **Sophisticated Spatial Analysis**: Implement clustering (HDBSCAN, K-Means), terrain analysis (viewshed, prominence), and distribution pattern analysis (Ripley's K) using Python and PostGIS.
2.  **Machine Learning Classification**: Train models (e.g., using scikit-learn, spacy) to automatically classify erratics based on descriptions (`usage_type`, `cultural_significance_score`) and other features.
3.  **Feature Engineering**: Develop robust feature extraction from text descriptions and spatial relationships.
4.  **Interactive Analysis Tools**: Build UI components allowing users to perform custom queries, comparisons, and potentially run analysis on-the-fly.
5.  **Enhanced Visualizations**: Use analysis results to drive map visualizations (e.g., color-coding by cluster/category, showing analysis layers).
6.  **Vector Embeddings**: Fully utilize vector embeddings (potentially using `pgvector`) for semantic search and ML features, moving beyond the current JSONB placeholder. 