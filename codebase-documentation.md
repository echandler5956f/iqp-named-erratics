# Glacial Erratics Map - Codebase Documentation

## Project Overview

This documentation provides a comprehensive overview of the Glacial Erratics Map application, a full-stack web application that displays a dynamic, interactive map of named glacial erratics (rocks transported and deposited by glaciers). The application allows users to explore these geological features with detailed information through an intuitive map interface.

### Key Features

- Interactive map with multiple toggleable layers (satellite, terrain, standard)
- Markers for glacial erratics with detailed pop-up information
- User location tracking for finding nearby erratics
- Admin interface for managing erratic data
- Detailed information display for each erratic

## Architecture Overview

The application follows a standard client-server architecture:

- **Frontend**: React-based single-page application using Leaflet.js for mapping
- **Backend**: Node.js/Express API server with PostgreSQL/PostGIS database
- **Authentication**: JWT-based authentication for admin access

## Directory Structure

```
/
├── frontend/                  # React frontend application
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   │   ├── Auth/          # Authentication-related components
│   │   │   ├── Layout/        # Layout components (header, footer, etc.)
│   │   │   └── Map/           # Map-related components
│   │   ├── pages/             # Page components
│   │   ├── contexts/          # React context providers
│   │   ├── hooks/             # Custom React hooks
│   │   ├── services/          # API service functions
│   │   ├── utils/             # Utility functions
│   │   ├── assets/            # Static assets
│   │   ├── App.jsx            # Main application component
│   │   └── main.jsx           # Application entry point
│   └── public/                # Static files
├── backend/                   # Express backend API server
│   ├── src/
│   │   ├── controllers/       # Request handlers
│   │   ├── models/            # Database models
│   │   ├── routes/            # API route definitions
│   │   ├── services/          # Business logic services
│   │   ├── utils/             # Utility functions
│   │   ├── data/              # Data files and imports
│   │   ├── scripts/           # Helper scripts
│   │   └── index.js           # Server entry point
│   └── .env                   # Environment configuration
```

## Database Schema

The application uses PostgreSQL with PostGIS extension for spatial data with the following primary models:

### Erratic

Represents a glacial erratic with its location and attributes:

```javascript
{
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
  image_url: String
}
```

### ErraticMedia

Stores media files (images, videos, documents) related to erratics:

```javascript
{
  id: Integer (PK),
  erraticId: Integer (FK),
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
  erraticId: Integer (FK),
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

The backend provides a RESTful API with the following endpoints:

### Erratic Endpoints

- `GET /api/erratics`: Get all erratics
- `GET /api/erratics/:id`: Get a specific erratic by ID
- `GET /api/erratics/nearby`: Get erratics near a specific location
- `POST /api/erratics`: Create a new erratic (admin only)
- `PUT /api/erratics/:id`: Update an existing erratic (admin only)
- `DELETE /api/erratics/:id`: Delete an erratic (admin only)

### Authentication Endpoints

- `POST /api/auth/login`: Admin login
- `GET /api/auth/profile`: Get admin profile information

## Frontend Components

### Core Components

#### ErraticsMap

The main map component that displays the interactive map with erratics:

- Renders a Leaflet map with multiple base layers
- Fetches and displays erratic markers
- Handles user location tracking
- Displays detailed information in popups and sidebar

#### Authentication Components

Handles user authentication and admin access:

- Login form
- Protected routes
- JWT token storage and management

#### Admin Interface

Provides CRUD operations for managing erratics:

- Form for adding/editing erratics
- Data table view of all erratics
- Media management

## Implementation Details

### Map Implementation

The map is implemented using Leaflet.js through the React-Leaflet wrapper:

- Multiple base layers: OpenStreetMap, Satellite, Terrain
- Custom markers for erratics
- Popups with basic information
- Sidebar with detailed information
- Location tracking for finding nearby erratics

### Authentication Flow

JWT-based authentication:

1. User submits credentials
2. Server validates credentials and issues a JWT token
3. Token is stored in browser (localStorage)
4. Token is included in subsequent API requests
5. Protected routes are secured with middleware

### Data Flow

1. Frontend fetches erratic data from backend API
2. Data is rendered on the map as markers
3. User interactions (clicks, search) filter or focus on specific erratics
4. Admin actions update data in the database

## Future Enhancements

Based on the Spatial-Analysis.md document, the project is planned to be enhanced with:

1. Advanced spatial analysis features
2. Machine learning for erratic classification
3. Qualitative categorization system
4. Interactive analysis tools
5. Advanced UI/UX for analysis features

These enhancements will include:

- Extended data schema with additional attributes
- NLP-based classification of erratics
- Proximity and terrain analysis
- Clustering algorithms
- Custom query builder for complex searches 