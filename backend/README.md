# Glacial Erratics Map - Backend

A robust Node.js backend providing RESTful APIs for glacial erratics data management, spatial analysis integration, and user authentication. Built with Express.js, PostgreSQL/PostGIS, and integrated Python analysis scripts.

## Overview

The backend serves as the central data management and analysis coordination layer for the Glacial Erratics Map application. It provides APIs for erratic data retrieval, manages spatial analysis workflows, and coordinates with Python scripts for advanced geospatial computations.

## Architecture

### Technology Stack

- **Runtime**: Node.js 20.19.0+
- **Framework**: Express.js for RESTful APIs
- **Database**: PostgreSQL 14+ with PostGIS extension
- **ORM**: Sequelize for database operations
- **Authentication**: JWT-based authentication
- **Process Management**: Child process execution for Python scripts
- **Extensions**: pgvector for vector similarity search (optional)
- **Logging**: Winston with daily rotation

### Project Structure

```
backend/
├── config/
│   └── database.js          # Database configuration
├── src/
│   ├── controllers/         # Request handlers
│   │   ├── erraticController.js
│   │   ├── analysisController.js
│   │   └── authController.js
│   ├── models/             # Sequelize data models
│   │   ├── index.js
│   │   ├── Erratic.js
│   │   ├── ErraticAnalysis.js
│   │   ├── ErraticMedia.js
│   │   ├── ErraticReference.js
│   │   └── User.js
│   ├── routes/             # API route definitions
│   │   ├── erratics.js
│   │   ├── analysis.js
│   │   └── auth.js
│   ├── services/           # Business logic services
│   │   └── pythonService.js
│   ├── utils/              # Utility functions
│   │   ├── auth.js
│   │   ├── logger.js
│   │   └── dbInit.js
│   ├── migrations/         # Database schema migrations
│   ├── scripts/            # Utility scripts and Python integration
│   │   ├── importData.js
│   │   └── python/         # Python analysis scripts
│   └── index.js           # Application entry point
├── tests/                  # Test suites
│   ├── unit/
│   └── integration/
├── logs/                   # Application logs
├── package.json           # Dependencies and scripts
└── README.md              # This file
```

## API Documentation

### Base URL

```
Development: http://localhost:3001/api
Production: https://your-domain.com/api
```

### Authentication

The API uses JWT (JSON Web Tokens) for authentication. Admin routes require a valid token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Endpoints

#### Erratics

**Get All Erratics**
```
GET /api/erratics
```
Returns all erratics with optional joined analysis data.

Response:
```json
[
  {
    "id": 1,
    "name": "Plymouth Rock",
    "location": {
      "type": "Point",
      "coordinates": [-70.6620, 41.9584]
    },
    "size_meters": 3.0,
    "rock_type": "Dedham Granite",
    "elevation": 2.0,
    "description": "Historic landing site...",
    "cultural_significance": "Symbolic importance...",
    "historical_notes": "Associated with Mayflower...",
    "image_url": "https://example.com/image.jpg",
    "createdAt": "2024-01-01T00:00:00.000Z",
    "updatedAt": "2024-01-01T00:00:00.000Z",
    // Joined analysis data
    "nearest_water_body_dist": 15.2,
    "nearest_settlement_dist": 200.5,
    "nearest_natd_road_dist": 50.0,
    "nearest_forest_trail_dist": 1500.0,
    "nearest_native_territory_dist": 0.0,
    "accessibility_score": 5,
    "size_category": "Medium",
    "usage_type": ["ceremonial", "landmark"],
    "has_inscriptions": false,
    "cultural_significance_score": 8,
    "elevation_category": "coastal",
    "geological_type": "igneous",
    "estimated_displacement_dist": 15000.0,
    "vector_embedding": [0.1, 0.2, ...],
    "ruggedness_tri": 0.05,
    "terrain_landform": "coastal_plain",
    "terrain_slope_position": "flat"
  }
]
```

**Get Single Erratic**
```
GET /api/erratics/:id
```
Returns a specific erratic by ID with analysis data.

**Get Nearby Erratics**
```
GET /api/erratics/nearby?lat=42.0&lng=-71.0&radius=10000
```
Query Parameters:
- `lat`: Latitude (required)
- `lng`: Longitude (required) 
- `radius`: Search radius in meters (default: 10000)

**Create Erratic** (Admin)
```
POST /api/erratics
```
Creates a new erratic record.

**Update Erratic** (Admin)
```
PUT /api/erratics/:id
```
Updates an existing erratic.

**Delete Erratic** (Admin)
```
DELETE /api/erratics/:id
```
Deletes an erratic and related records.

#### Analysis

**Proximity Analysis**
```
GET /api/analysis/proximity/:id
```
Triggers proximity analysis for a specific erratic. Executes Python script and updates database.

Response:
```json
{
  "success": true,
  "message": "Proximity analysis completed",
  "erraticId": 1,
  "analysisData": {
    "nearest_water_body_dist": 15.2,
    "nearest_settlement_dist": 200.5,
    // ... other proximity metrics
  }
}
```

**Batch Proximity Analysis** (Admin)
```
POST /api/analysis/proximity/batch
```
Request body:
```json
{
  "erraticIds": [1, 2, 3, 4, 5]
}
```

**Erratic Classification**
```
GET /api/analysis/classify/:id
```
Runs NLP classification analysis on erratic text data.

**Batch Classification** (Admin)
```
POST /api/analysis/classify/batch
```

**Spatial Clustering**
```
GET /api/analysis/cluster
```
Query Parameters:
- `algorithm`: 'dbscan', 'kmeans', or 'hierarchical' (default: 'dbscan')
- `features`: Comma-separated feature names (default: 'latitude,longitude')
- `algoParams`: JSON string of algorithm parameters
- `outputToFile`: 'true' or 'false' (default: 'true')
- `outputFilename`: Output filename (default: 'clustering_results.json')

**Build Topic Models** (Admin)
```
POST /api/analysis/build-topics
```
Triggers topic model training for NLP classification.

#### Authentication

**Login**
```
POST /api/auth/login
```
Request body:
```json
{
  "username": "admin",
  "password": "your-password"
}
```

Response:
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "admin",
    "email": "admin@example.com",
    "is_admin": true
  }
}
```

**Get Profile**
```
GET /api/auth/profile
```
Returns current user information (requires authentication).

## Database Models

### Erratic
Primary model storing basic erratic information.

```javascript
{
  id: INTEGER (Primary Key),
  name: STRING (Required),
  location: GEOMETRY('POINT', 4326) (Required),
  elevation: FLOAT,
  size_meters: FLOAT,
  rock_type: STRING,
  estimated_age: STRING,
  discovery_date: DATE,
  description: TEXT,
  cultural_significance: TEXT,
  historical_notes: TEXT,
  image_url: STRING,
  timestamps: true
}
```

### ErraticAnalysis
Stores computed analysis results.

```javascript
{
  erraticId: INTEGER (Primary Key, Foreign Key),
  usage_type: ARRAY(STRING),
  cultural_significance_score: INTEGER,
  has_inscriptions: BOOLEAN,
  accessibility_score: INTEGER,
  size_category: STRING,
  nearest_water_body_dist: FLOAT,
  nearest_settlement_dist: FLOAT,
  nearest_road_dist: FLOAT,
  nearest_native_territory_dist: FLOAT,
  nearest_natd_road_dist: FLOAT,
  nearest_forest_trail_dist: FLOAT,
  elevation_category: STRING,
  geological_type: STRING,
  estimated_displacement_dist: FLOAT,
  vector_embedding: VECTOR(384),
  ruggedness_tri: FLOAT,
  terrain_landform: STRING,
  terrain_slope_position: STRING,
  timestamps: true
}
```

### ErraticMedia
Media files associated with erratics.

```javascript
{
  id: INTEGER (Primary Key),
  erraticId: INTEGER (Foreign Key),
  media_type: ENUM('image', 'video', 'document', 'other'),
  url: STRING (Required),
  title: STRING,
  description: TEXT,
  credit: STRING,
  capture_date: DATE,
  timestamps: true
}
```

### ErraticReference
Scientific references and citations.

```javascript
{
  id: INTEGER (Primary Key),
  erraticId: INTEGER (Foreign Key),
  reference_type: ENUM('article', 'book', 'paper', 'website', 'other'),
  title: STRING (Required),
  authors: STRING,
  publication: STRING,
  year: INTEGER,
  url: STRING,
  doi: STRING,
  description: TEXT,
  timestamps: true
}
```

### User
Admin user accounts.

```javascript
{
  id: INTEGER (Primary Key),
  username: STRING (Unique, Required),
  email: STRING (Unique, Required),
  password: STRING (Hashed, Required),
  is_admin: BOOLEAN (Default: false),
  last_login: DATE,
  timestamps: true
}
```

## Python Integration

### Python Service

The `pythonService.js` module handles execution of Python analysis scripts:

```javascript
const { execPython } = require('./services/pythonService');

// Execute proximity analysis
const result = await execPython('proximity_analysis.py', [
  erraticId,
  '--update-db',
  '--output', 'results.json'
]);
```

### Available Scripts

- **proximity_analysis.py**: Calculates distances to various geographic features
- **classify_erratic.py**: NLP-based text classification and topic modeling
- **clustering.py**: Spatial clustering analysis
- **importData.js**: Data import utilities

### Environment Requirements

Python scripts require:
- Python 3.10+ with Conda environment (`iqp-py310`)
- GDAL/OGR tools for geospatial processing
- Various Python packages (see `src/scripts/python/requirements.txt`)

## Development

### Setup

1. **Install Dependencies**
```bash
npm install
```

2. **Environment Configuration**
Create `.env` file in project root:
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=glacial_erratics
DB_USER=your_username
DB_PASSWORD=your_password

# Authentication
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=24h

# Environment
NODE_ENV=development
PORT=3001
```

3. **Database Setup**
```bash
# Run migrations
npm run db:migrate

# Import data (optional)
npm run db:import
```

4. **Start Development Server**
```bash
npm run dev
```

### Scripts

```bash
# Development server with auto-reload
npm run dev

# Production server
npm start

# Database operations
npm run db:migrate          # Run pending migrations
npm run db:migrate:undo     # Undo last migration
npm run db:import           # Import CSV data

# Testing
npm test                    # Run test suite
npm run test:unit          # Unit tests only
npm run test:integration   # Integration tests only

# Python analysis
npm run spatial:analyze    # Run spatial analysis on all erratics
```

### Database Migrations

Create new migration:
```bash
npx sequelize-cli migration:generate --name migration-name
```

Run migrations:
```bash
npx sequelize-cli db:migrate
```

Undo migrations:
```bash
npx sequelize-cli db:migrate:undo
```

## Testing

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── controllers/
│   ├── models/
│   └── services/
├── integration/            # Integration tests
│   ├── api/
│   └── database/
└── mocks/                 # Test mocks and fixtures
```

### Running Tests

```bash
# All tests
npm test

# With coverage
npm run test:coverage

# Watch mode
npm run test:watch

# Specific test file
npm test -- --grep "erratic controller"
```

## Deployment

### Environment Variables

Production requires these environment variables:

```env
NODE_ENV=production
PORT=3001
DB_HOST=your-production-db-host
DB_NAME=your-production-db-name
DB_USER=your-production-db-user
DB_PASSWORD=your-production-db-password
JWT_SECRET=your-strong-production-secret
```

### Production Checklist

- [ ] Set `NODE_ENV=production`
- [ ] Use strong JWT secret
- [ ] Configure production database
- [ ] Set up HTTPS
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Configure backup strategy
- [ ] Set up Python environment on server

## Monitoring & Logging

### Winston Logger

Logs are written to:
- `logs/application.log` - General application logs
- `logs/error.log` - Error logs only
- Console output in development

### Log Levels

- `error`: System errors and exceptions
- `warn`: Warning conditions
- `info`: General information
- `debug`: Debug information (development only)

### Health Checks

The application provides basic health endpoints:

```
GET /health              # Basic health check
GET /health/database     # Database connectivity check
```

## Security

### Authentication

- JWT tokens for session management
- Bcrypt password hashing
- CORS configuration for cross-origin requests

### Input Validation

- Sequelize model validation
- Request parameter sanitization
- SQL injection prevention via ORM

### Best Practices

- Environment-based configuration
- Secure headers via middleware
- Rate limiting (recommended for production)
- HTTPS enforcement (production)

## Performance

### Database Optimization

- Indexes on frequently queried columns
- PostGIS spatial indexes for location queries
- Connection pooling via Sequelize

### Caching

- Python analysis results cached in database
- Consider Redis for session storage (production)

### Monitoring

Recommended monitoring:
- Database query performance
- API response times
- Memory usage
- Python script execution times

## Troubleshooting

### Common Issues

**Database Connection Errors**
- Verify PostgreSQL is running
- Check connection credentials in `.env`
- Ensure PostGIS extension is installed

**Python Script Failures**
- Verify Conda environment is activated
- Check Python script logs in `logs/`
- Ensure required GIS data files are present

**Migration Errors**
- Check database permissions
- Verify migration file syntax
- Review existing schema state

**Authentication Issues**
- Verify JWT secret configuration
- Check token expiration settings
- Validate user credentials

### Debug Mode

Enable detailed logging:
```bash
NODE_ENV=development DEBUG=* npm run dev
```

## Contributing

### Code Style

- ESLint configuration enforced
- Prettier formatting for consistency
- JSDoc comments for public functions
- Meaningful commit messages

### Adding Features

1. **Plan the change**: Document requirements and approach
2. **Update models**: Add/modify Sequelize models if needed
3. **Create migrations**: Database schema changes require migrations
4. **Implement controllers**: Add/modify request handlers
5. **Update routes**: Register new endpoints
6. **Add tests**: Unit and integration tests required
7. **Update documentation**: Keep README and API docs current

### Database Changes

All schema changes must be implemented via Sequelize migrations:

1. Generate migration file
2. Implement `up()` and `down()` functions
3. Test migration in development
4. Update model definitions
5. Update API documentation

## API Versioning

Currently using v1 APIs. Future versions should:
- Maintain backward compatibility
- Use URL versioning (`/api/v2/`)
- Document breaking changes
- Provide migration guides

## Future Enhancements

### Planned Features

- **Real-time Updates**: WebSocket support for live data updates
- **Advanced Caching**: Redis integration for improved performance
- **API Rate Limiting**: Request throttling for production
- **Batch Operations**: Bulk create/update endpoints
- **Data Export**: CSV/GeoJSON export functionality
- **Audit Logging**: Track data changes for research purposes

### Performance Improvements

- **Database Optimization**: Query optimization and indexing
- **Connection Pooling**: Advanced pool configuration
- **Horizontal Scaling**: Load balancer support
- **Background Jobs**: Queue system for long-running tasks 