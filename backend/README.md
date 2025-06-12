# Glacial Erratics Map - Backend

A robust Node.js backend providing RESTful APIs for glacial erratics data management, real-time spatial analysis, and user authentication. Built with Express.js, PostgreSQL/PostGIS, and integrated Python analysis scripts.

## Overview

The backend serves as the central data management and analysis coordination layer for the Glacial Erratics Map application. It provides APIs for erratic data retrieval, manages computationally intensive spatial analysis workflows via a job-based system, and coordinates with Python scripts for advanced geospatial computations.

## Architecture

### Technology Stack

- **Runtime**: Node.js 20.x
- **Framework**: Express.js for RESTful APIs
- **Database**: PostgreSQL 14+ with PostGIS and pgvector extensions
- **ORM**: Sequelize for database operations
- **Authentication**: JWT-based authentication for admin routes
- **Async Jobs**: In-memory job queue for long-running analysis tasks
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

#### Erratics (`/api/erratics`)
- `GET /`: Get all erratics. Supports pagination and filtering.
- `GET /:id`: Get a single erratic by ID, including its analysis data.
- `GET /nearby`: Get erratics within a radius. Query params: `lat`, `lng`, `radius`.
- `POST /` (Admin): Create a new erratic.
- `PUT /:id` (Admin): Update an existing erratic.
- `DELETE /:id` (Admin): Delete an erratic.

#### Analysis (`/api/analysis`)

Many analysis endpoints are asynchronous. They accept a request, start a background job, and immediately return a **`202 Accepted`** response with a `job_id`. The status of the job can be polled using the `/jobs/:jobId` endpoint.

- `GET /proximity/:id`: **Synchronous**. Triggers proximity analysis for a *single* erratic and returns the result directly.
- `POST /proximity/batch`: **Asynchronous**. Starts a batch job to run proximity analysis on multiple erratics.
  - **Body**: `{ "erraticIds": [1, 2, 3] }`
- `GET /classify/:id`: **Synchronous**. Runs NLP classification for a *single* erratic and returns the result directly.
- `POST /classify/batch`: **Asynchronous**. Starts a batch job to classify multiple erratics.
  - **Body**: `{ "erraticIds": [1, 2, 3] }`
- `GET /cluster`: **Asynchronous**. Starts a job to perform spatial clustering on all erratics.
  - **Query Params**: `algorithm`, `features`, `algoParams`
- `POST /build-topics` (Admin): **Asynchronous**. Starts a job to build/retrain NLP topic models.

#### Jobs (`/api/analysis/jobs`)
- `GET /:jobId`: Get the status and result of a background job.
  - **Response (example)**:
    ```json
    {
      "id": "batch_proximity_1678886400000_abcdef",
      "type": "batch_proximity",
      "status": "completed",
      "createdAt": "...",
      "updatedAt": "...",
      "params": { "count": 3 },
      "result": { "successful": 3, "failed": 0, "errors": [] },
      "error": null
    }
    ```

#### Authentication (`/api/auth`)
- `POST /login`: Admin login.
- `POST /register` (Admin): Register a new admin user.
- `GET /profile` (Admin): Get current user's profile.

## Database Models

The schema separates core erratic data from computed analysis results for clarity and performance.

### `Erratic`
Stores the fundamental, directly observed, or historically recorded information about each glacial erratic.
- **Fields**: `id`, `name`, `location` (PostGIS Point), `elevation`, `size_meters`, `rock_type`, `description`, etc.
- **Associations**: `hasOne(ErraticAnalysis)`, `hasMany(ErraticMedia)`, `hasMany(ErraticReference)`

### `ErraticAnalysis`
Stores all computed results from the backend Python analysis pipeline. This table has a one-to-one relationship with the `Erratics` table.
- **Fields**: `erraticId` (PK/FK), `usage_type`, `has_inscriptions`, `accessibility_score`, `nearest_water_body_dist`, `vector_embedding` (pgvector), `ruggedness_tri`, etc.

### Other Models
- **ErraticMedia**: Media files associated with erratics.
- **ErraticReference**: Scientific references and citations.
- **User**: Admin user accounts.

## Python Integration

The `pythonService.js` module executes Python analysis scripts from the `src/scripts/python/` directory using `child_process`. It manages passing arguments and parsing JSON results from the scripts.

- **Scripts**: `proximity_analysis.py`, `classify_erratic.py`, `clustering.py`.
- **Environment**: Scripts require the `glacial-erratics` Conda environment to be active when the Node.js server is running. Refer to `src/scripts/python/README.md` for setup and the easy-to-use conda environment setup script (`src/scripts/python/create_conda_env_strict.sh`, which will automatically create a new conda environbment called `glacial-erratics` and install all the necessary Python packages.).

## Development

### Setup

1.  **Install Dependencies**: `npm install`
2.  **Environment**: Create a `.env` file in the `root/` directory.
3.  **Database**: Ensure PostgreSQL is running with the PostGIS and pgvector extensions enabled.
4.  **Start Dev Server**: `npm run dev`

### Database Workflow
The database setup is managed from the **project root**.

1.  **Run Migrations**: `npm run db:migrate` (from project root)
2.  **Seed Initial Data**: `npm run db:import` (from `backend/` directory). This populates the `Erratics` table with initial data from `src/data/erratics.csv`.
3.  **Run Analysis (Optional but Recommended)**: The `ErraticAnalyses` table can be populated by running the spatial analysis scripts. A helper script is provided for this. Specifically, you can run `python src/scripts/python/run_analysis.py pipeline --classify-each --proximity-each --update-db --max-workers=4` to execute the spatial analysis and classification algorithms and update the corresponding database columns.

### Testing

The test suite uses **Mocha** and **Chai**. Tests are located in the `tests/` directory and are separated into `unit` and `integration` tests. Mocks and stubs are created using **Sinon** and **Proxyquire**.

- **Run all tests**:
  ```bash
  npm test
  ```
This command will automatically load environment variables from `tests/test-setup.js` and run all `*.test.js` files.

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