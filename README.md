# Glacial Erratics Map

A full-stack web application that displays a dynamic, interactive map of named glacial erratics with detailed information.

## Project Overview

This application provides an interactive map interface to explore glacial erratics (rocks transported and deposited by glaciers). Features include:

- Multiple toggleable map layers (satellite, terrain, elevation)
- Interactive markers for ~210 named glacial erratics
- Detailed pop-up information for each erratic
- Database backend to store and manage erratic data
- Admin interface for adding/editing erratics
- Responsive design for both desktop and mobile viewing

## Technical Stack

### Frontend
- React (via Vite)
- Leaflet.js for interactive mapping
- React Router for navigation
- Axios for API requests
- Responsive CSS styling

### Backend
- Node.js with Express.js
- PostgreSQL with PostGIS extension for spatial data
- Sequelize ORM for database interactions
- JWT for authentication
- RESTful API architecture

## Setup & Installation

### Prerequisites
- Node.js v20.19.0 and npm v10.8.2 (or compatible versions)
- PostgreSQL 14 with PostGIS extension
- Python 3.10 with geospatial packages (for optional data processing)

### Database Setup
1. Create a PostgreSQL database named 'glacial_erratics'
2. Enable the PostGIS extension on the database
3. Configure connection details in the `.env` file in the backend directory

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

The application uses the following main data models:

- **Erratic**: Core data about each glacial erratic, including location, size, rock type, etc.
- **ErraticMedia**: Images and other media associated with erratics
- **ErraticReference**: Scientific references and sources
- **User**: Admin user accounts for managing erratic data

## API Endpoints

### Public Endpoints
- `GET /api/erratics`: Get all erratics
- `GET /api/erratics/:id`: Get a specific erratic by ID
- `GET /api/erratics/nearby`: Get erratics near a specific location

### Protected Admin Endpoints
- `POST /api/erratics`: Create a new erratic
- `PUT /api/erratics/:id`: Update an existing erratic
- `DELETE /api/erratics/:id`: Delete an erratic
- `POST /api/auth/login`: Admin login
- `GET /api/auth/profile`: Get admin profile information

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

- Glacial erratic data sourced from various geological databases and publications
- Mapping libraries and components from Leaflet.js
- Styling inspiration from modern mapping applications