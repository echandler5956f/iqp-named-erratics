const { sequelize } = require('../models');
const logger = require('./logger'); // Import the logger

// Function to check if PostGIS extension is available and enable it
async function enablePostGIS() {
  try {
    // Execute raw SQL to enable PostGIS extension
    await sequelize.query('CREATE EXTENSION IF NOT EXISTS postgis;');
    logger.info('PostGIS extension enabled or already exists.');
  } catch (error) {
    logger.warn('Error enabling PostGIS extension. This might be a permissions issue or PostGIS may not be installed on the DB server. This is usually fine if migrations handle extension creation or if it was pre-installed:', { message: error.message });
    // Do not throw an error here, as the application might still run if PostGIS was enabled manually or by a migration
  }
}

// Function to check if pgvector extension is available and enable it
async function enablePgVector() {
  try {
    await sequelize.query('CREATE EXTENSION IF NOT EXISTS vector;');
    logger.info('pgvector extension enabled or already exists.');
  } catch (error) {
    logger.warn('Error enabling pgvector extension. This might be a permissions issue or pgvector may not be installed on the DB server. This is usually fine if migrations handle extension creation or if it was pre-installed:', { message: error.message });
    // Do not throw an error here
  }
}

// Initialize database connection and check extensions
async function initializeDatabase() {
  try {
    // Test database connection
    try {
      await sequelize.authenticate();
      logger.info('Database connection has been established successfully.');
    } catch (authError) {
      logger.error('Database authentication failed:', { message: authError.message, originalError: authError.parent ? authError.parent.message : undefined });
      throw new Error('Unable to connect to the database. Check your connection details and ensure the database server is running.');
    }
    
    // Attempt to enable PostGIS and pgvector extensions
    // These operations might fail if the DB user lacks permissions or if the extensions aren't installed on the server.
    // It's preferable for extensions to be provisioned by a DBA or dedicated migration script with higher privileges.
    await enablePostGIS();
    await enablePgVector();
    
    logger.info('Database schema should be managed by migrations.');
    
    return true;
  } catch (error) {
    logger.error('Failed to initialize database connection:', { message: error.message });
    return false;
  }
}

module.exports = {
  initializeDatabase
}; 