const { sequelize } = require('../models');
const { Sequelize } = require('sequelize');

// Function to check if PostGIS extension is available and enable it
async function enablePostGIS() {
  try {
    // Execute raw SQL to enable PostGIS extension
    await sequelize.query('CREATE EXTENSION IF NOT EXISTS postgis;');
    console.log('PostGIS extension enabled successfully');
  } catch (error) {
    console.error('Error enabling PostGIS extension:', error);
    throw error;
  }
}

// Initialize database
async function initializeDatabase() {
  try {
    // Test database connection
    try {
      await sequelize.authenticate();
      console.log('Database connection has been established successfully.');
    } catch (authError) {
      console.error('Database authentication failed:', authError.message);
      if (authError.parent) {
        console.error('Underlying error:', authError.parent.message);
      }
      throw new Error('Unable to connect to the database. Check your connection details.');
    }
    
    // Enable PostGIS extension
    await enablePostGIS();
    
    // Sync all models
    // Note: force: true will drop tables if they exist
    // Use with caution in production
    const force = process.env.NODE_ENV === 'development' && process.env.DB_FORCE_SYNC === 'true';
    try {
      await sequelize.sync({ force });
      console.log('Database synchronized successfully');
    } catch (syncError) {
      console.error('Failed to sync database models:', syncError.message);
      throw syncError;
    }
    
    return true;
  } catch (error) {
    console.error('Unable to initialize database:', error.message);
    return false;
  }
}

module.exports = {
  initializeDatabase
}; 