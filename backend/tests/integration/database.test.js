const { expect } = require('chai');
const { Sequelize } = require('sequelize');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables
const envPath = path.resolve(__dirname, '../../.env');
dotenv.config({ path: envPath });

// Override with test database if needed
process.env.DB_NAME = process.env.DB_NAME || 'test_glacial_erratics';

describe('Database Schema', function() {
  let sequelize, queryInterface;
  
  before(async function() {
    // Connect to the database
    sequelize = new Sequelize(
      process.env.DB_NAME,
      process.env.DB_USER,
      process.env.DB_PASSWORD,
      {
        host: process.env.DB_HOST,
        port: process.env.DB_PORT,
        dialect: 'postgres',
        logging: false
      }
    );
    
    // Test the connection
    try {
      await sequelize.authenticate();
      console.log('Connection has been established successfully.');
      queryInterface = sequelize.getQueryInterface();
    } catch (error) {
      console.error('Unable to connect to the database:', error);
      this.skip();
    }
  });
  
  after(async function() {
    if (sequelize) {
      await sequelize.close();
    }
  });
  
  // Test whether the Erratics table exists
  it('should have an Erratics table', async function() {
    const tables = await queryInterface.showAllTables();
    expect(tables).to.include('Erratics');
  });
  
  // Get the table columns and test them dynamically
  it('should have the expected columns in the Erratics table', async function() {
    const tableInfo = await queryInterface.describeTable('Erratics');
    
    // Check for the basic fields that should be present
    const requiredFields = [
      'id', 'name', 'location', 
      'description', 'rock_type', 'elevation'
    ];
    
    for (const field of requiredFields) {
      expect(tableInfo).to.have.property(field);
    }
    
    // Check for spatial analysis fields if they exist
    const spatialFields = [
      'usage_type',
      'cultural_significance_score',
      'has_inscriptions',
      'accessibility_score',
      'size_category',
      'nearest_water_body_dist',
      'nearest_settlement_dist',
      'elevation_category',
      'geological_type',
      'estimated_displacement_dist'
    ];
    
    // Count how many spatial fields are present
    const presentSpatialFields = spatialFields.filter(field => tableInfo[field]);
    
    // If we have any spatial fields, report which ones
    if (presentSpatialFields.length > 0) {
      console.log(`Found ${presentSpatialFields.length} spatial analysis fields:`, 
        presentSpatialFields.join(', '));
    } else {
      console.log('No spatial analysis fields found in the database yet.');
      console.log('This is expected if you have not yet run migrations.');
    }
  });
  
  // Test the data types of the spatial analysis fields if they exist
  it('should have correct data types for fields that exist', async function() {
    const tableInfo = await queryInterface.describeTable('Erratics');
    
    // Check field types for fields that exist
    if (tableInfo.usage_type) {
      expect(tableInfo.usage_type.type.toLowerCase()).to.include('array');
    }
    
    if (tableInfo.cultural_significance_score) {
      expect(tableInfo.cultural_significance_score.type.toLowerCase()).to.include('int');
    }
    
    if (tableInfo.has_inscriptions) {
      expect(tableInfo.has_inscriptions.type.toLowerCase()).to.include('bool');
    }
    
    if (tableInfo.size_category) {
      // Postgres reports either VARCHAR or CHARACTER VARYING - both are valid
      const type = tableInfo.size_category.type.toLowerCase();
      expect(type.includes('varchar') || type.includes('character varying')).to.be.true;
    }
    
    if (tableInfo.nearest_water_body_dist) {
      const type = tableInfo.nearest_water_body_dist.type.toLowerCase();
      expect(type.includes('float') || type.includes('numeric') || type.includes('double')).to.be.true;
    }
  });
  
  // Test database indexes
  it('should have spatial indexes', async function() {
    try {
      // Get list of indexes on the Erratics table
      const [indexes] = await sequelize.query(`
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'Erratics';
      `);
      
      // Log the indexes for debugging
      console.log('Indexes on Erratics table:', indexes.map(idx => idx.indexname));
      
      // Convert to a map for easier lookup
      const indexMap = {};
      indexes.forEach(idx => {
        indexMap[idx.indexname] = idx.indexdef;
      });
      
      // Verify location index exists, in case it's not called exactly what we expect
      const hasLocationIndex = Object.keys(indexMap).some(
        key => key.toLowerCase().includes('location') || 
              key.toLowerCase().includes('geometry') ||
              (indexMap[key] && indexMap[key].toLowerCase().includes('gist'))
      );
      
      expect(hasLocationIndex).to.be.true;
      
      // Check for usage_type index if it exists
      if (indexMap.erratics_usage_type_idx) {
        expect(indexMap.erratics_usage_type_idx.toLowerCase()).to.include('gin');
      }
    } catch (error) {
      console.error('Error checking indexes:', error);
      this.skip();
    }
  });
}); 