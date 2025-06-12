const { expect } = require('chai');
const { Sequelize } = require('sequelize');
const path = require('path');
const models = require('../../src/models'); // Import models index

describe('Database Schema and Associations', function() {
  let sequelize;
  let queryInterface;

  before(async function() {
    // This test requires a running database configured via environment variables
    if (!process.env.DB_NAME || !process.env.DB_USER) {
      console.error('Database environment variables (DB_NAME, DB_USER, etc.) are not set. Skipping schema tests.');
      this.skip();
      return;
    }

    // Use the sequelize instance from our models index to ensure we're testing the same connection
    sequelize = models.sequelize; 
    queryInterface = sequelize.getQueryInterface();

    try {
      await sequelize.authenticate();
      console.log('Database connection has been established successfully for testing.');
    } catch (error) {
      console.error('Unable to connect to the database for testing:', error);
      this.skip(); // Skip tests if DB connection fails
    }
  });

  // No need for an after hook to close sequelize, as it's managed by the app/models index.

  it('should have an "Erratics" table with the correct columns', async function() {
    const tableInfo = await queryInterface.describeTable('Erratics');
    
    // Check for essential fields
    expect(tableInfo).to.have.property('id');
    expect(tableInfo).to.have.property('name');
    expect(tableInfo).to.have.property('location');
    expect(tableInfo.location.type).to.equal('GEOMETRY');
    expect(tableInfo).to.have.property('description');
    expect(tableInfo).to.have.property('rock_type');
    expect(tableInfo).to.have.property('elevation');

    // Ensure analysis fields are NOT here
    expect(tableInfo).to.not.have.property('usage_type');
    expect(tableInfo).to.not.have.property('nearest_water_body_dist');
    expect(tableInfo).to.not.have.property('vector_embedding');
  });

  it('should have an "ErraticAnalyses" table with the correct columns', async function() {
    const tableInfo = await queryInterface.describeTable('ErraticAnalyses');
    
    // Check the primary/foreign key
    expect(tableInfo).to.have.property('erraticId');
    expect(tableInfo.erraticId.primaryKey).to.be.true;

    // Check for a representative sample of analysis fields
    expect(tableInfo).to.have.property('usage_type');
    expect(tableInfo.usage_type.type).to.equal('ARRAY');
    
    expect(tableInfo).to.have.property('has_inscriptions');
    expect(tableInfo.has_inscriptions.type).to.equal('BOOLEAN');

    expect(tableInfo).to.have.property('nearest_water_body_dist');
    expect(tableInfo.nearest_water_body_dist.type).to.be.oneOf(['REAL', 'DOUBLE PRECISION']); // float can be REAL or DOUBLE

    expect(tableInfo).to.have.property('vector_embedding');
    // In raw sequelize describeTable, custom types like vector appear as their underlying type or a user-defined type name
    // This check is intentionally broad.
    expect(tableInfo.vector_embedding.type).to.exist; 
    
    expect(tableInfo).to.have.property('ruggedness_tri');
  });

  it('should have a "Users" table', async function() {
    const tables = await queryInterface.showAllTables();
    expect(tables).to.include('Users');
  });

  it('should have an "ErraticMedia" table', async function() {
    const tables = await queryInterface.showAllTables();
    expect(tables).to.include('ErraticMedia');
  });

  it('should have an "ErraticReferences" table', async function() {
    const tables = await queryInterface.showAllTables();
    expect(tables).to.include('ErraticReferences');
  });

  describe('Model Associations', function() {
    it('Erratic should have a one-to-one association with ErraticAnalysis', function() {
      const association = models.Erratic.associations.analysis;
      expect(association).to.exist;
      expect(association.associationType).to.equal('HasOne');
      expect(association.foreignKey).to.equal('erraticId');
    });

    it('ErraticAnalysis should belong to one Erratic', function() {
      const association = models.ErraticAnalysis.associations.erratic;
      expect(association).to.exist;
      expect(association.associationType).to.equal('BelongsTo');
      expect(association.foreignKey).to.equal('erraticId');
    });

    it('Erratic should have a one-to-many association with ErraticMedia', function() {
      const association = models.Erratic.associations.media;
      expect(association).to.exist;
      expect(association.associationType).to.equal('HasMany');
      expect(association.foreignKey).to.equal('erraticId');
    });

    it('Erratic should have a one-to-many association with ErraticReference', function() {
      const association = models.Erratic.associations.references;
      expect(association).to.exist;
      expect(association.associationType).to.equal('HasMany');
      expect(association.foreignKey).to.equal('erraticId');
    });
  });
}); 