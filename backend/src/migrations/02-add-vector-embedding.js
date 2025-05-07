const { DataTypes } = require('sequelize');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables
const envPath = path.join(__dirname, '../../.env');
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
}

module.exports = {
  up: async (queryInterface, Sequelize) => {
    // Add vector_embedding_data column as fallback - this will always work
    console.log('Adding vector_embedding_data column as JSONB fallback...');
    await queryInterface.addColumn('Erratics', 'vector_embedding_data', {
      type: DataTypes.JSONB,
      allowNull: true,
      comment: 'JSONB storage for vector embeddings (fallback or when pgvector not available)'
    }).catch(err => {
      if (err.name === 'SequelizeUniqueConstraintError' || 
          (err.message && err.message.includes('already exists'))) {
        console.log('Column vector_embedding_data already exists, skipping...');
      } else {
        throw err;
      }
    });
    
    // Get database configuration from environment
    const dbHost = process.env.DB_HOST || 'localhost';
    const dbPort = process.env.DB_PORT || '5432';
    const dbName = process.env.DB_NAME || 'glacial_erratics';
    const dbUser = process.env.DB_USER || 'postgres';
    
    console.log(`Database configuration: ${dbName} on ${dbHost}:${dbPort}`);
    
    // Now try to add vector column if pgvector is available
    try {
      console.log('Checking if pgvector extension is available...');
      // First check if pgvector extension exists
      const [result] = await queryInterface.sequelize.query(`
        SELECT COUNT(*) AS count FROM pg_extension WHERE extname = 'vector';
      `, { type: Sequelize.QueryTypes.SELECT });
      
      const extensionExists = parseInt(result.count) > 0;
      
      if (!extensionExists) {
        console.log('pgvector extension not found in database. Attempting to create it...');
        
        try {
          // Try to create the extension directly
          await queryInterface.sequelize.query(`CREATE EXTENSION vector;`);
          console.log('Successfully created pgvector extension!');
        } catch (extError) {
          console.error('Could not create pgvector extension:', extError.message);
          console.log('Make sure pgvector is installed. Run: npm run pgvector:install');
          console.log('Then manually create the extension: psql -d ' + dbName + ' -c "CREATE EXTENSION vector;"');
          console.log('Skipping vector column creation, will use JSONB fallback instead.');
          return;
        }
      } else {
        console.log('pgvector extension already exists in database.');
      }
      
      // Extension exists, proceed with creating the column
      console.log('pgvector extension found, adding vector_embedding column...');
      
      await queryInterface.sequelize.query(`
        ALTER TABLE "Erratics" ADD COLUMN IF NOT EXISTS vector_embedding vector(1536);
      `);
      
      // Create index for vector search
      console.log('Creating vector index...');
      await queryInterface.sequelize.query(`
        CREATE INDEX IF NOT EXISTS erratics_vector_idx ON "Erratics" USING ivfflat (vector_embedding vector_l2_ops);
      `);
      
      console.log('Successfully added vector_embedding column and index!');
    } catch (error) {
      console.error('Error adding vector column:', error.message);
      console.log('Will use JSONB fallback column for vector data instead.');
    }
  },

  down: async (queryInterface, Sequelize) => {
    try {
      // Try to remove the vector extension features first
      try {
        await queryInterface.sequelize.query(`
          DROP INDEX IF EXISTS erratics_vector_idx;
        `);
        
        await queryInterface.sequelize.query(`
          ALTER TABLE "Erratics" DROP COLUMN IF EXISTS vector_embedding;
        `);
        console.log('Removed vector_embedding column and index');
      } catch (error) {
        console.log('No vector column to remove or error removing it:', error.message);
      }
      
      // Always try to remove the fallback column
      await queryInterface.removeColumn('Erratics', 'vector_embedding_data', { 
        ifExists: true 
      });
      console.log('Removed vector_embedding_data column');
    } catch (error) {
      console.error('Error in migration down:', error);
      throw error;
    }
  }
}; 