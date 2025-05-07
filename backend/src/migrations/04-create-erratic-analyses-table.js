'use strict';

const { DataTypes } = require('sequelize');

module.exports = {
  up: async (queryInterface, Sequelize) => {
    console.log("Migration 04 Up: Creating ErraticAnalyses table...");
    await queryInterface.createTable('ErraticAnalyses', {
      erraticId: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onDelete: 'CASCADE',
        onUpdate: 'CASCADE'
      },
      usage_type: {
        type: DataTypes.ARRAY(DataTypes.STRING(100)),
        allowNull: true
      },
      cultural_significance_score: {
        type: DataTypes.INTEGER,
        allowNull: true
      },
      has_inscriptions: {
        type: DataTypes.BOOLEAN,
        allowNull: true
      },
      accessibility_score: {
        type: DataTypes.INTEGER,
        allowNull: true
      },
      size_category: {
        type: DataTypes.STRING(50),
        allowNull: true
      },
      nearest_water_body_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      nearest_settlement_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      nearest_colonial_settlement_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      nearest_road_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      nearest_colonial_road_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      nearest_native_territory_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      elevation_category: {
        type: DataTypes.STRING(50),
        allowNull: true
      },
      geological_type: {
        type: DataTypes.STRING(100),
        allowNull: true
      },
      estimated_displacement_dist: {
        type: DataTypes.FLOAT,
        allowNull: true
      },
      vector_embedding: {
        // Conditionally added below if pgvector exists
        type: DataTypes.JSONB, // Default/Fallback type
        allowNull: true
      },
      vector_embedding_data: {
        type: DataTypes.JSONB,
        allowNull: true,
        comment: 'Placeholder or raw data for embeddings'
      },
      createdAt: {
        allowNull: false,
        type: DataTypes.DATE,
        defaultValue: Sequelize.literal('CURRENT_TIMESTAMP')
      },
      updatedAt: {
        allowNull: false,
        type: DataTypes.DATE,
        defaultValue: Sequelize.literal('CURRENT_TIMESTAMP')
      }
    });

    console.log("ErraticAnalyses table created.");

    // Conditionally add VECTOR column and index if pgvector exists
    try {
        const [results] = await queryInterface.sequelize.query(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector';"
        );
        if (results && results.length > 0) {
            console.log("pgvector extension found. Updating vector_embedding column type to VECTOR(384)...");
            // Note: Embedding size matches 'all-MiniLM-L6-v2' (384 dimensions)
            await queryInterface.sequelize.query(
              `ALTER TABLE "ErraticAnalyses" ALTER COLUMN vector_embedding TYPE vector(384) USING vector_embedding::text::vector;`
            );
            console.log("Creating index on vector_embedding...");
            // Use cosine distance as it often works well with sentence embeddings
            await queryInterface.sequelize.query(
              `CREATE INDEX IF NOT EXISTS erratic_analyses_vector_idx ON "ErraticAnalyses" USING ivfflat (vector_embedding vector_cosine_ops);`
            );
            console.log("vector_embedding column updated and indexed.");
        } else {
             console.log("pgvector extension not found. vector_embedding column remains JSONB.");
        }
    } catch (error) {
        console.warn(`Error checking/updating for pgvector: ${error.message}. vector_embedding column remains JSONB.`);
    }
    console.log("Migration 04 Up: Finished.");
  },

  down: async (queryInterface, Sequelize) => {
    console.log("Migration 04 Down: Dropping ErraticAnalyses table...");
    // Drop index first if it exists
    await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratic_analyses_vector_idx;');
    // Drop the table
    await queryInterface.dropTable('ErraticAnalyses');
    console.log("ErraticAnalyses table dropped.");
  }
}; 