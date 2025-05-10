'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up (queryInterface, Sequelize) {
    await queryInterface.sequelize.transaction(async (transaction) => {
      // Ensure pgvector extension is available (though ideally created outside migration)
      // await queryInterface.sequelize.query('CREATE EXTENSION IF NOT EXISTS vector;', { transaction });

      // Change vector_embedding column type to VECTOR(384)
      // Note: Direct type change might not always be possible if data exists and is incompatible.
      // For safety, one might need to add new column, copy data, drop old, rename new.
      // However, if current JSONB stores an array of numbers, direct cast might work or be acceptable with data reload.
      // Here, we assume a direct change is intended, or data will be repopulated.
      await queryInterface.changeColumn('ErraticAnalyses', 'vector_embedding', {
        type: 'VECTOR(384)', // Raw SQL type for pgvector
        allowNull: true
      }, { transaction });

      // Remove the redundant vector_embedding_data column
      try {
        // Check if column exists before trying to remove it to make it more robust
        const tableDescription = await queryInterface.describeTable('ErraticAnalyses', { transaction });
        if (tableDescription.vector_embedding_data) {
          await queryInterface.removeColumn('ErraticAnalyses', 'vector_embedding_data', { transaction });
        } else {
          console.log("Column 'vector_embedding_data' not found in 'ErraticAnalyses', skipping removal.");
        }
      } catch (error) {
        // If describeTable fails or removeColumn fails, log and continue if acceptable
        console.warn("Could not remove 'vector_embedding_data'. It might have been removed already or an error occurred:", error.message);
      }
    });
  },

  async down (queryInterface, Sequelize) {
    // Revert changes
    await queryInterface.sequelize.transaction(async (transaction) => {
      // Change vector_embedding column type back to JSONB
      await queryInterface.changeColumn('ErraticAnalyses', 'vector_embedding', {
        type: Sequelize.JSONB,
        allowNull: true
      }, { transaction });

      // Re-add the vector_embedding_data column as JSONB
      // This makes the down migration fully revert the up migration.
      await queryInterface.addColumn('ErraticAnalyses', 'vector_embedding_data', {
        type: Sequelize.JSONB,
        allowNull: true,
        comment: 'Placeholder or raw data for embeddings (restored)'
      }, { transaction });
    });
  }
}; 