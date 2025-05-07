'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    // This migration reverts changes made incorrectly to the Erratics table
    // by 01-add-spatial-analysis-fields.js and 02-add-vector-embedding.js.
    // The 'up' function intentionally does nothing, as we only want to run the 'down'.
    // Alternatively, the logic from the 'down' function below could be placed here
    // if we intended this migration to run forward to perform the cleanup.
    // However, placing it in 'down' aligns with the idea of reverting the previous migrations' state.
    console.log("Migration 03: No forward action needed, revert logic is in 'down' function.");
  },

  down: async (queryInterface, Sequelize) => {
    console.log("Migration 03 Reverting: Removing analysis columns and indexes from Erratics table...");

    // Remove columns added by 01-add-spatial-analysis-fields.js
    const columnsToRemoveFrom01 = [
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
    for (const column of columnsToRemoveFrom01) {
        try {
            console.log(`Removing column ${column} from Erratics...`);
            await queryInterface.removeColumn('Erratics', column);
        } catch (error) {
            // Log error if column doesn't exist, but continue
            console.warn(`Could not remove column ${column} (may not exist): ${error.message}`);
        }
    }

    // Remove columns/indexes added by 02-add-vector-embedding.js
     try {
        console.log("Removing vector_embedding_data column from Erratics...");
        await queryInterface.removeColumn('Erratics', 'vector_embedding_data');
    } catch (error) {
        console.warn(`Could not remove column vector_embedding_data (may not exist): ${error.message}`);
    }
    try {
      console.log("Removing vector index erratics_vector_idx if exists...");
      await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratics_vector_idx;');
      console.log("Removing vector_embedding column from Erratics...");
      await queryInterface.sequelize.query('ALTER TABLE "Erratics" DROP COLUMN IF EXISTS vector_embedding;');
    } catch (error) {
        console.warn(`Could not remove vector column/index (may not exist): ${error.message}`);
    }

    // Remove indexes added by 01-add-spatial-analysis-fields.js
    try {
        console.log("Removing index erratics_location_idx...");
        await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratics_location_idx;');
    } catch(error) {
         console.warn(`Could not remove index erratics_location_idx: ${error.message}`);
    }
     try {
        console.log("Removing index erratics_usage_type_idx...");
        await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratics_usage_type_idx;');
     } catch (error) {
         console.warn(`Could not remove index erratics_usage_type_idx: ${error.message}`);
     }

    console.log("Migration 03 Reverting: Finished removing analysis columns/indexes from Erratics.");
  }
}; 