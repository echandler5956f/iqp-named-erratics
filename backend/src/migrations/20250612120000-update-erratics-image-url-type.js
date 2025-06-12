'use strict';

/**
 * Migration: Updates the data type of the image_url column in the Erratics table.
 * 
 * This migration changes the `image_url` column from STRING (VARCHAR(255)) to TEXT
 * to accommodate longer URLs found in the source data.
 */
module.exports = {
  async up(queryInterface, Sequelize) {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      console.log('Starting migration: Change Erratics.image_url to TEXT');
      
      await queryInterface.changeColumn(
        'Erratics',
        'image_url',
        {
          type: Sequelize.TEXT,
          allowNull: true,
          comment: 'Stores the full URL of the erratic image, allowing for longer strings.'
        },
        { transaction }
      );
      
      await transaction.commit();
      console.log('Migration completed successfully: Changed Erratics.image_url to TEXT');
    } catch (error) {
      await transaction.rollback();
      console.error('Migration failed:', error);
      return Promise.reject(error);
    }
  },

  async down(queryInterface, Sequelize) {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      console.log('Starting rollback: Reverting Erratics.image_url to STRING');
      
      await queryInterface.changeColumn(
        'Erratics',
        'image_url',
        {
          type: Sequelize.STRING, // Reverts back to the original VARCHAR(255)
          allowNull: true
        },
        { transaction }
      );
      
      await transaction.commit();
      console.log('Rollback completed successfully: Reverted Erratics.image_url to STRING');
    } catch (error) {
      await transaction.rollback();
      console.error('Rollback failed:', error);
      return Promise.reject(error);
    }
  }
}; 