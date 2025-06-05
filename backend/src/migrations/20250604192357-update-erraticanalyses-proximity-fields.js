'use strict';

/**
 * Migration: Updates ErraticAnalyses table for new proximity fields.
 * - Removes nearest_colonial_settlement_dist and nearest_colonial_road_dist.
 * - Adds nearest_natd_road_dist and nearest_forest_trail_dist.
 */
module.exports = {
  async up(queryInterface, Sequelize) {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      console.log('Starting migration: Update ErraticAnalyses proximity fields');

      // Remove old columns
      console.log('Removing column: nearest_colonial_settlement_dist from ErraticAnalyses');
      await queryInterface.removeColumn('ErraticAnalyses', 'nearest_colonial_settlement_dist', { transaction });
      
      console.log('Removing column: nearest_colonial_road_dist from ErraticAnalyses');
      await queryInterface.removeColumn('ErraticAnalyses', 'nearest_colonial_road_dist', { transaction });

      // Add new columns
      console.log('Adding column: nearest_natd_road_dist to ErraticAnalyses');
      await queryInterface.addColumn('ErraticAnalyses', 'nearest_natd_road_dist', {
        type: Sequelize.FLOAT,
        allowNull: true,
        comment: 'Distance to nearest NATD road in meters'
      }, { transaction });

      console.log('Adding column: nearest_forest_trail_dist to ErraticAnalyses');
      await queryInterface.addColumn('ErraticAnalyses', 'nearest_forest_trail_dist', {
        type: Sequelize.FLOAT,
        allowNull: true,
        comment: 'Distance to nearest forest trail in meters'
      }, { transaction });

      await transaction.commit();
      console.log('Migration completed successfully: Updated ErraticAnalyses proximity fields');
    } catch (error) {
      await transaction.rollback();
      console.error('Migration failed:', error);
      return Promise.reject(error);
    }
  },

  async down(queryInterface, Sequelize) {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      console.log('Starting rollback of migration: Update ErraticAnalyses proximity fields');

      // Remove new columns
      console.log('Removing column: nearest_natd_road_dist from ErraticAnalyses');
      await queryInterface.removeColumn('ErraticAnalyses', 'nearest_natd_road_dist', { transaction });

      console.log('Removing column: nearest_forest_trail_dist from ErraticAnalyses');
      await queryInterface.removeColumn('ErraticAnalyses', 'nearest_forest_trail_dist', { transaction });

      // Add back old columns
      console.log('Adding column: nearest_colonial_settlement_dist back to ErraticAnalyses');
      await queryInterface.addColumn('ErraticAnalyses', 'nearest_colonial_settlement_dist', {
        type: Sequelize.FLOAT,
        allowNull: true,
        comment: 'Distance to nearest historical colonial settlement in meters (deprecated)'
      }, { transaction });
      
      console.log('Adding column: nearest_colonial_road_dist back to ErraticAnalyses');
      await queryInterface.addColumn('ErraticAnalyses', 'nearest_colonial_road_dist', {
        type: Sequelize.FLOAT,
        allowNull: true,
        comment: 'Distance to nearest historical colonial road in meters (deprecated)'
      }, { transaction });

      await transaction.commit();
      console.log('Rollback completed successfully: Reverted ErraticAnalyses proximity fields');
    } catch (error) {
      await transaction.rollback();
      console.error('Rollback failed:', error);
      return Promise.reject(error);
    }
  }
}; 