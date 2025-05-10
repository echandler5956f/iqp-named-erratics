'use strict';

/**
 * Migration to add terrain analysis fields to the ErraticAnalyses table
 * These fields store the results of the terrain analysis pipeline from DEM data
 */
module.exports = {
  async up(queryInterface, Sequelize) {
    try {
      console.log('Starting migration to add terrain analysis fields to ErraticAnalyses');
      
      // Check if the columns already exist to avoid errors
      const tableInfo = await queryInterface.describeTable('ErraticAnalyses');
      
      // Add ruggedness_tri if it doesn't exist
      if (!tableInfo.ruggedness_tri) {
        console.log('Adding ruggedness_tri column...');
        await queryInterface.addColumn(
          'ErraticAnalyses',
          'ruggedness_tri',
          {
            type: Sequelize.FLOAT,
            allowNull: true,
            comment: 'Terrain Ruggedness Index (TRI) calculated from DEM'
          }
        );
      } else {
        console.log('Column ruggedness_tri already exists, skipping...');
      }
      
      // Add terrain_landform if it doesn't exist
      if (!tableInfo.terrain_landform) {
        console.log('Adding terrain_landform column...');
        await queryInterface.addColumn(
          'ErraticAnalyses',
          'terrain_landform',
          {
            type: Sequelize.STRING(100),
            allowNull: true,
            comment: 'Classification of landform type based on terrain analysis'
          }
        );
      } else {
        console.log('Column terrain_landform already exists, skipping...');
      }
      
      // Add terrain_slope_position if it doesn't exist
      if (!tableInfo.terrain_slope_position) {
        console.log('Adding terrain_slope_position column...');
        await queryInterface.addColumn(
          'ErraticAnalyses',
          'terrain_slope_position',
          {
            type: Sequelize.STRING(100),
            allowNull: true,
            comment: 'Slope position classification from terrain analysis'
          }
        );
      } else {
        console.log('Column terrain_slope_position already exists, skipping...');
      }
      
      // Ensure all core proximity columns that might be missing due to database schema mismatch
      // This addresses the issue observed in the console error
      const coreSpatialColumns = [
        'nearest_colonial_settlement_dist',
        'nearest_road_dist',
        'nearest_colonial_road_dist',
        'nearest_native_territory_dist'
      ];
      
      for (const column of coreSpatialColumns) {
        if (!tableInfo[column]) {
          console.log(`Adding missing spatial analysis column: ${column}...`);
          await queryInterface.addColumn(
            'ErraticAnalyses',
            column,
            {
              type: Sequelize.FLOAT,
              allowNull: true,
              comment: `Distance to nearest ${column.replace('nearest_', '').replace('_dist', '')}`
            }
          );
        }
      }
      
      console.log('Migration completed successfully');
      return Promise.resolve();
    } catch (error) {
      console.error('Migration failed:', error);
      return Promise.reject(error);
    }
  },

  async down(queryInterface, Sequelize) {
    try {
      console.log('Starting rollback of terrain analysis fields from ErraticAnalyses');
      
      // Check if the columns exist before trying to remove them
      const tableInfo = await queryInterface.describeTable('ErraticAnalyses');
      
      // Remove the terrain analysis columns
      const columnsToRemove = [
        'ruggedness_tri',
        'terrain_landform',
        'terrain_slope_position'
      ];
      
      for (const column of columnsToRemove) {
        if (tableInfo[column]) {
          console.log(`Removing column: ${column}...`);
          await queryInterface.removeColumn('ErraticAnalyses', column);
        } else {
          console.log(`Column ${column} doesn't exist, skipping removal...`);
        }
      }
      
      console.log('Rollback completed successfully');
      return Promise.resolve();
    } catch (error) {
      console.error('Rollback failed:', error);
      return Promise.reject(error);
    }
  }
}; 