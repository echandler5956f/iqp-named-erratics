'use strict';

/**
 * Migration Template: Use this as a starting point for future migrations
 * 
 * This template demonstrates best practices for creating migrations:
 * 1. Always check if columns exist before attempting to add them
 * 2. Include thorough comments and documentation within the migration
 * 3. Handle errors gracefully and provide descriptive console output
 * 4. Support both up (apply) and down (revert) operations
 * 5. Use explicit sequelize data types and constraints
 */
module.exports = {
  async up(queryInterface, Sequelize) {
    try {
      console.log('Starting migration: Template for future migrations');
      
      // Always check table info first to ensure safe operations
      // const tableInfo = await queryInterface.describeTable('YourTableName');
      
      // EXAMPLE 1: Adding a new column if it doesn't exist
      // if (!tableInfo.example_new_column) {
      //   console.log('Adding example_new_column...');
      //   await queryInterface.addColumn(
      //     'YourTableName',
      //     'example_new_column',
      //     {
      //       type: Sequelize.STRING(100),
      //       allowNull: true,
      //       comment: 'Example column to demonstrate migration pattern'
      //     }
      //   );
      // }
      
      // EXAMPLE 2: Modifying an existing column (only if it exists)
      // if (tableInfo.existing_column) {
      //   console.log('Modifying existing_column...');
      //   await queryInterface.changeColumn(
      //     'YourTableName',
      //     'existing_column',
      //     {
      //       type: Sequelize.STRING(200), // Change type/length
      //       allowNull: false, // Change constraint
      //       defaultValue: 'default_value', // Add default
      //       comment: 'Updated column definition'
      //     }
      //   );
      // }
      
      // EXAMPLE 3: Adding an index (safely check if it exists first)
      // console.log('Adding index on example_new_column if needed...');
      // In practice, you would need to check if the index exists first
      // This is a simplified example
      // await queryInterface.addIndex(
      //   'YourTableName',
      //   ['example_new_column'],
      //   {
      //     name: 'idx_example_new_column',
      //     unique: false,
      //     // For a spatial index, you might use:
      //     // using: 'GIST',
      //     // where: Sequelize.literal('example_new_column IS NOT NULL')
      //   }
      // ).catch(error => {
      //   // Catch and handle the case where the index already exists
      //   console.log('Index already exists or could not be created:', error.message);
      // });
      
      console.log('Migration completed successfully');
      return Promise.resolve();
    } catch (error) {
      console.error('Migration failed:', error);
      return Promise.reject(error);
    }
  },

  async down(queryInterface, Sequelize) {
    try {
      console.log('Starting rollback of migration: Template for future migrations');
      
      // EXAMPLE 1: Removing a column (if it exists)
      // try {
      //   const tableInfo = await queryInterface.describeTable('YourTableName');
      //   if (tableInfo.example_new_column) {
      //     console.log('Removing example_new_column...');
      //     await queryInterface.removeColumn('YourTableName', 'example_new_column');
      //   }
      // } catch (error) {
      //   console.error('Error checking for or removing example_new_column:', error.message);
      // }
      
      // EXAMPLE 2: Restoring an existing column to its previous state (if it exists)
      // try {
      //   const tableInfo = await queryInterface.describeTable('YourTableName');
      //   if (tableInfo.existing_column) {
      //     console.log('Restoring existing_column to previous state...');
      //     await queryInterface.changeColumn(
      //       'YourTableName',
      //       'existing_column',
      //       {
      //         type: Sequelize.STRING(100), // Original type/length
      //         allowNull: true, // Original constraint
      //         defaultValue: null, // Original default
      //         comment: 'Original column definition'
      //       }
      //     );
      //   }
      // } catch (error) {
      //   console.error('Error restoring existing_column:', error.message);
      // }
      
      // EXAMPLE 3: Removing an index (if it exists)
      // console.log('Removing index on example_new_column if it exists...');
      // await queryInterface.removeIndex(
      //   'YourTableName',
      //   'idx_example_new_column'
      // ).catch(error => {
      //   // Catch and handle the case where the index doesn't exist
      //   console.log('Index does not exist or could not be removed:', error.message);
      // });
      
      console.log('Rollback completed successfully');
      return Promise.resolve();
    } catch (error) {
      console.error('Rollback failed:', error);
      return Promise.reject(error);
    }
  }
}; 