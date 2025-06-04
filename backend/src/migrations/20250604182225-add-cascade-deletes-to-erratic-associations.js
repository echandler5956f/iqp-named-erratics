'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up (queryInterface, Sequelize) {
    await queryInterface.sequelize.transaction(async (transaction) => {
      // Add/Update onDelete and onUpdate for ErraticMedia.erraticId
      // We need to remove the old constraint and add a new one if changing cascade rules on an existing FK.
      // However, a simpler approach that often works is to redefine the column with the new rules.
      // Sequelize might handle the drop/re-add of constraint internally.
      // If this fails, a more complex removeConstraint/addConstraint would be needed.
      await queryInterface.changeColumn('ErraticMedia', 'erraticId', {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE'
      }, { transaction });

      // Add/Update onDelete and onUpdate for ErraticReferences.erraticId
      await queryInterface.changeColumn('ErraticReferences', 'erraticId', {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE'
      }, { transaction });
    });
  },

  async down (queryInterface, Sequelize) {
    await queryInterface.sequelize.transaction(async (transaction) => {
      // Revert ErraticMedia.erraticId to default (usually NO ACTION or RESTRICT if not specified)
      // To be safe, we specify NO ACTION for rollback.
      await queryInterface.changeColumn('ErraticMedia', 'erraticId', {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onUpdate: 'NO ACTION', // Or whatever the previous default was
        onDelete: 'NO ACTION'  // Or whatever the previous default was
      }, { transaction });

      // Revert ErraticReferences.erraticId
      await queryInterface.changeColumn('ErraticReferences', 'erraticId', {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onUpdate: 'NO ACTION',
        onDelete: 'NO ACTION' 
      }, { transaction });
    });
  }
}; 