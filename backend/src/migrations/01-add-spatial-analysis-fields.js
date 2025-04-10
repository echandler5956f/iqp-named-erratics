const { DataTypes } = require('sequelize');

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.addColumn('Erratics', 'usage_type', {
      type: DataTypes.ARRAY(DataTypes.STRING(100)),
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'cultural_significance_score', {
      type: DataTypes.INTEGER,
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'has_inscriptions', {
      type: DataTypes.BOOLEAN,
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'accessibility_score', {
      type: DataTypes.INTEGER,
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'size_category', {
      type: DataTypes.STRING(50),
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'nearest_water_body_dist', {
      type: DataTypes.FLOAT,
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'nearest_settlement_dist', {
      type: DataTypes.FLOAT,
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'elevation_category', {
      type: DataTypes.STRING(50),
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'geological_type', {
      type: DataTypes.STRING(100),
      allowNull: true
    });

    await queryInterface.addColumn('Erratics', 'estimated_displacement_dist', {
      type: DataTypes.FLOAT,
      allowNull: true
    });

    // Create indexes for efficient spatial queries
    await queryInterface.sequelize.query(
      'CREATE INDEX IF NOT EXISTS erratics_location_idx ON "Erratics" USING GIST(location);'
    );

    await queryInterface.sequelize.query(
      'CREATE INDEX IF NOT EXISTS erratics_usage_type_idx ON "Erratics" USING GIN(usage_type);'
    );

    // Note: For vector_embedding, we need to install the pgvector extension first
    // This is commented out and will be handled in a separate migration
    /* 
    await queryInterface.addColumn('Erratics', 'vector_embedding', {
      type: 'VECTOR(1536)',
      allowNull: true
    });

    await queryInterface.sequelize.query(
      'CREATE INDEX IF NOT EXISTS erratics_vector_idx ON "Erratics" USING ivfflat (vector_embedding vector_l2_ops);'
    );
    */
  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.removeColumn('Erratics', 'usage_type');
    await queryInterface.removeColumn('Erratics', 'cultural_significance_score');
    await queryInterface.removeColumn('Erratics', 'has_inscriptions');
    await queryInterface.removeColumn('Erratics', 'accessibility_score');
    await queryInterface.removeColumn('Erratics', 'size_category');
    await queryInterface.removeColumn('Erratics', 'nearest_water_body_dist');
    await queryInterface.removeColumn('Erratics', 'nearest_settlement_dist');
    await queryInterface.removeColumn('Erratics', 'elevation_category');
    await queryInterface.removeColumn('Erratics', 'geological_type');
    await queryInterface.removeColumn('Erratics', 'estimated_displacement_dist');
    // await queryInterface.removeColumn('Erratics', 'vector_embedding');

    // Remove indexes
    await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratics_location_idx;');
    await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratics_usage_type_idx;');
    // await queryInterface.sequelize.query('DROP INDEX IF EXISTS erratics_vector_idx;');
  }
}; 