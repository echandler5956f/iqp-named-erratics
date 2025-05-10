module.exports = (sequelize, DataTypes) => {
  const ErraticAnalysis = sequelize.define('ErraticAnalysis', {
    // Use erraticId as both Primary Key and Foreign Key for a true 1-to-1
    erraticId: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      references: {
        model: 'Erratics', // References the Erratics table
        key: 'id'
      },
      onDelete: 'CASCADE',
      onUpdate: 'CASCADE'
    },
    // Moved Analysis Fields
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
    // Proximity distances calculated by analysis scripts.
    // Some, like colonial settlements/roads, still depend on specific dataset acquisition.
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
    ruggedness_tri: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    terrain_landform: {
      type: DataTypes.STRING(100),
      allowNull: true
    },
    terrain_slope_position: {
      type: DataTypes.STRING(100),
      allowNull: true
    },
    vector_embedding: {
      type: 'VECTOR(384)', // Raw type for pgvector after migration
      allowNull: true
    }
    // Note: createdAt and updatedAt are added by default with timestamps: true
  }, {
    timestamps: true,
    // Optional: explicitly define table name if needed, though Sequelize usually infers correctly
    // tableName: 'erratic_analyses' 
  });

  ErraticAnalysis.associate = function(models) {
    ErraticAnalysis.belongsTo(models.Erratic, {
      foreignKey: 'erraticId',
      as: 'erratic' // Alias to access the associated Erratic from an Analysis record
    });
  };

  return ErraticAnalysis;
}; 