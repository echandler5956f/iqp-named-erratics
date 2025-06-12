module.exports = (sequelize, DataTypes) => {
  const ErraticAnalysis = sequelize.define('ErraticAnalysis', {
    // Use erraticId as both Primary Key and Foreign Key for a true 1-to-1
    erraticId: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      allowNull: false,
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
      allowNull: true,
      validate: {
        isInt: {
          args: true,
          msg: 'Cultural significance score must be an integer.'
        },
        min: {
          args: [0],
          msg: 'Cultural significance score cannot be negative.'
        },
        max: {
          args: [10], // Assuming a 0-10 scale
          msg: 'Cultural significance score cannot exceed 10.'
        }
      }
    },
    has_inscriptions: {
      type: DataTypes.BOOLEAN,
      allowNull: true
    },
    accessibility_score: {
      type: DataTypes.INTEGER,
      allowNull: true,
      validate: {
        isInt: {
          args: true,
          msg: 'Accessibility score must be an integer.'
        },
        min: {
          args: [1],
          msg: 'Accessibility score must be at least 1.'
        },
        max: {
          args: [10], // 1-10 scale
          msg: 'Accessibility score cannot exceed 10.'
        }
      }
    },
    size_category: {
      type: DataTypes.STRING(50),
      allowNull: true,
      validate: {
        len: {
          args: [0, 50],
          msg: 'Size category must be 50 characters or less.'
        }
      }
    },
    nearest_water_body_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance must be a number.' },
        min: { args: [0], msg: 'Distance cannot be negative.' }
      }
    },
    nearest_settlement_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance must be a number.' },
        min: { args: [0], msg: 'Distance cannot be negative.' }
      }
    },
    nearest_road_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance must be a number.' },
        min: { args: [0], msg: 'Distance cannot be negative.' }
      }
    },
    nearest_native_territory_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance must be a number.' },
        min: { args: [0], msg: 'Distance cannot be negative.' }
      }
    },
    nearest_natd_road_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance to NATD road must be a number.' },
        min: { args: [0], msg: 'Distance to NATD road cannot be negative.' }
      }
    },
    nearest_forest_trail_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance to forest trail must be a number.' },
        min: { args: [0], msg: 'Distance to forest trail cannot be negative.' }
      }
    },
    elevation_category: {
      type: DataTypes.STRING(50),
      allowNull: true,
      validate: {
        len: {
          args: [0, 50],
          msg: 'Elevation category must be 50 characters or less.'
        }
      }
    },
    geological_type: {
      type: DataTypes.STRING(100),
      allowNull: true,
      validate: {
        len: {
          args: [0, 100],
          msg: 'Geological type must be 100 characters or less.'
        }
      }
    },
    estimated_displacement_dist: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Distance must be a number.' },
        min: { args: [0], msg: 'Distance cannot be negative.' }
      }
    },
    ruggedness_tri: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: { args: true, msg: 'Ruggedness (TRI) must be a number.' }
      }
    },
    terrain_landform: {
      type: DataTypes.STRING(100),
      allowNull: true,
      validate: {
        len: {
          args: [0, 100],
          msg: 'Terrain landform must be 100 characters or less.'
        }
      }
    },
    terrain_slope_position: {
      type: DataTypes.STRING(100),
      allowNull: true,
      validate: {
        len: {
          args: [0, 100],
          msg: 'Terrain slope position must be 100 characters or less.'
        }
      }
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