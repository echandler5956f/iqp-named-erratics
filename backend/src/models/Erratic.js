module.exports = (sequelize, DataTypes) => {
  const Erratic = sequelize.define('Erratic', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
      validate: {
        notEmpty: {
          args: true,
          msg: 'Erratic name cannot be empty.'
        },
        len: {
          args: [1, 255],
          msg: 'Erratic name must be between 1 and 255 characters.'
        }
      }
    },
    location: {
      type: DataTypes.GEOMETRY('POINT'),
      allowNull: false // Validated at service level for specific lat/lon presence
    },
    elevation: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: {
          args: true,
          msg: 'Elevation must be a valid number.'
        }
        // Add min/max here if applicable, e.g., min: -500, max: 9000
      }
    },
    size_meters: {
      type: DataTypes.FLOAT,
      allowNull: true,
      validate: {
        isFloat: {
          args: true,
          msg: 'Size must be a valid number.'
        },
        min: {
          args: [0],
          msg: 'Size cannot be negative.'
        }
      }
    },
    rock_type: {
      type: DataTypes.STRING(100),
      allowNull: true,
      validate: {
        len: {
          args: [0, 100],
          msg: 'Rock type must be 100 characters or less.'
        }
      }
    },
    estimated_age: {
      type: DataTypes.STRING(100),
      allowNull: true,
      validate: {
        len: {
          args: [0, 100],
          msg: 'Estimated age must be 100 characters or less.'
        }
      }
    },
    discovery_date: {
      type: DataTypes.DATE,
      allowNull: true,
      validate: {
        isDate: {
          args: true,
          msg: 'Discovery date must be a valid date.'
        }
      }
    },
    description: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    cultural_significance: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    historical_notes: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    image_url: {
      type: DataTypes.TEXT,
      allowNull: true,
      validate: {
        isUrlOrEmpty(value) {
          if (value === null || value === '' || value === undefined) {
            return; // Allow empty or null
          }
          // Basic URL validation, since validator.js's isUrl is too strict
          // and we can't easily pass options to it in this context.
          if (typeof value === 'string' && (value.startsWith('http://') || value.startsWith('https://'))) {
            return;
          }
          throw new Error('Image URL must be a valid URL.');
        }
      }
    }
  }, {
    timestamps: true
  });
  
  Erratic.associate = function(models) {
    Erratic.hasMany(models.ErraticMedia, {
      foreignKey: 'erraticId',
      as: 'media'
    });
    
    Erratic.hasMany(models.ErraticReference, {
      foreignKey: 'erraticId',
      as: 'references'
    });

    Erratic.hasOne(models.ErraticAnalysis, {
      foreignKey: 'erraticId',
      as: 'analysis'
    });
  };
  
  return Erratic;
}; 