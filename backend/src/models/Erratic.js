module.exports = (sequelize, DataTypes) => {
  const Erratic = sequelize.define('Erratic', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    location: {
      type: DataTypes.GEOMETRY('POINT'),
      allowNull: false
    },
    elevation: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    size_meters: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    rock_type: {
      type: DataTypes.STRING(100),
      allowNull: true
    },
    estimated_age: {
      type: DataTypes.STRING(100),
      allowNull: true
    },
    discovery_date: {
      type: DataTypes.DATE,
      allowNull: true
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
      type: DataTypes.STRING,
      allowNull: true
    },
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
  };
  
  return Erratic;
}; 