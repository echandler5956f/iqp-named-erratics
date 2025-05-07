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