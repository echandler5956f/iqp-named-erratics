module.exports = (sequelize, DataTypes) => {
  const ErraticReference = sequelize.define('ErraticReference', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    erraticId: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: {
        model: 'Erratics',
        key: 'id'
      }
    },
    reference_type: {
      type: DataTypes.ENUM('article', 'book', 'paper', 'website', 'other'),
      allowNull: false
    },
    title: {
      type: DataTypes.STRING,
      allowNull: false
    },
    authors: {
      type: DataTypes.STRING,
      allowNull: true
    },
    publication: {
      type: DataTypes.STRING,
      allowNull: true
    },
    year: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    url: {
      type: DataTypes.STRING,
      allowNull: true
    },
    doi: {
      type: DataTypes.STRING,
      allowNull: true
    },
    description: {
      type: DataTypes.TEXT,
      allowNull: true
    }
  }, {
    timestamps: true
  });
  
  ErraticReference.associate = function(models) {
    ErraticReference.belongsTo(models.Erratic, {
      foreignKey: 'erraticId',
      as: 'erratic'
    });
  };
  
  return ErraticReference;
}; 