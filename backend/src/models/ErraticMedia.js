module.exports = (sequelize, DataTypes) => {
  const ErraticMedia = sequelize.define('ErraticMedia', {
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
      },
      onUpdate: 'CASCADE',
      onDelete: 'CASCADE'
    },
    media_type: {
      type: DataTypes.ENUM('image', 'video', 'document', 'other'),
      allowNull: false,
      defaultValue: 'image'
    },
    url: {
      type: DataTypes.STRING,
      allowNull: false
    },
    title: {
      type: DataTypes.STRING,
      allowNull: true
    },
    description: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    credit: {
      type: DataTypes.STRING,
      allowNull: true
    },
    capture_date: {
      type: DataTypes.DATE,
      allowNull: true
    }
  }, {
    timestamps: true
  });
  
  ErraticMedia.associate = function(models) {
    ErraticMedia.belongsTo(models.Erratic, {
      foreignKey: 'erraticId',
      as: 'erratic'
    });
  };
  
  return ErraticMedia;
}; 