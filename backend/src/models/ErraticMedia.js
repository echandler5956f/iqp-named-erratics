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
      defaultValue: 'image',
      validate: {
        notEmpty: { args: true, msg: 'Media type cannot be empty.' },
        isIn: {
          args: [['image', 'video', 'document', 'other']],
          msg: 'Media type must be one of: image, video, document, other.'
        }
      }
    },
    url: {
      type: DataTypes.STRING,
      allowNull: false,
      validate: {
        notEmpty: { args: true, msg: 'URL cannot be empty.' },
        isUrl: { args: true, msg: 'Must be a valid URL.' }
      }
    },
    title: {
      type: DataTypes.STRING,
      allowNull: true,
      validate: {
        len: {
          args: [0, 255],
          msg: 'Title must be 255 characters or less.'
        }
      }
    },
    description: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    credit: {
      type: DataTypes.STRING,
      allowNull: true,
      validate: {
        len: {
          args: [0, 255],
          msg: 'Credit must be 255 characters or less.'
        }
      }
    },
    capture_date: {
      type: DataTypes.DATE,
      allowNull: true,
      validate: {
        isDate: {
          args: true,
          msg: 'Capture date must be a valid date.'
        }
      }
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