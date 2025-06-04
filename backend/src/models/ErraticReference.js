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
      },
      onUpdate: 'CASCADE',
      onDelete: 'CASCADE'
    },
    reference_type: {
      type: DataTypes.ENUM('article', 'book', 'paper', 'website', 'other'),
      allowNull: false,
      validate: {
        notEmpty: { args: true, msg: 'Reference type cannot be empty.' },
        isIn: {
          args: [['article', 'book', 'paper', 'website', 'other']],
          msg: 'Reference type must be one of: article, book, paper, website, other.'
        }
      }
    },
    title: {
      type: DataTypes.STRING,
      allowNull: false,
      validate: {
        notEmpty: { args: true, msg: 'Title cannot be empty.' },
        len: { args: [1, 255], msg: 'Title must be between 1 and 255 characters.' }
      }
    },
    authors: {
      type: DataTypes.STRING,
      allowNull: true,
      validate: {
        len: { args: [0, 500], msg: 'Authors field must be 500 characters or less.' }
      }
    },
    publication: {
      type: DataTypes.STRING,
      allowNull: true,
      validate: {
        len: { args: [0, 255], msg: 'Publication must be 255 characters or less.' }
      }
    },
    year: {
      type: DataTypes.INTEGER,
      allowNull: true,
      validate: {
        isInt: { args: true, msg: 'Year must be an integer.' },
        min: { args: [1000], msg: 'Year must be a valid year (e.g., after 1000).' },
        max: { args: [new Date().getFullYear() + 5], msg: 'Year cannot be too far in the future.' }
      }
    },
    url: {
      type: DataTypes.STRING,
      allowNull: true,
      validate: {
        isUrlOrEmpty(value) {
          if (value === null || value === '' || value === undefined) {
            return;
          }
          if (!DataTypes.STRING.prototype.options.validate.isUrl(value)) {
            throw new Error('Reference URL must be a valid URL.');
          }
        }
      }
    },
    doi: {
      type: DataTypes.STRING,
      allowNull: true,
      validate: {
        len: { args: [0, 100], msg: 'DOI must be 100 characters or less.' }
      }
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