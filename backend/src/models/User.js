const bcrypt = require('bcrypt');

module.exports = (sequelize, DataTypes) => {
  const User = sequelize.define('User', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    username: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: {
        args: true,
        msg: 'Username already in use!'
      },
      validate: {
        notEmpty: {
          args: true,
          msg: 'Username cannot be empty.'
        },
        len: {
          args: [3, 30],
          msg: 'Username must be between 3 and 30 characters.'
        }
      }
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: {
        args: true,
        msg: 'Email address already in use!'
      },
      validate: {
        isEmail: {
          args: true,
          msg: 'Please enter a valid email address.'
        },
        notEmpty: {
          args: true,
          msg: 'Email cannot be empty.'
        }
      }
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false,
      validate: {
        notEmpty: {
          args: true,
          msg: 'Password cannot be empty.'
        },
        len: {
          args: [8, 100],
          msg: 'Password must be between 8 and 100 characters.'
        }
      }
    },
    is_admin: {
      type: DataTypes.BOOLEAN,
      defaultValue: false
    },
    last_login: {
      type: DataTypes.DATE,
      allowNull: true
    }
  }, {
    timestamps: true,
    hooks: {
      beforeCreate: async (user) => {
        if (user.password) {
          const salt = await bcrypt.genSalt(10);
          user.password = await bcrypt.hash(user.password, salt);
        }
      },
      beforeUpdate: async (user) => {
        if (user.changed('password')) {
          const salt = await bcrypt.genSalt(10);
          user.password = await bcrypt.hash(user.password, salt);
        }
      }
    }
  });
  
  // Instance method to check password validity
  User.prototype.validPassword = async function(password) {
    return await bcrypt.compare(password, this.password);
  };
  
  return User;
}; 