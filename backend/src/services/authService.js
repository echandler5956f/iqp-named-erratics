const db = require('../models');
const { generateToken } = require('../utils/auth');
const { User, Sequelize } = db;

class AuthService {
  async register(userData) {
    const { username, email, password } = userData;

    if (!username || !email || !password) {
      const error = new Error('Username, email, and password are required');
      error.statusCode = 400;
      throw error;
    }

    const existingUser = await User.findOne({
      where: {
        [Sequelize.Op.or]: [{ username }, { email }],
      },
    });

    if (existingUser) {
      const error = new Error('Username or email already exists');
      error.statusCode = 400;
      throw error;
    }

    // Password hashing is handled by User model hooks
    const user = await User.create({
      username,
      email,
      password,
      is_admin: false, // Default to non-admin
    });

    const token = generateToken(user);

    return {
      message: 'User registered successfully',
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        is_admin: user.is_admin,
      },
    };
  }

  async login(credentials) {
    const { username, password } = credentials;

    if (!username || !password) {
      const error = new Error('Username and password are required');
      error.statusCode = 400;
      throw error;
    }

    const user = await User.findOne({ where: { username } });

    if (!user) {
      const error = new Error('Invalid credentials');
      error.statusCode = 401;
      throw error;
    }

    const isPasswordValid = await user.validPassword(password);
    if (!isPasswordValid) {
      const error = new Error('Invalid credentials');
      error.statusCode = 401;
      throw error;
    }

    await user.update({ last_login: new Date() });
    const token = generateToken(user);

    return {
      message: 'Login successful',
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        is_admin: user.is_admin,
      },
    };
  }

  async getProfile(userId) {
    const user = await User.findByPk(userId, {
      attributes: { exclude: ['password'] },
    });

    if (!user) {
      const error = new Error('User not found');
      error.statusCode = 404;
      throw error;
    }
    return user;
  }
}

module.exports = new AuthService(); 