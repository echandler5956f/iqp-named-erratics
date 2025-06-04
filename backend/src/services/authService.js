const db = require('../models');
const { generateToken } = require('../utils/auth');
const logger = require('../utils/logger');
const { User, Sequelize } = db;

class AuthService {
  async register(userData) {
    const { username, email, password } = userData;
    logger.info(`Attempting to register user: ${username}`);

    if (!username || !email || !password) {
      const error = new Error('Username, email, and password are required');
      error.statusCode = 400;
      logger.warn(`Registration failed for ${username}: ${error.message}`);
      throw error;
    }

    try {
      const existingUser = await User.findOne({
        where: {
          [Sequelize.Op.or]: [{ username }, { email }],
        },
      });

      if (existingUser) {
        const error = new Error('Username or email already exists');
        error.statusCode = 400;
        logger.warn(`Registration failed for ${username}: ${error.message}`);
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
      logger.info(`User ${username} registered successfully with ID: ${user.id}`);

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
    } catch (dbError) {
      logger.error(`Database error during registration for ${username}: ${dbError.message}`, { stack: dbError.stack });
      // Re-throw or throw a generic error
      const serviceError = new Error('User registration failed due to a server error.');
      serviceError.statusCode = 500;
      throw serviceError;
    }
  }

  async login(credentials) {
    const { username, password } = credentials;
    logger.info(`Attempting login for user: ${username}`);

    if (!username || !password) {
      const error = new Error('Username and password are required');
      error.statusCode = 400;
      logger.warn(`Login failed for ${username}: ${error.message}`);
      throw error;
    }

    try {
      const user = await User.findOne({ where: { username } });

      if (!user) {
        const error = new Error('Invalid credentials');
        error.statusCode = 401;
        logger.warn(`Login failed for ${username}: User not found.`);
        throw error;
      }

      const isPasswordValid = await user.validPassword(password);
      if (!isPasswordValid) {
        const error = new Error('Invalid credentials');
        error.statusCode = 401;
        logger.warn(`Login failed for ${username}: Invalid password.`);
        throw error;
      }

      await user.update({ last_login: new Date() });
      const token = generateToken(user);
      logger.info(`User ${username} logged in successfully.`);

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
    } catch (dbError) {
      // Catch errors not already handled (e.g. user.validPassword or user.update throwing something unexpected)
      logger.error(`Database or unexpected error during login for ${username}: ${dbError.message}`, { stack: dbError.stack });
      const serviceError = new Error('Login failed due to a server error.');
      serviceError.statusCode = 500;
      throw serviceError;
    }
  }

  async getProfile(userId) {
    logger.info(`Fetching profile for user ID: ${userId}`);
    try {
      const user = await User.findByPk(userId, {
        attributes: { exclude: ['password'] },
      });

      if (!user) {
        const error = new Error('User not found');
        error.statusCode = 404;
        logger.warn(`Profile fetch failed: User ID ${userId} not found.`);
        throw error;
      }
      logger.info(`Profile fetched successfully for user ID: ${userId}`);
      return user;
    } catch (dbError) {
      logger.error(`Database error fetching profile for user ID ${userId}: ${dbError.message}`, { stack: dbError.stack });
      const serviceError = new Error('Failed to fetch user profile.');
      serviceError.statusCode = 500;
      throw serviceError;
    }
  }
}

module.exports = new AuthService(); 