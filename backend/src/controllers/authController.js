const authService = require('../services/authService');
const logger = require('../utils/logger'); // Import logger

// Register a new user
exports.register = async (req, res) => {
  try {
    const result = await authService.register(req.body);
    res.status(201).json(result);
  } catch (error) {
    logger.error('Registration controller error', { 
      message: error.message, 
      statusCode: error.statusCode, 
      requestBody: req.body, // Log problematic request body for debugging
      stack: error.stack 
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'An unexpected error occurred during registration.' });
  }
};

// User login
exports.login = async (req, res) => {
  try {
    const result = await authService.login(req.body);
    res.json(result);
  } catch (error) {
    logger.error('Login controller error', { 
      message: error.message, 
      statusCode: error.statusCode, 
      usernameAttempt: req.body?.username, // Log username for context
      stack: error.stack 
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'An unexpected error occurred during login.' });
  }
};

// Get current user profile
exports.getProfile = async (req, res) => {
  try {
    // req.user is attached by the authenticateToken middleware
    const userProfile = await authService.getProfile(req.user.id);
    res.json(userProfile);
  } catch (error) {
    logger.error('Profile controller error', { 
      message: error.message, 
      statusCode: error.statusCode, 
      userId: req.user?.id, 
      stack: error.stack 
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'An unexpected error occurred fetching profile.' });
  }
}; 