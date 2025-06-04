const authService = require('../services/authService');

// Register a new user
exports.register = async (req, res) => {
  try {
    const result = await authService.register(req.body);
    res.status(201).json(result);
  } catch (error) {
    console.error('Registration controller error:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// User login
exports.login = async (req, res) => {
  try {
    const result = await authService.login(req.body);
    res.json(result);
  } catch (error) {
    console.error('Login controller error:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// Get current user profile
exports.getProfile = async (req, res) => {
  try {
    // req.user is attached by the authenticateToken middleware
    const userProfile = await authService.getProfile(req.user.id);
    res.json(userProfile);
  } catch (error) {
    console.error('Profile controller error:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
}; 