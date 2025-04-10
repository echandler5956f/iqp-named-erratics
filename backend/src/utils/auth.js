const jwt = require('jsonwebtoken');
const db = require('../models');

// Generate JWT token for a user
const generateToken = (user) => {
  return jwt.sign(
    { 
      id: user.id,
      username: user.username,
      is_admin: user.is_admin 
    },
    process.env.JWT_SECRET,
    { expiresIn: '24h' }
  );
};

// Middleware to verify JWT token
const authenticateToken = async (req, res, next) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN
    
    if (!token) {
      return res.status(401).json({ message: 'Authentication required' });
    }
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Find the user in the database to ensure they still exist
    const user = await db.User.findByPk(decoded.id);
    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }
    
    // Attach user info to request
    req.user = {
      id: user.id,
      username: user.username,
      is_admin: user.is_admin
    };
    
    next();
  } catch (error) {
    console.error('Authentication error:', error);
    return res.status(401).json({ message: 'Invalid or expired token' });
  }
};

// Middleware to require admin privileges
const requireAdmin = (req, res, next) => {
  if (!req.user || !req.user.is_admin) {
    return res.status(403).json({ message: 'Admin privileges required' });
  }
  next();
};

module.exports = {
  generateToken,
  authenticateToken,
  requireAdmin
}; 