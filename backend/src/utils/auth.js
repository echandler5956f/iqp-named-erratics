const jwt = require('jsonwebtoken');
const db = require('../models');
const logger = require('./logger'); // Import logger

// Generate JWT token for a user
const generateToken = (user) => {
  return jwt.sign(
    { 
      id: user.id,
      username: user.username,
      is_admin: user.is_admin 
    },
    process.env.JWT_SECRET,
    { expiresIn: '24h' } // Consider making expiresIn configurable via .env
  );
};

// Middleware to verify JWT token
const authenticateToken = async (req, res, next) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN
    
    if (!token) {
      logger.warn('[AuthMiddleware] Authentication required: No token provided.');
      return res.status(401).json({ message: 'Authentication required' });
    }
    
    let decoded;
    try {
      decoded = jwt.verify(token, process.env.JWT_SECRET);
    } catch (jwtError) {
      logger.warn('[AuthMiddleware] Invalid or expired token.', { error: jwtError.message, token });
      return res.status(401).json({ message: 'Invalid or expired token' });
    }
        
    const user = await db.User.findByPk(decoded.id);
    if (!user) {
      logger.warn('[AuthMiddleware] User not found for token.', { userId: decoded.id });
      return res.status(401).json({ message: 'User not found for token' }); // More specific message
    }
    
    req.user = {
      id: user.id,
      username: user.username,
      is_admin: user.is_admin
    };
    
    next();
  } catch (error) {
    // This catch is for unexpected errors during the middleware process itself
    logger.error('[AuthMiddleware] Unexpected authentication error', { message: error.message, stack: error.stack });
    return res.status(500).json({ message: 'Internal server error during authentication' });
  }
};

// Middleware to require admin privileges
const requireAdmin = (req, res, next) => {
  if (!req.user || !req.user.is_admin) {
    logger.warn('[AuthMiddleware] Admin privileges required.', { userId: req.user?.id, username: req.user?.username });
    return res.status(403).json({ message: 'Admin privileges required' });
  }
  next();
};

module.exports = {
  generateToken,
  authenticateToken,
  requireAdmin
}; 