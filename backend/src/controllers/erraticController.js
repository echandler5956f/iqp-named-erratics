const erraticService = require('../services/erraticService');
const logger = require('../utils/logger'); // Import logger

// Get all erratics with optional filtering
exports.getAllErratics = async (req, res) => {
  try {
    const erratics = await erraticService.getAllErratics(req.query);
    res.json(erratics);
  } catch (error) {
    logger.error('[ErraticController] Error fetching all erratics', {
      message: error.message,
      statusCode: error.statusCode,
      query: req.query,
      stack: error.stack
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'Failed to fetch erratics.' });
  }
};

// Get a single erratic by ID
exports.getErraticById = async (req, res) => {
  try {
    const erratic = await erraticService.getErraticById(req.params.id);
    res.json(erratic);
  } catch (error) {
    logger.error('[ErraticController] Error fetching erratic by ID', {
      message: error.message,
      statusCode: error.statusCode,
      erraticId: req.params.id,
      stack: error.stack
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'Failed to fetch erratic details.' });
  }
};

// Get erratics within a certain radius
exports.getNearbyErratics = async (req, res) => {
  try {
    const { lat, lng, radius } = req.query;
    const erratics = await erraticService.getNearbyErratics(lat, lng, radius);
    res.json(erratics);
  } catch (error) {
    logger.error('[ErraticController] Error fetching nearby erratics', {
      message: error.message,
      statusCode: error.statusCode,
      query: req.query,
      stack: error.stack
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'Failed to fetch nearby erratics.' });
  }
};

// Create a new erratic
exports.createErratic = async (req, res) => {
  try {
    const result = await erraticService.createErratic(req.body);
    res.status(201).json(result);
  } catch (error) {
    logger.error('[ErraticController] Error creating erratic', {
      message: error.message,
      statusCode: error.statusCode,
      requestBody: req.body,
      stack: error.stack
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'Failed to create erratic.' });
  }
};

// Update an existing erratic
exports.updateErratic = async (req, res) => {
  try {
    const result = await erraticService.updateErratic(req.params.id, req.body);
    res.json(result);
  } catch (error) {
    logger.error('[ErraticController] Error updating erratic', {
      message: error.message,
      statusCode: error.statusCode,
      erraticId: req.params.id,
      requestBody: req.body,
      stack: error.stack
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'Failed to update erratic.' });
  }
};

// Delete an erratic
exports.deleteErratic = async (req, res) => {
  try {
    const result = await erraticService.deleteErratic(req.params.id);
    res.json(result);
  } catch (error) {
    logger.error('[ErraticController] Error deleting erratic', {
      message: error.message,
      statusCode: error.statusCode,
      erraticId: req.params.id,
      stack: error.stack
    });
    res.status(error.statusCode || 500).json({ message: error.message || 'Failed to delete erratic.' });
  }
}; 