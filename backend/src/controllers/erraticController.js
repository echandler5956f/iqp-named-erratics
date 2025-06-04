const erraticService = require('../services/erraticService');

// Get all erratics with optional filtering
exports.getAllErratics = async (req, res) => {
  try {
    const erratics = await erraticService.getAllErratics(req.query);
    res.json(erratics);
  } catch (error) {
    console.error('[ErraticController] Error fetching all erratics:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// Get a single erratic by ID
exports.getErraticById = async (req, res) => {
  try {
    const erratic = await erraticService.getErraticById(req.params.id);
    res.json(erratic);
  } catch (error) {
    console.error('[ErraticController] Error fetching erratic by ID:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// Get erratics within a certain radius
exports.getNearbyErratics = async (req, res) => {
  try {
    const { lat, lng, radius } = req.query;
    const erratics = await erraticService.getNearbyErratics(lat, lng, radius);
    res.json(erratics);
  } catch (error) {
    console.error('[ErraticController] Error fetching nearby erratics:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// Create a new erratic
exports.createErratic = async (req, res) => {
  try {
    const result = await erraticService.createErratic(req.body);
    res.status(201).json(result);
  } catch (error) {
    console.error('[ErraticController] Error creating erratic:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// Update an existing erratic
exports.updateErratic = async (req, res) => {
  try {
    const result = await erraticService.updateErratic(req.params.id, req.body);
    res.json(result);
  } catch (error) {
    console.error('[ErraticController] Error updating erratic:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
};

// Delete an erratic
exports.deleteErratic = async (req, res) => {
  try {
    const result = await erraticService.deleteErratic(req.params.id);
    res.json(result);
  } catch (error) {
    console.error('[ErraticController] Error deleting erratic:', error.message);
    res.status(error.statusCode || 500).json({ message: error.message });
  }
}; 