const express = require('express');
const router = express.Router();
const erraticController = require('../controllers/erraticController');
const { authenticateToken, requireAdmin } = require('../utils/auth');

// Public routes
router.get('/', erraticController.getAllErratics);
router.get('/nearby', erraticController.getNearbyErratics);
router.get('/:id', erraticController.getErraticById);

// Protected admin routes
router.post('/', authenticateToken, requireAdmin, erraticController.createErratic);
router.put('/:id', authenticateToken, requireAdmin, erraticController.updateErratic);
router.delete('/:id', authenticateToken, requireAdmin, erraticController.deleteErratic);

module.exports = router; 