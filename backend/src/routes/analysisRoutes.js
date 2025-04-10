const express = require('express');
const router = express.Router();
const analysisController = require('../controllers/analysisController');
const { authenticateToken, requireAdmin } = require('../utils/auth');

/**
 * @route GET /api/analysis/proximity/:id
 * @description Get proximity analysis for a single erratic
 * @access Public
 */
router.get('/proximity/:id', analysisController.getProximityAnalysis);

/**
 * @route POST /api/analysis/proximity/batch
 * @description Run batch proximity analysis on multiple erratics
 * @access Admin only
 */
router.post('/proximity/batch', authenticateToken, requireAdmin, analysisController.batchProximityAnalysis);

module.exports = router; 