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

/**
 * @route GET /api/analysis/classify/:id
 * @description Classify an erratic based on its description and attributes
 * @access Public
 */
router.get('/classify/:id', analysisController.classifyErratic);

/**
 * @route POST /api/analysis/classify/batch
 * @description Run batch classification on multiple erratics
 * @access Admin only
 */
router.post('/classify/batch', authenticateToken, requireAdmin, analysisController.batchClassifyErratics);

module.exports = router; 