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

/**
 * @route GET /api/analysis/cluster
 * @description Perform spatial clustering on erratics. Triggers a background job.
 * @access Public 
 * @query {string} [algorithm=dbscan] - Clustering algorithm (dbscan, kmeans, hierarchical).
 * @query {string} [features] - Comma-separated features to cluster on (e.g., 'latitude,longitude').
 * @query {string} [algoParams] - JSON string of algorithm-specific parameters.
 * @query {string} [outputToFile=true] - 'true' or 'false' to output to file.
 * @query {string} [outputFilename=clustering_results.json] - Filename if outputToFile is true.
 */
router.get('/cluster', analysisController.getClusterAnalysis);

/**
 * @route POST /api/analysis/build-topics
 * @description Trigger the building of NLP topic models. Triggers a background job.
 * @access Admin only
 * @body {{ outputPath?: string }} [outputPath=build_topics_result.json] - Path to save the output/log.
 */
router.post('/build-topics', authenticateToken, requireAdmin, analysisController.triggerBuildTopicModels);

module.exports = router; 