const pythonService = require('../services/pythonService');
const logger = require('../utils/logger');

/**
 * Controller for spatial analysis operations
 */
class AnalysisController {
  /**
   * Constructor to bind methods
   */
  constructor() {
    // Bind public methods used as route handlers
    this.getProximityAnalysis = this.getProximityAnalysis.bind(this);
    this.batchProximityAnalysis = this.batchProximityAnalysis.bind(this);
    this.classifyErratic = this.classifyErratic.bind(this);
    this.batchClassifyErratics = this.batchClassifyErratics.bind(this);
    this.getClusterAnalysis = this.getClusterAnalysis.bind(this);
    this.triggerBuildTopicModels = this.triggerBuildTopicModels.bind(this);

    // Bind private helper methods (if their `this` context might be lost, though less likely if called via this.*)
    this._processBatchAnalysis = this._processBatchAnalysis.bind(this);
    this._processBatchClassification = this._processBatchClassification.bind(this);
  }

  /**
   * Get proximity analysis for an erratic
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async getProximityAnalysis(req, res) {
    const erraticId = parseInt(req.params.id, 10);
    logger.info(`[AnalysisController] getProximityAnalysis called for ID: ${erraticId}`);

    if (isNaN(erraticId)) {
      logger.warn('[AnalysisController] Invalid erratic ID received for proximity analysis', { requestedId: req.params.id });
      return res.status(400).json({ error: 'Invalid erratic ID' });
    }
    
    // Extract feature layers from query
    const featureLayers = req.query.features ? req.query.features.split(',') : [];
    
    // Determine if we should update the database
    const updateDb = req.query.update === 'true';
    
    try {
      logger.info(`[AnalysisController] Calling pythonService.runProximityAnalysis for ID: ${erraticId}`, { featureLayers, updateDb });
      const results = await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
      logger.info(`[AnalysisController] pythonService.runProximityAnalysis returned for ID: ${erraticId}`);
      logger.debug('[AnalysisController] Proximity results:', { results });
      
      // Check for error in results
      if (results.error) {
        // Log specific error from pythonService if available
        logger.warn(`[AnalysisController] Proximity analysis for ID ${erraticId} returned error from service`, { error: results.error });
        return res.status(results.statusCode || 404).json({ error: results.error });
      }
      
      res.json(results);
    } catch (error) {
      logger.error(`[AnalysisController] Error in getProximityAnalysis for ID ${erraticId}`, { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error in proximity analysis' });
    }
  }
  
  /**
   * Run proximity analysis in batch mode for multiple erratics
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async batchProximityAnalysis(req, res) {
    logger.info('[AnalysisController] batchProximityAnalysis called', { body: req.body });
    try {
      // Extract erratic IDs from request body
      const { erraticIds, featureLayers, updateDb } = req.body;
      
      if (!Array.isArray(erraticIds) || erraticIds.length === 0) {
        logger.warn('[AnalysisController] Invalid or empty erratic ID list for batch proximity', { erraticIds });
        return res.status(400).json({ error: 'Invalid or empty erratic ID list' });
      }
      
      // Start a background job for processing
      const jobId = `batch_proximity_${Date.now()}`;
      logger.info(`[AnalysisController] Starting batch proximity analysis job ID: ${jobId}`, { count: erraticIds.length });
      res.status(202).json({ 
        message: 'Batch proximity analysis started', 
        job_id: jobId,
        erratics_count: erraticIds.length
      });
      
      // Process erratics in background
      this._processBatchAnalysis(jobId, erraticIds, featureLayers, updateDb)
        .then(summary => logger.info(`[AnalysisController] Batch proximity analysis job ID ${jobId} completed.`, { summary }))
        .catch(error => logger.error(`[AnalysisController] Error in background batch proximity processing for job ID ${jobId}`, { message: error.message, stack: error.stack }));
    } catch (error) {
      logger.error('[AnalysisController] Error starting batch proximity analysis', { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error starting batch analysis' });
    }
  }
  
  /**
   * Process batch analysis in background
   * @param {string} jobId - Job ID
   * @param {Array<number>} erraticIds - List of erratic IDs to process
   * @param {Array<string>} featureLayers - Feature layers to analyze
   * @param {boolean} updateDb - Whether to update the database
   * @private
   */
  async _processBatchAnalysis(jobId, erraticIds, featureLayers = [], updateDb = true) {
    logger.info(`[AnalysisController] Background processing batch proximity analysis for job ID: ${jobId}`, { count: erraticIds.length });
    const resultsSummary = { successful: 0, failed: 0, errors: [] };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
        if (result.error) {
          logger.warn(`[AnalysisController] Batch proximity failed for erratic ID ${erraticId} (Job ${jobId})`, { error: result.error });
          resultsSummary.failed++;
          resultsSummary.errors.push({ id: erraticId, error: result.error });
        } else {
          resultsSummary.successful++;
        }
      } catch (error) {
        logger.error(`[AnalysisController] Exception during batch proximity for erratic ID ${erraticId} (Job ${jobId})`, { message: error.message });
        resultsSummary.failed++;
        resultsSummary.errors.push({ id: erraticId, error: error.message });
      }
    }
    logger.info(`[AnalysisController] Batch proximity processing finished for job ID: ${jobId}`, { resultsSummary });
    return resultsSummary;
  }
  
  /**
   * Classify an erratic using NLP and ML techniques
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async classifyErratic(req, res) {
    const erraticId = parseInt(req.params.id, 10);
    logger.info(`[AnalysisController] classifyErratic called for ID: ${erraticId}`);

    if (isNaN(erraticId)) {
      logger.warn('[AnalysisController] Invalid erratic ID received for classification', { requestedId: req.params.id });
      return res.status(400).json({ error: 'Invalid erratic ID' });
    }
    const updateDb = req.query.update === 'true';

    try {
      logger.info(`[AnalysisController] Calling pythonService.runClassification for ID: ${erraticId}`, { updateDb });
      const results = await pythonService.runClassification(erraticId, updateDb);
      logger.info(`[AnalysisController] pythonService.runClassification returned for ID: ${erraticId}`);
      logger.debug('[AnalysisController] Classification results:', { results });

      if (results.error) {
        logger.warn(`[AnalysisController] Classification for ID ${erraticId} returned error from service`, { error: results.error });
        return res.status(results.statusCode || 404).json({ error: results.error });
      }
      res.json(results);
    } catch (error) {
      logger.error(`[AnalysisController] Error in classifyErratic for ID ${erraticId}`, { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error in erratic classification' });
    }
  }
  
  /**
   * Run classification in batch mode for multiple erratics
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async batchClassifyErratics(req, res) {
    logger.info('[AnalysisController] batchClassifyErratics called', { body: req.body });
    try {
      // Extract erratic IDs from request body
      const { erraticIds, updateDb } = req.body;
      
      if (!Array.isArray(erraticIds) || erraticIds.length === 0) {
        logger.warn('[AnalysisController] Invalid or empty erratic ID list for batch classification', { erraticIds });
        return res.status(400).json({ error: 'Invalid or empty erratic ID list' });
      }
      
      // Start a background job for processing
      const jobId = `batch_classify_${Date.now()}`;
      logger.info(`[AnalysisController] Starting batch classification job ID: ${jobId}`, { count: erraticIds.length });
      res.status(202).json({ 
        message: 'Batch classification started', 
        job_id: jobId,
        erratics_count: erraticIds.length
      });
      
      // Process erratics in background
      this._processBatchClassification(jobId, erraticIds, updateDb)
        .then(summary => logger.info(`[AnalysisController] Batch classification job ID ${jobId} completed.`, { summary }))
        .catch(error => logger.error(`[AnalysisController] Error in background batch classification for job ID ${jobId}`, { message: error.message, stack: error.stack }));
    } catch (error) {
      logger.error('[AnalysisController] Error starting batch classification', { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error starting batch classification' });
    }
  }
  
  /**
   * Process batch classification in background
   * @param {Array<number>} erraticIds - List of erratic IDs to process
   * @param {boolean} updateDb - Whether to update the database
   * @private
   */
  async _processBatchClassification(jobId, erraticIds, updateDb = true) {
    logger.info(`[AnalysisController] Background processing batch classification for job ID: ${jobId}`, { count: erraticIds.length });
    const resultsSummary = { successful: 0, failed: 0, errors: [] };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonService.runClassification(erraticId, updateDb);
        if (result.error) {
          logger.warn(`[AnalysisController] Batch classification failed for erratic ID ${erraticId} (Job ${jobId})`, { error: result.error });
          resultsSummary.failed++;
          resultsSummary.errors.push({ id: erraticId, error: result.error });
        } else {
          resultsSummary.successful++;
        }
      } catch (error) {
        logger.error(`[AnalysisController] Exception during batch classification for erratic ID ${erraticId} (Job ${jobId})`, { message: error.message });
        resultsSummary.failed++;
        resultsSummary.errors.push({ id: erraticId, error: error.message });
      }
    }
    logger.info(`[AnalysisController] Batch classification processing finished for job ID: ${jobId}`, { resultsSummary });
    return resultsSummary;
  }

  /**
   * Perform spatial clustering on erratics
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async getClusterAnalysis(req, res) {
    logger.info('[AnalysisController] getClusterAnalysis called', { query: req.query });
    try {
      const { 
        algorithm = 'dbscan', // Default algorithm
        features, 
        algoParams, 
        outputToFile = 'true', 
        outputFilename = 'clustering_results.json' 
      } = req.query;

      const featuresToCluster = features ? features.split(',') : [];
      let parsedAlgoParams = {};
      if (algoParams) {
        try {
          parsedAlgoParams = JSON.parse(algoParams);
        } catch (e) {
          logger.warn('[AnalysisController] Invalid algoParams JSON string for clustering', { algoParams, error: e.message });
          return res.status(400).json({ error: 'Invalid algoParams JSON string' });
        }
      }
      const doOutputToFile = outputToFile === 'true';
      const jobId = `cluster_analysis_${Date.now()}`;

      logger.info(`[AnalysisController] Starting clustering analysis job ID: ${jobId}`, { algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename });
      res.status(202).json({ 
        message: 'Clustering analysis job started', 
        job_id: jobId,
        details: { algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename }
      });

      pythonService.runClusteringAnalysis(algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename)
        .then(results => {
          if (results.error) {
            logger.error(`[AnalysisController] Error in clustering analysis (Job ${jobId})`, { error: results.error, details: results.details });
          } else {
            logger.info(`[AnalysisController] Clustering analysis (Job ${jobId}) completed successfully.`, { results }); // Log full results on success for background job
          }
        })
        .catch(error => {
          logger.error(`[AnalysisController] Failed to execute clustering analysis script (Job ${jobId})`, { message: error.message, stack: error.stack });
        });

    } catch (error) {
      logger.error('[AnalysisController] Error starting clustering analysis', { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error starting clustering analysis' });
    }
  }

  /**
   * Trigger the building of NLP topic models
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async triggerBuildTopicModels(req, res) {
    logger.info('[AnalysisController] triggerBuildTopicModels called', { body: req.body });
    try {
      const { outputPath = 'build_topics_result.json' } = req.body;
      const jobId = `build_topics_${Date.now()}`;

      logger.info(`[AnalysisController] Starting topic model building job ID: ${jobId}`, { outputPath });
      res.status(202).json({ 
        message: 'Topic model building process started',
        job_id: jobId,
        details: { outputPath }
      });

      pythonService.runBuildTopicModels(outputPath)
        .then(results => {
          if (results.error) {
            logger.error(`[AnalysisController] Error in topic model building (Job ${jobId})`, { error: results.error, details: results.details });
          } else {
            logger.info(`[AnalysisController] Topic model building (Job ${jobId}) completed successfully.`, { results }); // Log full results on success
          }
        })
        .catch(error => {
          logger.error(`[AnalysisController] Failed to execute topic model building script (Job ${jobId})`, { message: error.message, stack: error.stack });
        });

    } catch (error) {
      logger.error('[AnalysisController] Error starting topic model building', { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error starting topic model building' });
    }
  }
}

module.exports = new AnalysisController(); 