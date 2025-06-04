const pythonService = require('../services/pythonService');
const logger = require('../utils/logger');
const { jobStore, generateJobId } = require('../utils/jobStore');

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
    this.getJobStatus = this.getJobStatus.bind(this);

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
      
      const jobId = generateJobId('batch_proximity');
      jobStore.addJob(jobId, 'batch_proximity', { count: erraticIds.length, featureLayers, updateDb });
      logger.info(`[AnalysisController] Starting batch proximity analysis job ID: ${jobId}`, { count: erraticIds.length });
      
      res.status(202).json({ 
        message: 'Batch proximity analysis accepted', 
        job_id: jobId,
      });
      
      // Non-blocking execution
      this._processBatchAnalysis(jobId, erraticIds, featureLayers, updateDb)
        // The then/catch here are for the _processBatchAnalysis promise itself, not the Python script calls within it.
        .then(summary => {
            logger.info(`[AnalysisController] Batch proximity analysis job ID ${jobId} processing function completed.`, { summary });
            // Final status update will be handled within _processBatchAnalysis based on accumulated results
        })
        .catch(error => {
            logger.error(`[AnalysisController] Critical error in _processBatchAnalysis itself for job ID ${jobId}`, { message: error.message, stack: error.stack });
            jobStore.updateJobStatus(jobId, 'failed', { error: 'Background processing function failed critically.' });
        });
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
    jobStore.updateJobStatus(jobId, 'running');
    logger.info(`[AnalysisController] Background processing batch proximity analysis for job ID: ${jobId}`, { count: erraticIds.length });
    const summary = { successful: 0, failed: 0, errors: [], total: erraticIds.length };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
        if (result.error) {
          logger.warn(`[AnalysisController] Batch proximity item failed for erratic ID ${erraticId} (Job ${jobId})`, { error: result.error });
          summary.failed++;
          summary.errors.push({ id: erraticId, error: result.error });
        } else {
          summary.successful++;
        }
      } catch (error) {
        logger.error(`[AnalysisController] Exception during batch proximity item for erratic ID ${erraticId} (Job ${jobId})`, { message: error.message });
        summary.failed++;
        summary.errors.push({ id: erraticId, error: error.message });
      }
    }

    const finalStatus = summary.failed === 0 ? 'completed' : 'completed_with_errors';
    jobStore.updateJobStatus(jobId, finalStatus, { result: summary });
    logger.info(`[AnalysisController] Batch proximity processing finished for job ID: ${jobId}`, { summary });
    return summary;
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
      
      const jobId = generateJobId('batch_classify');
      jobStore.addJob(jobId, 'batch_classification', { count: erraticIds.length, updateDb });
      logger.info(`[AnalysisController] Starting batch classification job ID: ${jobId}`, { count: erraticIds.length });

      res.status(202).json({ 
        message: 'Batch classification accepted', 
        job_id: jobId,
      });
      
      this._processBatchClassification(jobId, erraticIds, updateDb)
        .then(summary => {
            logger.info(`[AnalysisController] Batch classification job ID ${jobId} processing function completed.`, { summary });
        })
        .catch(error => {
            logger.error(`[AnalysisController] Critical error in _processBatchClassification itself for job ID ${jobId}`, { message: error.message, stack: error.stack });
            jobStore.updateJobStatus(jobId, 'failed', { error: 'Background processing function failed critically.' });
        });
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
    jobStore.updateJobStatus(jobId, 'running');
    logger.info(`[AnalysisController] Background processing batch classification for job ID: ${jobId}`, { count: erraticIds.length });
    const summary = { successful: 0, failed: 0, errors: [], total: erraticIds.length };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonService.runClassification(erraticId, updateDb);
        if (result.error) {
          logger.warn(`[AnalysisController] Batch classification item failed for erratic ID ${erraticId} (Job ${jobId})`, { error: result.error });
          summary.failed++;
          summary.errors.push({ id: erraticId, error: result.error });
        } else {
          summary.successful++;
        }
      } catch (error) {
        logger.error(`[AnalysisController] Exception during batch classification item for erratic ID ${erraticId} (Job ${jobId})`, { message: error.message });
        summary.failed++;
        summary.errors.push({ id: erraticId, error: error.message });
      }
    }
    const finalStatus = summary.failed === 0 ? 'completed' : 'completed_with_errors';
    jobStore.updateJobStatus(jobId, finalStatus, { result: summary });
    logger.info(`[AnalysisController] Batch classification processing finished for job ID: ${jobId}`, { summary });
    return summary;
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
      const jobId = generateJobId('clustering');
      const jobParams = { algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename }; 
      jobStore.addJob(jobId, 'clustering', jobParams);

      logger.info(`[AnalysisController] Starting clustering analysis job ID: ${jobId}`, jobParams);
      res.status(202).json({ 
        message: 'Clustering analysis job accepted', 
        job_id: jobId,
      });

      jobStore.updateJobStatus(jobId, 'running');
      pythonService.runClusteringAnalysis(algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename)
        .then(results => {
          if (results.error) {
            logger.error(`[AnalysisController] Error in clustering analysis (Job ${jobId})`, { error: results.error, details: results.details });
            jobStore.updateJobStatus(jobId, 'failed', { error: results.error, details: results.details });
          } else {
            logger.info(`[AnalysisController] Clustering analysis (Job ${jobId}) completed successfully.`, { results });
            jobStore.updateJobStatus(jobId, 'completed', { result: results });
          }
        })
        .catch(error => {
          logger.error(`[AnalysisController] Failed to execute clustering analysis script (Job ${jobId})`, { message: error.message, stack: error.stack });
          jobStore.updateJobStatus(jobId, 'failed', { error: error.message });
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
      const jobId = generateJobId('topic_modeling');
      jobStore.addJob(jobId, 'topic_modeling', { outputPath });

      logger.info(`[AnalysisController] Starting topic model building job ID: ${jobId}`, { outputPath });
      res.status(202).json({ 
        message: 'Topic model building process accepted',
        job_id: jobId,
      });

      jobStore.updateJobStatus(jobId, 'running');
      pythonService.runBuildTopicModels(outputPath)
        .then(results => {
          if (results.error) {
            logger.error(`[AnalysisController] Error in topic model building (Job ${jobId})`, { error: results.error, details: results.details });
            jobStore.updateJobStatus(jobId, 'failed', { error: results.error, details: results.details });
          } else {
            logger.info(`[AnalysisController] Topic model building (Job ${jobId}) completed successfully.`, { results });
            jobStore.updateJobStatus(jobId, 'completed', { result: results });
          }
        })
        .catch(error => {
          logger.error(`[AnalysisController] Failed to execute topic model building script (Job ${jobId})`, { message: error.message, stack: error.stack });
          jobStore.updateJobStatus(jobId, 'failed', { error: error.message });
        });

    } catch (error) {
      logger.error('[AnalysisController] Error starting topic model building', { message: error.message, stack: error.stack });
      res.status(500).json({ error: error.message || 'Error starting topic model building' });
    }
  }

  // New method to get job status
  async getJobStatus(req, res) {
    const { jobId } = req.params;
    logger.info(`[AnalysisController] getJobStatus called for Job ID: ${jobId}`);
    const job = jobStore.getJob(jobId);
    if (!job) {
      logger.warn(`[AnalysisController] Job not found for ID: ${jobId}`);
      return res.status(404).json({ message: 'Job not found' });
    }
    // For simplicity, we return the whole job object. 
    // In a more complex scenario, you might want to filter what's returned.
    res.json(job);
  }
}

module.exports = new AnalysisController(); 