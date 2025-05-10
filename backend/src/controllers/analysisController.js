const pythonService = require('../services/pythonService');
const db = require('../models');

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
    console.log(`[AnalysisController] getProximityAnalysis called for ID: ${req.params.id}`);
    try {
      const erraticId = parseInt(req.params.id);
      
      if (isNaN(erraticId)) {
        console.error('[AnalysisController] Invalid erratic ID:', req.params.id);
        return res.status(400).json({ error: 'Invalid erratic ID' });
      }
      
      // Extract feature layers from query
      const featureLayers = req.query.features ? req.query.features.split(',') : [];
      
      // Determine if we should update the database
      const updateDb = req.query.update === 'true';
      
      console.log(`[AnalysisController] Calling pythonService.runProximityAnalysis for ID: ${erraticId} with features: ${featureLayers.join(',')} and updateDb: ${updateDb}`);
      const results = await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
      console.log(`[AnalysisController] pythonService.runProximityAnalysis returned for ID: ${erraticId}. Results:`, JSON.stringify(results));
      
      // Check for error in results
      if (results.error) {
        return res.status(404).json({ error: results.error });
      }
      
      res.json(results);
    } catch (error) {
      console.error('Error in proximity analysis:', error);
      res.status(500).json({ error: error.message });
    }
  }
  
  /**
   * Run proximity analysis in batch mode for multiple erratics
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async batchProximityAnalysis(req, res) {
    try {
      // Extract erratic IDs from request body
      const { erraticIds, featureLayers, updateDb } = req.body;
      
      if (!Array.isArray(erraticIds) || erraticIds.length === 0) {
        return res.status(400).json({ error: 'Invalid or empty erratic ID list' });
      }
      
      // Start a background job for processing
      res.json({ 
        message: 'Batch analysis started', 
        job_id: `batch_${Date.now()}`,
        erratics_count: erraticIds.length
      });
      
      // Process erratics in background
      this._processBatchAnalysis(erraticIds, featureLayers, updateDb)
        .catch(error => console.error('Error in batch processing:', error));
    } catch (error) {
      console.error('Error starting batch analysis:', error);
      res.status(500).json({ error: error.message });
    }
  }
  
  /**
   * Process batch analysis in background
   * @param {Array<number>} erraticIds - List of erratic IDs to process
   * @param {Array<string>} featureLayers - Feature layers to analyze
   * @param {boolean} updateDb - Whether to update the database
   * @private
   */
  async _processBatchAnalysis(erraticIds, featureLayers = [], updateDb = true) {
    const results = {
      successful: [],
      failed: []
    };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
        if (result.error) {
          results.failed.push({ id: erraticId, error: result.error });
        } else {
          results.successful.push(erraticId);
        }
      } catch (error) {
        results.failed.push({ id: erraticId, error: error.message });
      }
    }
    
    console.log(`Batch analysis completed. Successful: ${results.successful.length}, Failed: ${results.failed.length}`);
    return results;
  }
  
  /**
   * Classify an erratic using NLP and ML techniques
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async classifyErratic(req, res) {
    try {
      const erraticId = parseInt(req.params.id);
      
      if (isNaN(erraticId)) {
        return res.status(400).json({ error: 'Invalid erratic ID' });
      }
      
      // Determine if we should update the database
      const updateDb = req.query.update === 'true';
      
      // Run the classification
      const results = await pythonService.runClassification(erraticId, updateDb);
      
      // Check for error in results
      if (results.error) {
        return res.status(404).json({ error: results.error });
      }
      
      res.json(results);
    } catch (error) {
      console.error('Error in erratic classification:', error);
      res.status(500).json({ error: error.message });
    }
  }
  
  /**
   * Run classification in batch mode for multiple erratics
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async batchClassifyErratics(req, res) {
    try {
      // Extract erratic IDs from request body
      const { erraticIds, updateDb } = req.body;
      
      if (!Array.isArray(erraticIds) || erraticIds.length === 0) {
        return res.status(400).json({ error: 'Invalid or empty erratic ID list' });
      }
      
      // Start a background job for processing
      res.json({ 
        message: 'Batch classification started', 
        job_id: `batch_classify_${Date.now()}`,
        erratics_count: erraticIds.length
      });
      
      // Process erratics in background
      this._processBatchClassification(erraticIds, updateDb)
        .catch(error => console.error('Error in batch classification:', error));
    } catch (error) {
      console.error('Error starting batch classification:', error);
      res.status(500).json({ error: error.message });
    }
  }
  
  /**
   * Process batch classification in background
   * @param {Array<number>} erraticIds - List of erratic IDs to process
   * @param {boolean} updateDb - Whether to update the database
   * @private
   */
  async _processBatchClassification(erraticIds, updateDb = true) {
    const results = {
      successful: [],
      failed: []
    };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonService.runClassification(erraticId, updateDb);
        if (result.error) {
          results.failed.push({ id: erraticId, error: result.error });
        } else {
          results.successful.push(erraticId);
        }
      } catch (error) {
        results.failed.push({ id: erraticId, error: error.message });
      }
    }
    
    console.log(`Batch classification completed. Successful: ${results.successful.length}, Failed: ${results.failed.length}`);
    return results;
  }

  /**
   * Perform spatial clustering on erratics
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async getClusterAnalysis(req, res) {
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
          return res.status(400).json({ error: 'Invalid algoParams JSON string' });
        }
      }
      const doOutputToFile = outputToFile === 'true';

      // Respond quickly
      res.json({ 
        message: 'Clustering analysis job started', 
        details: { algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename }
      });

      // Run clustering in the background
      pythonService.runClusteringAnalysis(algorithm, featuresToCluster, parsedAlgoParams, doOutputToFile, outputFilename)
        .then(results => {
          if (results.error) {
            console.error('Error in clustering analysis:', results.error, results.details || '');
          } else {
            console.log('Clustering analysis completed successfully:', results);
          }
        })
        .catch(error => {
          console.error('Failed to execute clustering analysis script:', error);
        });

    } catch (error) {
      console.error('Error starting clustering analysis:', error);
      res.status(500).json({ error: error.message });
    }
  }

  /**
   * Trigger the building of NLP topic models
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async triggerBuildTopicModels(req, res) {
    try {
      const { outputPath = 'build_topics_result.json' } = req.body;

      // Respond quickly
      res.status(202).json({ 
        message: 'Topic model building process started', 
        details: { outputPath }
      });

      // Run model building in the background
      pythonService.runBuildTopicModels(outputPath)
        .then(results => {
          if (results.error) {
            console.error('Error in topic model building:', results.error, results.details || '');
          } else {
            console.log('Topic model building completed successfully:', results);
          }
        })
        .catch(error => {
          console.error('Failed to execute topic model building script:', error);
        });

    } catch (error) {
      console.error('Error starting topic model building:', error);
      res.status(500).json({ error: error.message });
    }
  }
}

module.exports = new AnalysisController(); 