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
    // Bind methods to the instance
    this._processBatchAnalysis = this._processBatchAnalysis.bind(this);
    this._processBatchClassification = this._processBatchClassification.bind(this);
  }

  /**
   * Get proximity analysis for an erratic
   * @param {object} req - Express request object
   * @param {object} res - Express response object
   */
  async getProximityAnalysis(req, res) {
    try {
      const erraticId = parseInt(req.params.id);
      
      if (isNaN(erraticId)) {
        return res.status(400).json({ error: 'Invalid erratic ID' });
      }
      
      // Extract feature layers from query
      const featureLayers = req.query.features ? req.query.features.split(',') : [];
      
      // Determine if we should update the database
      const updateDb = req.query.update === 'true';
      
      // Run the analysis
      const results = await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
      
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
}

module.exports = new AnalysisController(); 