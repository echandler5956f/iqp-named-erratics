const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const logger = require('../utils/logger'); // Import logger

/**
 * Service for executing Python scripts and processing results
 */
class PythonService {
  /**
   * Execute a Python script with arguments
   * @param {string} scriptName - Name of the script file without path (e.g., 'proximity_analysis.py')
   * @param {Array<string>} args - Arguments to pass to the script
   * @returns {Promise<object>} - JSON result from the script
   */
  async runScript(scriptName, args = []) {
    logger.info(`[PythonService] Attempting to run script: ${scriptName}`, { args });
    // Build full path to the script
    const scriptPath = path.join(__dirname, '..', 'scripts', 'python', scriptName);
    
    // Check if the script exists
    if (!fs.existsSync(scriptPath)) {
      logger.error(`[PythonService] Python script not found: ${scriptPath}`);
      throw new Error(`Python script not found: ${scriptPath}`);
    }
    
    // Use a simple python command. Assumes the correct environment is already activated
    // where this Node.js process is running, or that 'python' resolves correctly.
    // Consider using 'python3' if that is the standard command for your Python 3 environment.
    const pythonCommand = 'python'; 
    
    // Build the command
    const command = `${pythonCommand} "${scriptPath}" ${args.join(' ')}`;
    logger.debug(`[PythonService] Executing command: ${command}`);
    
    // Execute the command
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          logger.error(`[PythonService] Error executing Python script '${scriptName}'`, {
            message: error.message,
            command,
            stderr,
            stdout,
            stack: error.stack
          });
          reject(new Error(`Execution failed for script ${scriptName}: ${error.message}`));
          return;
        }
        
        if (stderr) {
          logger.warn(`[PythonService] Python script '${scriptName}' produced stderr output`, {
            stderr,
            command
          });
        }
        
        try {
          // Try to parse the stdout as JSON
          const result = JSON.parse(stdout);
          logger.info(`[PythonService] Successfully executed script '${scriptName}' and parsed JSON output.`);
          logger.debug(`[PythonService] Script '${scriptName}' stdout (parsed)`, { result });
          resolve(result);
        } catch (parseError) {
          logger.error(`[PythonService] Failed to parse JSON output from script '${scriptName}'`, {
            parseErrorMessage: parseError.message,
            stdout,
            stderr,
            command,
            stack: parseError.stack
          });
          reject(new Error(`Invalid JSON output from Python script ${scriptName}: ${parseError.message}`));
        }
      });
    });
  }
  
  /**
   * Run proximity analysis for an erratic
   * @param {number} erraticId - ID of the erratic to analyze
   * @param {Array<string>} featureLayers - Feature layers to analyze
   * @param {boolean} updateDb - Whether to update the database with results
   * @returns {Promise<object>} - Analysis results
   */
  async runProximityAnalysis(erraticId, featureLayers = [], updateDb = false) {
    logger.info(`[PythonService] Running proximity analysis for erratic ID: ${erraticId}`, { featureLayers, updateDb });
    // Build arguments
    const args = [erraticId.toString()];
    
    if (featureLayers.length > 0) {
      args.push('--features', ...featureLayers);
    }
    
    if (updateDb) {
      args.push('--update-db');
    }
    
    // Run the script
    return this.runScript('proximity_analysis.py', args);
  }
  
  /**
   * Run classification for an erratic
   * @param {number} erraticId - ID of the erratic to classify
   * @param {boolean} updateDb - Whether to update the database with results
   * @returns {Promise<object>} - Classification results
   */
  async runClassification(erraticId, updateDb = false) {
    logger.info(`[PythonService] Running classification for erratic ID: ${erraticId}`, { updateDb });
    // Build arguments
    const args = [erraticId.toString()];
    
    if (updateDb) {
      args.push('--update-db');
    }
    
    // Run the script
    return this.runScript('classify_erratic.py', args);
  }

  /**
   * Run clustering analysis on erratics
   * @param {string} algorithm - Clustering algorithm to use (e.g., 'dbscan', 'kmeans', 'hierarchical')
   * @param {Array<string>} [featuresToCluster=[]] - Features to use for clustering (e.g., ['latitude', 'longitude'])
   * @param {object} [algoParams={}] - Algorithm-specific parameters (e.g., { eps: 0.5, min_samples: 5 })
   * @param {boolean} [outputToFile=false] - Whether to output results to a file
   * @param {string} [outputFilename='clustering_results.json'] - Filename for output
   * @returns {Promise<object>} - Clustering results
   */
  async runClusteringAnalysis(algorithm, featuresToCluster = [], algoParams = {}, outputToFile = false, outputFilename = 'clustering_results.json') {
    logger.info('[PythonService] Running clustering analysis', { algorithm, featuresToCluster, algoParams, outputToFile, outputFilename });
    const args = ['--algorithm', algorithm];

    if (featuresToCluster.length > 0) {
      args.push('--features', ...featuresToCluster);
    }

    if (Object.keys(algoParams).length > 0) {
      args.push('--algo_params', JSON.stringify(algoParams));
    }
    
    if (outputToFile) {
      args.push('--output', outputFilename);
    }

    return this.runScript('clustering.py', args);
  }

  /**
   * Trigger the building of topic models.
   * @param {string} [outputPath='build_topics_result.json'] - Path to save the output/log of the build process.
   * @returns {Promise<object>} - Result from the script (usually a log or status message).
   */
  async runBuildTopicModels(outputPath = 'build_topics_result.json') {
    logger.info('[PythonService] Running build topic models', { outputPath });
    // The script expects an erratic_id, but it's not used for --build-topics. Use a placeholder.
    const placeholderErraticId = '1'; 
    const args = [
      placeholderErraticId,
      '--build-topics',
      '--output', outputPath
    ];
    
    return this.runScript('classify_erratic.py', args);
  }
}

module.exports = new PythonService(); 