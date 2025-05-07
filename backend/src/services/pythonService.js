const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

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
    // Build full path to the script
    const scriptPath = path.join(__dirname, '..', 'scripts', 'python', scriptName);
    
    // Check if the script exists
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`Python script not found: ${scriptPath}`);
    }
    
    // Determine the Python executable to use
    // First check if the conda environment is active
    const condaEnvName = 'iqp-py310';
    const condaEnv = process.env.CONDA_DEFAULT_ENV;
    
    let pythonCommand = 'python';
    
    // If we're not in the right conda environment, try to activate it
    if (condaEnv !== condaEnvName) {
      // On Windows
      if (os.platform() === 'win32') {
        pythonCommand = `conda run -n ${condaEnvName} python`;
      } 
      // On Unix-like systems
      else {
        // Try to find conda executable
        try {
          const condaPath = await this.getCondaPath();
          if (condaPath) {
            pythonCommand = `${condaPath} run -n ${condaEnvName} python`;
          }
        } catch (error) {
          console.warn('Unable to find conda. Using system Python.');
        }
      }
    }
    
    // Build the command
    const command = `${pythonCommand} ${scriptPath} ${args.join(' ')}`;
    
    // Execute the command
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          console.error(`Error executing Python script: ${error.message}`);
          console.error(`Command: ${command}`);
          if (stderr) console.error(`stderr: ${stderr}`);
          reject(error);
          return;
        }
        
        if (stderr) {
          console.warn(`Python script warning: ${stderr}`);
        }
        
        try {
          // Try to parse the stdout as JSON
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (parseError) {
          console.error('Failed to parse Python script output as JSON:', parseError);
          console.error('Raw output:', stdout);
          reject(new Error('Invalid output format from Python script'));
        }
      });
    });
  }
  
  /**
   * Try to find the conda executable path
   * @returns {Promise<string|null>} - Path to conda executable or null if not found
   * @private
   */
  async getCondaPath() {
    return new Promise((resolve) => {
      exec('which conda', (error, stdout) => {
        if (error || !stdout) {
          resolve(null);
          return;
        }
        
        resolve(stdout.trim());
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
    // Build arguments
    const args = [erraticId.toString()];
    
    if (updateDb) {
      args.push('--update-db');
    }
    
    // Run the script
    return this.runScript('classify_erratic.py', args);
  }
}

module.exports = new PythonService(); 