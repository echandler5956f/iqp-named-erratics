/**
 * Mock for the Python service
 */

const mockAnalysisResults = {
  1: {
    "erratic_id": 1,
    "erratic_name": "Plymouth Rock",
    "location": {
      "longitude": -70.6619,
      "latitude": 41.958
    },
    "proximity_analysis": {
      "elevation_category": "lowland",
      "nearest_water_body_dist": 1200,
      "nearest_settlement_dist": 3500,
      "nearest_trails_dist": 800,
      "nearest_roads_dist": 1500
    }
  },
  2: {
    "erratic_id": 2,
    "erratic_name": "Madison Boulder",
    "location": {
      "longitude": -71.1503,
      "latitude": 43.9153
    },
    "proximity_analysis": {
      "elevation_category": "upland",
      "nearest_water_body_dist": 850,
      "nearest_settlement_dist": 2800,
      "nearest_trails_dist": 300,
      "nearest_roads_dist": 1200
    }
  }
};

class PythonServiceMock {
  /**
   * Mock for the runProximityAnalysis method
   */
  async runProximityAnalysis(erraticId, featureLayers = [], updateDb = false) {
    // Return mock data for known erratic IDs
    if (mockAnalysisResults[erraticId]) {
      return mockAnalysisResults[erraticId];
    }
    
    // Return error for unknown erratic IDs
    return {
      "error": `Erratic with ID ${erraticId} not found`
    };
  }
  
  /**
   * Mock for the runScript method
   */
  async runScript(scriptName, args = []) {
    // Return success for known scripts
    if (scriptName === 'proximity_analysis.py') {
      const erraticId = parseInt(args[0], 10);
      return this.runProximityAnalysis(erraticId);
    }
    
    // Return error for unknown scripts
    return {
      "error": `Script ${scriptName} not found or failed to execute`
    };
  }
}

module.exports = new PythonServiceMock(); 