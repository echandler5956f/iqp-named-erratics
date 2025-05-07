# Extending the North American Glacial Erratics Map with Machine Learning & Advanced Spatial Analysis

## Project Enhancement Overview

Building on the existing glacial erratics mapping application, implement advanced spatial analysis and machine learning features to categorize, analyze, and visualize patterns in the dataset. Focus on both qualitative categorization (cultural/historical usage) and quantitative spatial analysis.

IMPORTANT: ALL NAMED GLACIAL ERRATICS IN THE DATASET ARE GUARANTEED TO BE IN NORTH AMERICA (US and Canada). THIS IS AN ANALYSIS OF NORTH AMERICAN NAMED GLACIAL ERRATICS. WE ARE INTERESTED IN COLONIAL/NATIVE AMERICAN SIGNIFIGANCE AND HISTORICAL/CULTURAL CONNECTIONS.

## New Requirements

### 1. Data Schema Extensions

Extend the current database schema to support categorization and spatial analysis:

```sql
-- Add these columns to the erratics table
ALTER TABLE erratics 
ADD COLUMN usage_type VARCHAR(100)[],  -- Array of usage types (fort, meeting place, etc.)
ADD COLUMN cultural_significance_score INTEGER, -- Scale 1-10
ADD COLUMN has_inscriptions BOOLEAN,
ADD COLUMN accessibility_score INTEGER, -- Scale 1-5
ADD COLUMN size_category VARCHAR(50), -- Small, Medium, Large, Monumental
ADD COLUMN nearest_water_body_dist FLOAT,
ADD COLUMN nearest_settlement_dist FLOAT,
ADD COLUMN elevation_category VARCHAR(50),
ADD COLUMN geological_type VARCHAR(100),
ADD COLUMN estimated_displacement_dist FLOAT,
ADD COLUMN vector_embedding VECTOR(1536); -- For ML feature embedding
```

### 2. Qualitative Categorization System

Implement a comprehensive classification system for erratics based on historical and cultural usage:

1. Create a taxonomy of usage types:
   - Religious/Ceremonial
   - Navigational/Landmark
   - Meeting Place/Gathering
   - Boundary Marker
   - Military/Defensive
   - Trade/Commerce
   - Artistic/Inscribed
   - Mythological/Legendary
   - Scientific Interest

2. Build a classification module using NLP:
   - Extract key terms from descriptions
   - Apply Named Entity Recognition to identify locations, dates, and people
   - Use semantic similarity to match descriptions with predefined categories
   - Allow for multi-label classification (erratics can belong to multiple categories)

3. Create visualization tools for qualitative data:
   - Color-coding markers by primary usage
   - Multi-layered icons showing multiple uses
   - Timeline visualization of usage changes over time

### 3. Quantitative Spatial Analysis Features

Implement spatial analysis algorithms to analyze geographic relationships:

1. Proximity Analysis:
   - Calculate distances to nearest water bodies (rivers, lakes, ocean)
   - Measure proximity to historical settlement locations
   - Analyze relationship to colonial-era transportation routes

2. Terrain Analysis:
   - Extract elevation profiles
   - Calculate viewshed analysis (what can be seen from each erratic)
   - Determine prominence (how much an erratic stands out from surrounding terrain)

3. Clustering Analysis:
   - Implement HDBSCAN spatial clustering to identify groupings
   - Calculate Ripley's K function to analyze spatial distribution patterns
   - Identify anomalous erratics (spatial outliers)

### 4. Machine Learning Components

Develop ML models to enhance understanding of the dataset:

1. Supervised Learning:
   - Train a multi-label classifier to predict usage types based on location, size, and nearby features

2. Unsupervised Learning:
   - Apply k-means clustering on geographic and attribute data
   - Use hierarchical clustering to identify natural groupings
   - Implement UMAP or t-SNE for dimensionality reduction and visualization

3. Feature Importance Analysis:
   - Identify spatial patterns that correlate with specific usage types

### 5. Interactive Analysis Tools

Create interactive tools for users to perform their own analyses:

1. Custom Query Builder:
   - Allow users to filter erratics by multiple attributes
   - Support spatial queries (e.g., "show all erratics within 5km of water that were used as meeting places")

2. Comparison Tools:
   - Side-by-side comparison of erratics with similar attributes
   - Statistical summaries of different categories
   - Correlation matrix for various attributes

3. User-Defined Analysis:
   - Custom buffer zones around selected features
   - On-the-fly clustering with adjustable parameters
   - Downloadable analysis results in various formats (CSV, GeoJSON, shapefile)

### 6. UI/UX for Analysis Features

Design intuitive interfaces for accessing advanced features:

1. Analysis Panel:
   - Collapsible sidebar with analysis tools
   - Results visualization area
   - Save/load analysis configurations

2. Layer Controls:
   - Toggle between different categorization schemes
   - Adjust visualization parameters (size, color, opacity)
   - Show/hide analysis results

3. Dashboard Elements:
   - Key metrics and statistics
   - Distribution charts for major attributes
   - Top correlations and patterns detected

## Technical Implementation Approach

### Data Processing Pipeline

1. Create a Python-based ETL pipeline:
```python
# Example pipeline structure
class ErraticsAnalysisPipeline:
    def __init__(self, db_connection):
        self.db = db_connection
        self.nlp = spacy.load("en_core_web_lg")  # For text analysis
        
    def extract_features(self, text):
        """Extract key features from descriptive text"""
        doc = self.nlp(text)
        # Extract entities, keywords, etc.
        
    def calculate_spatial_metrics(self, lat, lng):
        """Calculate distances to key features"""
        # Use PostGIS for spatial calculations
        
    def classify_usage(self, description):
        """Classify erratic usage based on description"""
        # Apply NLP classification
        
    def cluster_erratics(self, algorithm="dbscan", params={}):
        """Cluster erratics by location and attributes"""
        # Implement various clustering algorithms
```

2. Implement NLP-based classification:
```python
# Example classification system
def classify_erratic_usage(description):
    # Load pre-trained model
    classifier = ErraticClassifier.load("models/usage_classifier.pkl")
    
    # Preprocess text
    tokens = preprocess_text(description)
    
    # Generate predictions
    predictions = classifier.predict(tokens)
    
    # Return top categories with confidence scores
    return [
        {"category": cat, "confidence": score} 
        for cat, score in predictions if score > 0.4
    ]
```

3. Create spatial analysis functions:
```python
# Example spatial analysis
def proximity_analysis(erratic_id, feature_layers=["rivers", "settlements"]):
    """Calculate distances to nearest features"""
    
    results = {}
    for layer in feature_layers:
        query = f"""
            SELECT ST_Distance(
                e.location, 
                ST_ClosestPoint(f.geom, e.location)
            ) as distance,
            f.name
            FROM erratics e, {layer} f
            WHERE e.id = %s
            ORDER BY distance ASC
            LIMIT 1
        """
        results[layer] = execute_query(query, [erratic_id])
        
    return results
```

### Frontend Implementation

1. Create analysis components:
```jsx
// Example React component for analysis controls
function SpatialAnalysisControls({ onRunAnalysis }) {
  const [analysisType, setAnalysisType] = useState('proximity');
  const [parameters, setParameters] = useState({
    radius: 5000, // meters
    features: ['rivers', 'settlements'],
    clusteringAlgorithm: 'dbscan',
    eps: 0.5,
    minSamples: 5
  });
  
  return (
    <div className="analysis-controls">
      <h3>Spatial Analysis</h3>
      
      <select value={analysisType} onChange={e => setAnalysisType(e.target.value)}>
        <option value="proximity">Proximity Analysis</option>
        <option value="clustering">Clustering</option>
        <option value="terrain">Terrain Analysis</option>
      </select>
      
      {analysisType === 'clustering' && (
        <div className="clustering-options">
          <label>
            Algorithm:
            <select 
              value={parameters.clusteringAlgorithm}
              onChange={e => setParameters({...parameters, clusteringAlgorithm: e.target.value})}
            >
              <option value="dbscan">HDBSCAN</option>
              <option value="kmeans">K-Means</option>
              <option value="hierarchical">Hierarchical</option>
            </select>
          </label>
          {/* Additional parameters */}
        </div>
      )}
      
      <button onClick={() => onRunAnalysis(analysisType, parameters)}>
        Run Analysis
      </button>
    </div>
  );
}
```

2. Implement visualization components:
```jsx
// Example visualization component
function AnalysisResults({ results, analysisType }) {
  if (!results) return null;
  
  return (
    <div className="analysis-results">
      <h3>Analysis Results</h3>
      
      {analysisType === 'clustering' && (
        <div>
          <p>Found {results.clusters.length} clusters</p>
          <ul>
            {results.clusters.map((cluster, i) => (
              <li key={i}>
                Cluster {i+1}: {cluster.members.length} erratics
                <button onClick={() => highlightCluster(cluster.id)}>
                  Highlight
                </button>
              </li>
            ))}
          </ul>
          <p>Silhouette score: {results.silhouetteScore.toFixed(2)}</p>
        </div>
      )}
      
      {/* Other result types */}
    </div>
  );
}
```

### Backend API Endpoints

Add these new endpoints to support analysis features:

```
GET /api/analysis/proximity/:id - Get proximity analysis for specified erratic
POST /api/analysis/cluster - Perform clustering analysis with specified parameters
GET /api/analysis/terrain/:id - Get terrain analysis for specified erratic
POST /api/erratics/classify - Classify erratics based on descriptions
GET /api/statistics/summary - Get statistical summary of the dataset
POST /api/analysis/custom - Run custom analysis with specified parameters
```

## ML Model Training Approach

1. Feature Extraction Pipeline:

```python
def extract_features(erratics_dataframe):
    """Extract features for machine learning models"""
    features = pd.DataFrame()
    
    # Geographic features
    features['lat'] = erratics_dataframe['location'].apply(lambda p: p.y)
    features['lng'] = erratics_dataframe['location'].apply(lambda p: p.x)
    features['elevation'] = erratics_dataframe['elevation']
    features['coast_dist'] = erratics_dataframe['nearest_water_body_dist']
    features['settlement_dist'] = erratics_dataframe['nearest_settlement_dist']
    
    # Physical features
    features['size_numeric'] = erratics_dataframe['size_meters']
    features['has_inscriptions'] = erratics_dataframe['has_inscriptions'].astype(int)
    
    # Text-based features using TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    text_features = tfidf.fit_transform(erratics_dataframe['description'])
    text_df = pd.DataFrame(text_features.toarray(), 
                          columns=[f'text_{i}' for i in range(text_features.shape[1])])
    
    # Combine all features
    features = pd.concat([features, text_df], axis=1)
    
    return features
```

2. Model Training Script:

```python
def train_usage_classifier(X, y):
    """Train a multi-label classifier for usage types"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Define model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2
        )))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    # Save model
    joblib.dump(pipeline, 'models/usage_classifier.pkl')
    
    return pipeline, report
```

3. Clustering Implementation:

```python
def cluster_erratics(features_df, algorithm='dbscan', eps=0.5, min_samples=5):
    """Perform spatial clustering on erratics"""
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)
    
    if algorithm == 'dbscan':
        # Apply DBSCAN
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(X)
        
        # Calculate silhouette score (ignoring noise points marked as -1)
        if len(set(labels)) > 1 and -1 not in labels:
            silhouette_avg = silhouette_score(X, labels)
        else:
            # Filter out noise points for silhouette calculation
            mask = labels != -1
            if sum(mask) > 1 and len(set(labels[mask])) > 1:
                silhouette_avg = silhouette_score(X[mask], labels[mask])
            else:
                silhouette_avg = 0
                
    elif algorithm == 'kmeans':
        # Determine optimal k using elbow method
        wcss = []
        k_range = range(2, min(15, len(features_df) // 10 + 1))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            
        # Find elbow point
        k = find_elbow_point(wcss)
        
        # Apply K-means with optimal k
        clusterer = KMeans(n_clusters=k, random_state=42)
        labels = clusterer.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, labels)
        
    return {
        'labels': labels,
        'silhouette_score': silhouette_avg,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
    }
```

## Integration with Existing Application

1. Update the database schema first
2. Create the analysis backend modules
3. Implement ML model training scripts
4. Add analysis endpoints to the API
5. Extend the frontend with analysis components
6. Update the map visualization to show analysis results

## Example Sequence

1. First, import required Python packages:
```python
# Install with pip:
# pip install scikit-learn geopandas spacy tensorflow pandas nltk shapely rtree

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import spacy
import joblib
import tensorflow as tf
```

2. Update the database schema:
```sql
-- Create the extensions needed
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- Add new columns
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS usage_type VARCHAR(100)[];
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS cultural_significance_score INTEGER;
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS has_inscriptions BOOLEAN DEFAULT false;
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS accessibility_score INTEGER;
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS size_category VARCHAR(50);
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS nearest_water_body_dist FLOAT;
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS nearest_settlement_dist FLOAT;
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS elevation_category VARCHAR(50);
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS geological_type VARCHAR(100);
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS estimated_displacement_dist FLOAT;
ALTER TABLE erratics ADD COLUMN IF NOT EXISTS vector_embedding VECTOR(1536);
```

3. Implement a spatial analysis API in the backend:
```javascript
// /backend/src/controllers/analysisController.js
const db = require('../config/database');
const { exec } = require('child_process');
const path = require('path');

// Run Python script for complex analysis
const runPythonAnalysis = (scriptName, args) => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '../scripts', scriptName);
    const process = exec(
      `python ${scriptPath} ${args.join(' ')}`,
      (error, stdout, stderr) => {
        if (error) {
          reject(error);
          return;
        }
        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${stdout}`));
        }
      }
    );
  });
};

exports.getProximityAnalysis = async (req, res) => {
  try {
    const { id } = req.params;
    const { features } = req.query;
    
    const featuresList = features ? features.split(',') : ['water', 'settlements'];
    
    const result = await runPythonAnalysis('proximity_analysis.py', [id, ...featuresList]);
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.performClustering = async (req, res) => {
  try {
    const { algorithm, eps, minSamples, k, features } = req.body;
    
    const result = await runPythonAnalysis('clustering.py', [
      algorithm || 'dbscan',
      eps || 0.5,
      minSamples || 5,
      k || 5,
      features ? features.join(',') : 'location,elevation'
    ]);
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.classifyErratic = async (req, res) => {
  try {
    const { id, description } = req.body;
    
    const result = await runPythonAnalysis('classify_erratic.py', [
      id || 0,
      Buffer.from(description || '').toString('base64')
    ]);
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```

4. Create analysis routes:
```javascript
// /backend/src/routes/analysisRoutes.js
const express = require('express');
const analysisController = require('../controllers/analysisController');

const router = express.Router();

router.get('/proximity/:id', analysisController.getProximityAnalysis);
router.post('/cluster', analysisController.performClustering);
router.post('/classify', analysisController.classifyErratic);

module.exports = router;
```

5. Update the frontend with an analysis panel:
```jsx
// /frontend/src/components/AnalysisPanel.jsx
import React, { useState } from 'react';
import { Tabs, Tab, Box, Typography, Slider, FormControl, InputLabel, Select, MenuItem, Button } from '@mui/material';
import { useAnalysis } from '../hooks/useAnalysis';

function AnalysisPanel({ selectedErratic, onAnalysisComplete }) {
  const [activeTab, setActiveTab] = useState(0);
  const [analysisParams, setAnalysisParams] = useState({
    clusteringAlgorithm: 'dbscan',
    eps: 0.5,
    minSamples: 5,
    k: 5,
    features: ['location', 'elevation']
  });
  
  const { runProximityAnalysis, runClustering, classifyErratic, loading, error } = useAnalysis();
  
  const handleRunAnalysis = async () => {
    if (activeTab === 0 && selectedErratic) {
      const result = await runProximityAnalysis(selectedErratic.id);
      onAnalysisComplete(result, 'proximity');
    } else if (activeTab === 1) {
      const result = await runClustering(analysisParams);
      onAnalysisComplete(result, 'clustering');
    } else if (activeTab === 2 && selectedErratic) {
      const result = await classifyErratic(selectedErratic.id, selectedErratic.description);
      onAnalysisComplete(result, 'classification');
    }
  };
  
  return (
    <Box sx={{ width: '100%', bgcolor: 'background.paper' }}>
      <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
        <Tab label="Proximity" />
        <Tab label="Clustering" />
        <Tab label="Classification" />
      </Tabs>
      
      <Box sx={{ p: 3 }}>
        {activeTab === 0 && (
          <div>
            <Typography variant="h6">Proximity Analysis</Typography>
            <Typography variant="body2">
              Calculate distances from selected erratic to geographical features.
            </Typography>
            {!selectedErratic && (
              <Typography color="error">
                Please select an erratic on the map first.
              </Typography>
            )}
          </div>
        )}
        
        {activeTab === 1 && (
          <div>
            <Typography variant="h6">Clustering Analysis</Typography>
            
            <FormControl fullWidth margin="normal">
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={analysisParams.clusteringAlgorithm}
                onChange={(e) => setAnalysisParams({
                  ...analysisParams,
                  clusteringAlgorithm: e.target.value
                })}
              >
                <MenuItem value="dbscan">DBSCAN</MenuItem>
                <MenuItem value="kmeans">K-Means</MenuItem>
              </Select>
            </FormControl>
            
            {analysisParams.clusteringAlgorithm === 'dbscan' && (
              <>
                <Typography gutterBottom>Epsilon (distance parameter)</Typography>
                <Slider
                  value={analysisParams.eps}
                  onChange={(e, value) => setAnalysisParams({
                    ...analysisParams,
                    eps: value
                  })}
                  min={0.1}
                  max={2}
                  step={0.1}
                  valueLabelDisplay="auto"
                />
                
                <Typography gutterBottom>Minimum Samples</Typography>
                <Slider
                  value={analysisParams.minSamples}
                  onChange={(e, value) => setAnalysisParams({
                    ...analysisParams,
                    minSamples: value
                  })}
                  min={2}
                  max={20}
                  step={1}
                  valueLabelDisplay="auto"
                />
              </>
            )}
            
            {analysisParams.clusteringAlgorithm === 'kmeans' && (
              <>
                <Typography gutterBottom>Number of Clusters (K)</Typography>
                <Slider
                  value={analysisParams.k}
                  onChange={(e, value) => setAnalysisParams({
                    ...analysisParams,
                    k: value
                  })}
                  min={2}
                  max={15}
                  step={1}
                  valueLabelDisplay="auto"
                />
              </>
            )}
          </div>
        )}
        
        {activeTab === 2 && (
          <div>
            <Typography variant="h6">Classification</Typography>
            <Typography variant="body2">
              Analyze description text to classify erratic usage and significance.
            </Typography>
            {!selectedErratic && (
              <Typography color="error">
                Please select an erratic on the map first.
              </Typography>
            )}
          </div>
        )}
        
        <Button 
          variant="contained" 
          onClick={handleRunAnalysis}
          disabled={loading || (activeTab !== 1 && !selectedErratic)}
          sx={{ mt: 2 }}
        >
          {loading ? 'Running...' : 'Run Analysis'}
        </Button>
        
        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}
      </Box>
    </Box>
  );
}

export default AnalysisPanel;
```

Proceed with implementation by focusing on one module at a time, starting with the database schema updates and core spatial analysis functions. Use Python for the complex data processing and ML tasks, and integrate the results with the Node.js backend using either subprocess calls or a simple API between the two components.