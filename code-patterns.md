# Glacial Erratics Map - Code Patterns & Conventions

This document outlines the code patterns and conventions used in the Glacial Erratics Map application to ensure consistency for future development.

## Backend Patterns

### Model Definitions

Models follow the Sequelize pattern with consistent structure:

```javascript
module.exports = (sequelize, DataTypes) => {
  const ModelName = sequelize.define('ModelName', {
    // Field definitions
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    // Other fields...
  }, {
    timestamps: true  // Includes createdAt and updatedAt
  });
  
  // Define associations
  ModelName.associate = function(models) {
    // Define relationships
  };
  
  return ModelName;
};
```

### Controllers

Controllers follow the RESTful pattern with consistent error handling:

```javascript
// Get all resources
exports.getAllResources = async (req, res) => {
  try {
    // Implementation
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Get single resource
exports.getResourceById = async (req, res) => {
  try {
    const { id } = req.params;
    // Implementation
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Create resource
exports.createResource = async (req, res) => {
  try {
    // Validation
    // Implementation
    res.status(201).json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Update resource
exports.updateResource = async (req, res) => {
  try {
    const { id } = req.params;
    // Implementation
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// Delete resource
exports.deleteResource = async (req, res) => {
  try {
    const { id } = req.params;
    // Implementation
    res.json({ message: 'Resource deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
```

### Routes

Routes are organized by resource with consistent structure:

```javascript
const express = require('express');
const router = express.Router();
const controller = require('../controllers/resourceController');
const { authenticateToken, requireAdmin } = require('../utils/auth');

// Public routes
router.get('/', controller.getAllResources);
router.get('/:id', controller.getResourceById);

// Protected admin routes
router.post('/', authenticateToken, requireAdmin, controller.createResource);
router.put('/:id', authenticateToken, requireAdmin, controller.updateResource);
router.delete('/:id', authenticateToken, requireAdmin, controller.deleteResource);

module.exports = router;
```

## Frontend Patterns

### React Components

Components follow a functional component pattern with hooks:

```jsx
import { useState, useEffect } from 'react';
import './ComponentName.css';

function ComponentName({ prop1, prop2 }) {
  const [state, setState] = useState(initialState);
  
  useEffect(() => {
    // Side effect handling
    return () => {
      // Cleanup
    };
  }, [dependencies]);
  
  // Helper functions
  const handleSomething = () => {
    // Implementation
  };
  
  return (
    <div className="component-name">
      {/* JSX structure */}
    </div>
  );
}

export default ComponentName;
```

### API Service Functions

Services encapsulate API calls with a consistent pattern:

```javascript
import axios from 'axios';

const API_URL = 'http://localhost:3001/api';

export const getResources = async () => {
  try {
    const response = await axios.get(`${API_URL}/resources`);
    return response.data;
  } catch (error) {
    console.error('Error fetching resources:', error);
    throw error;
  }
};

export const getResourceById = async (id) => {
  try {
    const response = await axios.get(`${API_URL}/resources/${id}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching resource ${id}:`, error);
    throw error;
  }
};

// Other CRUD operations...
```

### Custom Hooks

Custom hooks for reusing logic across components:

```javascript
import { useState, useEffect } from 'react';
import { getResources } from '../services/resourceService';

export const useResources = () => {
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchResources = async () => {
      try {
        setLoading(true);
        const data = await getResources();
        setResources(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchResources();
  }, []);
  
  return { resources, loading, error };
};
```

## Leaflet Map Implementation

The application uses React-Leaflet for map implementation with the following patterns:

```jsx
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function MapComponent() {
  return (
    <MapContainer 
      center={[initialLat, initialLng]} 
      zoom={initialZoom} 
      style={{ height: '100vh', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      
      {/* Markers */}
      {markers.map(marker => (
        <Marker 
          key={marker.id} 
          position={[marker.lat, marker.lng]}
        >
          <Popup>
            {/* Popup content */}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
```

## Error Handling

Error handling follows these patterns:

1. Backend: Try/catch blocks with appropriate error responses
2. Frontend: Try/catch blocks with error state management
3. User-friendly error messages in the UI

## Naming Conventions

1. Files:
   - React components: PascalCase.jsx
   - JavaScript utilities: camelCase.js
   - CSS files: ComponentName.css (matching component name)

2. Variables and functions:
   - camelCase for variables and functions
   - PascalCase for component names and classes

3. Database:
   - snake_case for database fields
   - PascalCase for model names

## Future Development - Spatial Analysis

For implementing the upcoming spatial analysis features from Spatial-Analysis.md:

1. Database extensions:
   - Follow the schema extensions mentioned in the document
   - Add vector support for ML embeddings

2. Python integration:
   - Use child_process for executing Python scripts
   - Standardize data exchange format (JSON)

3. ML implementation:
   - Separate ML logic into dedicated services
   - Use standardized model interfaces

4. UI components:
   - Create analysis-specific UI components
   - Follow consistent React patterns for analysis panels

These patterns should be followed for all future development to maintain code consistency and readability. 