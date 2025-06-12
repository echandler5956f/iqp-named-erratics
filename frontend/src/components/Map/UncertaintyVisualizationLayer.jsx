import React, { useMemo } from 'react';
import { Circle, Polygon, Polyline } from 'react-leaflet';
import './UncertaintyVisualizationLayer.css';

const UncertaintyVisualizationLayer = ({ erratics, isVisible }) => {
  // Create comprehensive data uncertainty visualizations
  const uncertaintyElements = useMemo(() => {
    if (!erratics || erratics.length === 0) return { 
      knownData: [], 
      uncertaintyFields: [], 
      knowledgeGaps: [],
      confidenceConnections: []
    };

    console.log('UncertaintyVisualizationLayer: Processing', erratics.length, 'erratics');

    // Analyze data completeness for all erratics
    const erraticsWithUncertainty = erratics
      .filter(e => e.location && e.location.coordinates)
      .map(erratic => ({
        ...erratic,
        uncertainty: calculateComprehensiveUncertainty(erratic)
      }));

    // Classify erratics by overall data quality
    const highQualityErratics = erraticsWithUncertainty.filter(e => e.uncertainty.overallScore >= 0.7);
    const mediumQualityErratics = erraticsWithUncertainty.filter(e => e.uncertainty.overallScore >= 0.4 && e.uncertainty.overallScore < 0.7);
    const lowQualityErratics = erraticsWithUncertainty.filter(e => e.uncertainty.overallScore < 0.4);

    console.log('UncertaintyVisualizationLayer: High quality:', highQualityErratics.length, 'Medium:', mediumQualityErratics.length, 'Low:', lowQualityErratics.length);

    // Create knowledge gaps around clusters of low-quality data
    const knowledgeGaps = createDataQualityGaps(lowQualityErratics, highQualityErratics);
    
    // Remove uncertainty fields - they're redundant with data quality circles
    const uncertaintyFields = [];

    // Create confidence connections between high-quality erratics
    const confidenceConnections = createDataQualityConnections(highQualityErratics);

    // Show data quality indicators for ALL erratics (this is the main visualization)
    const dataQualityPoints = erraticsWithUncertainty.map(erratic => {
      const [lng, lat] = erratic.location.coordinates;
      
      return {
        id: `quality-${erratic.id}`,
        position: [lat, lng],
        uncertainty: erratic.uncertainty,
        erratic: erratic
      };
    });

    console.log('UncertaintyVisualizationLayer: Created', knowledgeGaps.length, 'knowledge gaps,', uncertaintyFields.length, 'uncertainty fields,', confidenceConnections.length, 'connections,', dataQualityPoints.length, 'quality points');

    return {
      knownData: dataQualityPoints,
      uncertaintyFields: uncertaintyFields,
      knowledgeGaps: knowledgeGaps,
      confidenceConnections: confidenceConnections
    };
  }, [erratics]);

  if (!isVisible) return null;

  console.log('UncertaintyVisualizationLayer: Rendering comprehensive data uncertainty visualization');

  return (
    <>
      {/* Data Uncertainty Legend */}
      <DataUncertaintyLegend />
      
      {/* Data quality indicators - circles showing overall data completeness */}
      {uncertaintyElements.knownData.map((point) => (
        <Circle
          key={point.id}
          center={point.position}
          radius={point.uncertainty.qualityRadius}
          pathOptions={{
            color: point.uncertainty.qualityColor,
            fillColor: point.uncertainty.qualityColor,
            fillOpacity: 0.5,
            weight: 2,
            opacity: 0.9
          }}
        />
      ))}

      {/* Confidence connections - lines between high-quality erratics */}
      {uncertaintyElements.confidenceConnections.map((connection) => (
        <Polyline
          key={connection.id}
          positions={connection.path}
          pathOptions={{
            color: connection.color,
            weight: connection.weight,
            opacity: connection.opacity,
            dashArray: connection.dashArray,
            lineCap: 'round',
            lineJoin: 'round'
          }}
        />
      ))}

      {/* Removed redundant uncertainty fields - data quality circles serve this purpose */}

      {/* Knowledge gaps - areas with clusters of low-quality data */}
      {uncertaintyElements.knowledgeGaps.map((gap) => (
        <Polygon
          key={gap.id}
          positions={gap.polygon}
          pathOptions={{
            color: '#e74c3c', // Red border
            fillColor: '#c0392b', // Darker red fill
            fillOpacity: 0.25 + (gap.severity * 0.25), // 0.25-0.5 opacity
            weight: 3,
            opacity: 0.8,
            dashArray: '10, 6'
          }}
        />
      ))}
    </>
  );
};

// Calculate comprehensive uncertainty based on all available data
function calculateComprehensiveUncertainty(erratic) {
  let totalScore = 0;
  let maxScore = 0;
  const missingFields = [];
  const presentFields = [];

  // 1. Location data (weight: 3 - critical)
  maxScore += 3;
  if (erratic.location && erratic.location.coordinates) {
    totalScore += 3;
    presentFields.push('location');
  } else {
    missingFields.push('location');
  }

  // 2. Rock type identification (weight: 3 - critical)
  maxScore += 3;
  if (erratic.rock_type && !erratic.rock_type.toLowerCase().includes('placeholder') && erratic.rock_type.trim() !== '') {
    totalScore += 3;
    presentFields.push('rock_type');
  } else {
    missingFields.push('rock_type');
  }

  // 3. Size measurements (weight: 2 - important)
  maxScore += 2;
  if (erratic.size_meters && erratic.size_meters > 0) {
    totalScore += 2;
    presentFields.push('size');
  } else {
    missingFields.push('size');
  }

  // 4. Age estimation (weight: 2 - important)
  maxScore += 2;
  if (erratic.estimated_age && !erratic.estimated_age.toLowerCase().includes('placeholder') && erratic.estimated_age.trim() !== '') {
    totalScore += 2;
    presentFields.push('age');
  } else {
    missingFields.push('age');
  }

  // 5. Description quality (weight: 2 - important)
  maxScore += 2;
  if (erratic.description && erratic.description.length > 50) {
    if (erratic.description.length > 200) {
      totalScore += 2; // Detailed description
    } else {
      totalScore += 1; // Basic description
    }
    presentFields.push('description');
  } else {
    missingFields.push('description');
  }

  // 6. Cultural significance (weight: 1 - valuable)
  maxScore += 1;
  if (erratic.cultural_significance && erratic.cultural_significance.length > 20) {
    totalScore += 1;
    presentFields.push('cultural_significance');
  } else {
    missingFields.push('cultural_significance');
  }

  // 7. Visual documentation (weight: 1 - valuable)
  maxScore += 1;
  if (erratic.image_url && erratic.image_url.trim() !== '' && !erratic.image_url.includes('placeholder')) {
    totalScore += 1;
    presentFields.push('image');
  } else {
    missingFields.push('image');
  }

  // 8. Discovery information (weight: 1 - valuable)
  maxScore += 1;
  if (erratic.discovery_year && erratic.discovery_year > 1800) {
    totalScore += 1;
    presentFields.push('discovery_info');
  } else {
    missingFields.push('discovery_info');
  }

  const overallScore = totalScore / maxScore;
  const uncertaintyLevel = 1 - overallScore;

  // Determine uncertainty visualization properties
  let color, radius, qualityColor, qualityRadius;
  
  if (overallScore >= 0.8) {
    color = '#3498db'; // Green - excellent data
    qualityColor = '#3498db';
    radius = 10000;
    qualityRadius = 10000; // Largest circles for best data
  } else if (overallScore >= 0.5) {
    color = '#27ae60'; // Blue - good data
    qualityColor = '#27ae60';
    radius = 7000;
    qualityRadius = 7000;
  } else if (overallScore >= 0.4) {
    color = '#f39c12'; // Orange - fair data
    qualityColor = '#f39c12';
    radius = 5000;
    qualityRadius = 5000;
  } else if (overallScore >= 0.34) {
    color = '#e67e22'; // Dark orange - poor data
    qualityColor = '#e67e22';
    radius = 3000;
    qualityRadius = 3000;
  } else {
    color = '#e74c3c'; // Red - very poor data
    qualityColor = '#e74c3c';
    radius = 1500;
    qualityRadius = 1500; // Smallest circles for worst data
  }

  return {
    overallScore: overallScore,
    uncertaintyLevel: uncertaintyLevel,
    missingFields: missingFields,
    presentFields: presentFields,
    missingCount: missingFields.length,
    presentCount: presentFields.length,
    color: color,
    radius: radius,
    qualityColor: qualityColor,
    qualityRadius: qualityRadius,
    dataCompleteness: `${presentFields.length}/${presentFields.length + missingFields.length} fields`
  };
}

// Create knowledge gaps around clusters of low-quality data
function createDataQualityGaps(lowQualityErratics, highQualityErratics) {
  if (lowQualityErratics.length === 0) return [];

  const gaps = [];
  const processedErratics = new Set();
  
  // Find clusters of low-quality erratics
  lowQualityErratics.forEach((erratic, index) => {
    if (processedErratics.has(erratic.id)) return;
    
    const [lng, lat] = erratic.location.coordinates;
    
    // Find nearby low-quality erratics (within 80km)
    const nearbyLowQuality = lowQualityErratics.filter(other => {
      if (other.id === erratic.id || processedErratics.has(other.id)) return false;
      const [otherLng, otherLat] = other.location.coordinates;
      const distance = calculateDistance(lat, lng, otherLat, otherLng);
      return distance < 80; // 80km clustering radius
    });
    
    // Only create gap if there are multiple low-quality erratics clustered together
    if (nearbyLowQuality.length >= 1) { // Lower threshold since we want to show data quality issues
      const clusterErratics = [erratic, ...nearbyLowQuality];
      
      // Mark as processed
      clusterErratics.forEach(e => processedErratics.add(e.id));
      
      // Calculate cluster bounds
      const bounds = calculateClusterBounds(clusterErratics);
      
      // Check isolation from high-quality data
      const nearestHighQualityDistance = findNearestHighQualityDistance(bounds.center, highQualityErratics);
      
      // Show gap if cluster has significant data quality issues
      if (nearestHighQualityDistance > 30 || clusterErratics.length >= 3) { // 30km isolation or 3+ erratics
        const avgUncertainty = clusterErratics.reduce((sum, e) => sum + e.uncertainty.uncertaintyLevel, 0) / clusterErratics.length;
        const polygon = createClusterPolygon(clusterErratics, 20); // 20km buffer
        
        gaps.push({
          id: `data-quality-gap-${index}`,
          polygon: polygon,
          severity: avgUncertainty,
          erraticCount: clusterErratics.length,
          avgDataCompleteness: avgUncertainty,
          isolation: nearestHighQualityDistance
        });
      }
    }
  });
  
  console.log('UncertaintyVisualizationLayer: Created', gaps.length, 'data quality gaps');
  return gaps;
}

// Find distance to nearest high-quality erratic
function findNearestHighQualityDistance(center, highQualityErratics) {
  if (highQualityErratics.length === 0) return Infinity;
  
  const [centerLat, centerLng] = center;
  let minDistance = Infinity;
  
  highQualityErratics.forEach(erratic => {
    const [lng, lat] = erratic.location.coordinates;
    const distance = calculateDistance(centerLat, centerLng, lat, lng);
    minDistance = Math.min(minDistance, distance);
  });
  
  return minDistance;
}

// Create confidence connections between high-quality erratics
function createDataQualityConnections(highQualityErratics) {
  if (highQualityErratics.length < 2) return [];
  
  const connections = [];
  const maxErratics = Math.min(highQualityErratics.length, 20);
  
  // Create connections between nearby high-quality erratics
  for (let i = 0; i < maxErratics; i++) {
    for (let j = i + 1; j < maxErratics; j++) {
      const erratic1 = highQualityErratics[i];
      const erratic2 = highQualityErratics[j];
      
      const [lng1, lat1] = erratic1.location.coordinates;
      const [lng2, lat2] = erratic2.location.coordinates;
      
      const distance = calculateDistance(lat1, lng1, lat2, lng2);
      
      // Connect nearby high-quality erratics
      if (distance < 400) { // 400km threshold
        const dataQuality1 = erratic1.uncertainty.overallScore;
        const dataQuality2 = erratic2.uncertainty.overallScore;
        const avgQuality = (dataQuality1 + dataQuality2) / 2;
        
        // Also consider geological similarity if both have rock type data
        let geologicalSimilarity = 0.5; // Default
        if (erratic1.rock_type && erratic2.rock_type && 
            !erratic1.rock_type.includes('placeholder') && 
            !erratic2.rock_type.includes('placeholder')) {
          const rockType1 = normalizeRockType(erratic1.rock_type);
          const rockType2 = normalizeRockType(erratic2.rock_type);
          geologicalSimilarity = rockType1 === rockType2 ? 1.0 : 0.3;
        }
        
        const confidence = Math.exp(-distance / 180) * avgQuality * (0.7 + 0.3 * geologicalSimilarity);
        
        if (confidence > 0.15) {
          let color;
          if (avgQuality > 0.8 && geologicalSimilarity > 0.8) {
            color = '#3498db'; // Green for high quality + same rock type
          } else if (avgQuality > 0.7) {
            color = '#27ae60'; // Blue for high quality
          } else {
            color = '#f39c12'; // Orange for medium quality
          }
          
          connections.push({
            id: `quality-connection-${erratic1.id}-${erratic2.id}`,
            path: [[lat1, lng1], [lat2, lng2]],
            color: color,
            weight: Math.max(2, confidence * 6),
            opacity: Math.max(0.4, confidence * 0.9),
            dashArray: geologicalSimilarity > 0.8 ? null : '8, 4',
            confidence: confidence,
            dataQuality: avgQuality
          });
        }
      }
    }
  }
  
  console.log('UncertaintyVisualizationLayer: Created', connections.length, 'data quality connections');
  return connections;
}

// Data Uncertainty Legend Component
const DataUncertaintyLegend = () => {
  return (
    <div className="uncertainty-legend">
      <div className="uncertainty-legend-header">
        <h3>Data Completeness Guide</h3>
        <p>Understanding database quality and research gaps</p>
      </div>

      {/* Research Priority Areas */}
            <div className="legend-section">
        <h4>Research Priority Areas</h4>
        <div className="legend-item">
          <div className="legend-symbol knowledge-gap"></div>
          <div className="legend-text">
            <strong>Data Quality Gaps</strong>
            <span>Red polygons - clusters of incomplete records</span>
          </div>
        </div>
      </div>
      
      <div className="uncertainty-legend-content">
        {/* Data Quality Circles - Main Visualization */}
        <div className="legend-section">
          <h4>Data Quality Circles</h4>
          <div className="legend-item">
            <div className="legend-symbol quality-excellent"></div>
            <div className="legend-text">
              <strong>Excellent (80%+)</strong>
              <span>Blue circles - complete data fields</span>
            </div>
          </div>
          <div className="legend-item">
            <div className="legend-symbol quality-good"></div>
            <div className="legend-text">
              <strong>Good (60-80%)</strong>
              <span>Green circles - most data present</span>
            </div>
          </div>
          <div className="legend-item">
            <div className="legend-symbol quality-fair"></div>
            <div className="legend-text">
              <strong>Fair (40-60%)</strong>
              <span>Orange circles - some data gaps</span>
            </div>
          </div>
          <div className="legend-item">
            <div className="legend-symbol quality-poor"></div>
            <div className="legend-text">
              <strong>Poor (20-40%)</strong>
              <span>Dark orange circles - major data gaps</span>
            </div>
          </div>
          <div className="legend-item">
            <div className="legend-symbol quality-very-poor"></div>
            <div className="legend-text">
              <strong>Very Poor (&lt;20%)</strong>
              <span>Red circles - minimal data available</span>
            </div>
          </div>
        </div>

        {/* Data Quality Connections */}
        <div className="legend-section">
          <h4>Data Relationship Lines</h4>
          <div className="legend-item">
            <div className="legend-symbol connection-high"></div>
            <div className="legend-text">
              <strong>Blue Solid Lines</strong>
              <span>High-quality erratics, same rock type</span>
            </div>
          </div>
          <div className="legend-item">
            <div className="legend-symbol connection-medium"></div>
            <div className="legend-text">
              <strong>Green Dashed Lines</strong>
              <span>High-quality erratics, different rock types</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Calculate bounds for a cluster of erratics
function calculateClusterBounds(erratics) {
  let minLat = Infinity, maxLat = -Infinity;
  let minLng = Infinity, maxLng = -Infinity;
  
  erratics.forEach(erratic => {
    const [lng, lat] = erratic.location.coordinates;
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
    minLng = Math.min(minLng, lng);
    maxLng = Math.max(maxLng, lng);
  });
  
  return {
    minLat, maxLat, minLng, maxLng,
    center: [(minLat + maxLat) / 2, (minLng + maxLng) / 2]
  };
}

// Create polygon around a cluster of erratics
function createClusterPolygon(erratics, bufferKm) {
  const bounds = calculateClusterBounds(erratics);
  const centerLat = bounds.center[0];
  const centerLng = bounds.center[1];
  
  // Convert buffer from km to degrees (rough approximation)
  const bufferDegrees = bufferKm / 111; // ~111km per degree
  
  const points = [];
  const numPoints = 8;
  
  for (let i = 0; i < numPoints; i++) {
    const angle = (i / numPoints) * 2 * Math.PI;
    const radiusLat = (bounds.maxLat - bounds.minLat) / 2 + bufferDegrees;
    const radiusLng = (bounds.maxLng - bounds.minLng) / 2 + bufferDegrees;
    
    const latOffset = Math.sin(angle) * radiusLat;
    const lngOffset = Math.cos(angle) * radiusLng;
    
    points.push([centerLat + latOffset, centerLng + lngOffset]);
  }
  
  // Close the polygon
  points.push(points[0]);
  
  return points;
}

// Utility functions
function calculateBounds(erratics) {
  let minLat = Infinity, maxLat = -Infinity;
  let minLng = Infinity, maxLng = -Infinity;
  
  erratics.forEach(erratic => {
    if (erratic.location && erratic.location.coordinates) {
      const [lng, lat] = erratic.location.coordinates;
      minLat = Math.min(minLat, lat);
      maxLat = Math.max(maxLat, lat);
      minLng = Math.min(minLng, lng);
      maxLng = Math.max(maxLng, lng);
    }
  });
  
  return { minLat, maxLat, minLng, maxLng };
}

function calculateDistance(lat1, lng1, lat2, lng2) {
  const R = 6371; // Earth's radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLng/2) * Math.sin(dLng/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

function normalizeRockType(rockType) {
  if (!rockType || typeof rockType !== 'string') return 'unknown';
  
  const normalized = rockType.toLowerCase().trim();
  
  if (normalized.includes('granite')) return 'granite';
  if (normalized.includes('quartzite')) return 'quartzite';
  if (normalized.includes('gneiss')) return 'gneiss';
  if (normalized.includes('sandstone')) return 'sandstone';
  if (normalized.includes('limestone')) return 'limestone';
  if (normalized.includes('basalt')) return 'basalt';
  if (normalized.includes('schist')) return 'schist';
  if (normalized.includes('conglomerate')) return 'conglomerate';
  if (normalized.includes('greenstone')) return 'greenstone';
  
  return 'unknown';
}

export default UncertaintyVisualizationLayer; 