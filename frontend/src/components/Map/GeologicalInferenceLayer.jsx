import React, { useMemo } from 'react';
import { Polygon, Circle } from 'react-leaflet';

const GeologicalInferenceLayer = ({ erratics, isVisible }) => {
  // Advanced geological inference based on spatial patterns and known data
  const inferenceVisualization = useMemo(() => {
    if (!erratics || erratics.length === 0) return { zones: [], confidence: [] };

    console.log('GeologicalInferenceLayer: Processing', erratics.length, 'erratics');

    // Separate known vs unknown rock types
    const knownErratics = erratics.filter(e => 
      e.rock_type && 
      !e.rock_type.toLowerCase().includes('placeholder') &&
      e.location && e.location.coordinates
    );
    
    const unknownErratics = erratics.filter(e => 
      (!e.rock_type || e.rock_type.toLowerCase().includes('placeholder')) &&
      e.location && e.location.coordinates
    );

    console.log('GeologicalInferenceLayer: Known erratics:', knownErratics.length, 'Unknown:', unknownErratics.length);

    if (knownErratics.length === 0) return { zones: [], confidence: [] };

    // Geological source regions based on North American geology
    const sourceRegions = {
      'Canadian Shield': {
        center: [55.0, -85.0],
        radius: 800, // km
        dominantRocks: ['granite', 'gneiss', 'greenstone', 'metamorphic'],
        color: '#8B4513'
      },
      'Appalachian': {
        center: [44.0, -71.0], 
        radius: 400,
        dominantRocks: ['schist', 'gneiss', 'quartzite'],
        color: '#2F4F4F'
      },
      'Great Lakes Basin': {
        center: [46.0, -84.0],
        radius: 300,
        dominantRocks: ['sandstone', 'limestone', 'conglomerate'],
        color: '#CD853F'
      },
      'Rocky Mountain': {
        center: [51.0, -115.0],
        radius: 500,
        dominantRocks: ['quartzite', 'limestone', 'granite'],
        color: '#696969'
      }
    };

    // Create geological probability zones - MUCH MORE VISIBLE
    const zones = Object.entries(sourceRegions).map(([name, region]) => {
      const influencePolygon = createInfluenceZone(region, knownErratics, erratics);
      
      return {
        id: `geo-zone-${name}`,
        name: name,
        polygon: influencePolygon,
        dominantRocks: region.dominantRocks,
        color: region.color,
        confidence: calculateZoneConfidence(region, knownErratics)
      };
    });

    // Create MUCH MORE VISIBLE confidence indicators for unknown erratics
    const confidenceIndicators = unknownErratics.slice(0, 50).map(erratic => { // Limit to first 50 for performance
      const [lng, lat] = erratic.location.coordinates;
      const prediction = predictRockType(erratic, knownErratics, sourceRegions);
      
      return {
        id: `confidence-${erratic.id}`,
        position: [lat, lng],
        prediction: prediction.rockType,
        confidence: prediction.confidence,
        color: getRockTypeColor(prediction.rockType),
        erratic: erratic
      };
    });

    console.log('GeologicalInferenceLayer: Created', zones.length, 'zones and', confidenceIndicators.length, 'confidence indicators');

    return { zones, confidence: confidenceIndicators };
  }, [erratics]);

  if (!isVisible) return null;

  console.log('GeologicalInferenceLayer: Rendering', inferenceVisualization.zones.length, 'zones and', inferenceVisualization.confidence.length, 'confidence indicators');

  return (
    <>
      {/* Geological influence zones - MUCH MORE VISIBLE */}
      {inferenceVisualization.zones.map((zone) => (
        <Polygon
          key={zone.id}
          positions={zone.polygon}
          pathOptions={{
            color: zone.color,
            fillColor: zone.color,
            fillOpacity: 0.3, // Increased from 0.1
            weight: 4, // Increased from 2
            opacity: 0.8, // Increased from 0.4
            dashArray: '15, 10' // More prominent dashing
          }}
        />
      ))}
      
      {/* Confidence indicators for predictions - MUCH MORE VISIBLE */}
      {inferenceVisualization.confidence.map((indicator) => (
        <Circle
          key={indicator.id}
          center={indicator.position}
          radius={Math.max(5000, indicator.confidence * 8000)} // Much larger, minimum 5km
          pathOptions={{
            color: indicator.color,
            fillColor: indicator.color,
            fillOpacity: 0.4 + (indicator.confidence * 0.4), // Much more opaque
            weight: 3, // Increased from 2
            opacity: 0.9 // Increased from 0.6
          }}
        />
      ))}
    </>
  );
};

// Create geological influence zones based on transport patterns
function createInfluenceZone(region, knownErratics, allErratics) {
  const center = region.center;
  const baseRadius = region.radius;
  
  // Create elliptical zone accounting for ice flow direction
  // Most North American ice flowed generally south/southeast
  const iceFlowAngle = -45; // degrees (southeast)
  const ellipseRatio = 1.5; // elongated in flow direction
  
  const points = [];
  const numPoints = 32;
  
  for (let i = 0; i < numPoints; i++) {
    const angle = (i / numPoints) * 2 * Math.PI;
    
    // Create ellipse
    let x = Math.cos(angle) * baseRadius;
    let y = Math.sin(angle) * baseRadius * ellipseRatio;
    
    // Rotate by ice flow direction
    const rotAngle = (iceFlowAngle * Math.PI) / 180;
    const rotX = x * Math.cos(rotAngle) - y * Math.sin(rotAngle);
    const rotY = x * Math.sin(rotAngle) + y * Math.cos(rotAngle);
    
    // Convert to lat/lng (rough approximation)
    const lat = center[0] + (rotY / 111000); // ~111km per degree lat
    const lng = center[1] + (rotX / (111000 * Math.cos(center[0] * Math.PI / 180)));
    
    points.push([lat, lng]);
  }
  
  return points;
}

// Predict rock type using spatial analysis and geological principles
function predictRockType(unknownErratic, knownErratics, sourceRegions) {
  const [lng, lat] = unknownErratic.location.coordinates;
  
  // Distance-weighted influence from known erratics
  let totalWeight = 0;
  const rockTypeScores = {};
  
  knownErratics.forEach(known => {
    const [knownLng, knownLat] = known.location.coordinates;
    const distance = calculateDistance(lat, lng, knownLat, knownLng);
    
    // Inverse distance weighting with geological transport considerations
    const weight = 1 / (1 + Math.pow(distance / 200, 2)); // 200km characteristic scale
    
    const rockType = normalizeRockType(known.rock_type);
    rockTypeScores[rockType] = (rockTypeScores[rockType] || 0) + weight;
    totalWeight += weight;
  });
  
  // Source region influence
  Object.entries(sourceRegions).forEach(([name, region]) => {
    const distanceToSource = calculateDistance(lat, lng, region.center[0], region.center[1]);
    const maxTransport = 2000; // Maximum reasonable transport distance (km)
    
    if (distanceToSource < maxTransport) {
      const sourceWeight = Math.exp(-distanceToSource / 500) * 0.3; // Exponential decay
      
      region.dominantRocks.forEach(rockType => {
        rockTypeScores[rockType] = (rockTypeScores[rockType] || 0) + sourceWeight;
        totalWeight += sourceWeight;
      });
    }
  });
  
  // Size-based geological likelihood
  const size = unknownErratic.size_meters || 1;
  if (size > 20) {
    // Very large erratics more likely to be hard rocks
    rockTypeScores['granite'] = (rockTypeScores['granite'] || 0) + 0.2;
    rockTypeScores['quartzite'] = (rockTypeScores['quartzite'] || 0) + 0.2;
    totalWeight += 0.4;
  }
  
  // Find most likely rock type
  let bestRockType = 'granite'; // default
  let bestScore = 0;
  
  Object.entries(rockTypeScores).forEach(([rockType, score]) => {
    const normalizedScore = totalWeight > 0 ? score / totalWeight : 0;
    if (normalizedScore > bestScore) {
      bestScore = normalizedScore;
      bestRockType = rockType;
    }
  });
  
  return {
    rockType: bestRockType,
    confidence: Math.min(0.9, Math.max(0.1, bestScore * 2)) // Scale confidence, ensure minimum
  };
}

// Calculate zone confidence based on known data density
function calculateZoneConfidence(region, knownErratics) {
  let relevantErratics = 0;
  let totalErratics = 0;
  
  knownErratics.forEach(erratic => {
    const [lng, lat] = erratic.location.coordinates;
    const distance = calculateDistance(lat, lng, region.center[0], region.center[1]);
    
    if (distance < region.radius) {
      totalErratics++;
      const rockType = normalizeRockType(erratic.rock_type);
      if (region.dominantRocks.includes(rockType)) {
        relevantErratics++;
      }
    }
  });
  
  return totalErratics > 0 ? relevantErratics / totalErratics : 0;
}

// Utility functions
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
  if (!rockType || typeof rockType !== 'string') return 'granite';
  
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
  if (normalized.includes('metamorphic')) return 'gneiss'; // Approximate
  
  return 'granite'; // Most common default
}

function getRockTypeColor(rockType) {
  const colors = {
    'granite': '#FF4444', // Bright red
    'quartzite': '#4444FF', // Bright blue
    'gneiss': '#FF44FF', // Bright magenta
    'sandstone': '#FF8844', // Bright orange
    'limestone': '#888888', // Gray
    'basalt': '#444444', // Dark gray
    'schist': '#44FF44', // Bright green
    'conglomerate': '#FF8800', // Orange
    'greenstone': '#00FF88' // Bright teal
  };
  
  return colors[rockType] || '#44AAFF'; // Bright blue default
}

export default GeologicalInferenceLayer; 