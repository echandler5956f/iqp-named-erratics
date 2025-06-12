import React, { useMemo } from 'react';
import { Polyline } from 'react-leaflet';

const GlacialFlowLayer = ({ erratics, isVisible }) => {
  // Generate elegant flow lines using only the real data we have
  const flowLines = useMemo(() => {
    if (!erratics || erratics.length === 0) return [];

    // Simple, realistic source region - Canadian Shield around Hudson Bay
    const sourceRegion = {
      centerLat: 55.0,
      centerLng: -85.0,
      radius: 8.0  // Large enough region for natural variation
    };

    // Beautiful, geologically-inspired colors for rock types
    const rockTypeColors = {
      'granite': '#C0392B',      // Deep red
      'quartzite': '#2980B9',    // Rich blue  
      'gneiss': '#8E44AD',       // Purple
      'sandstone': '#E67E22',    // Orange
      'limestone': '#7F8C8D',    // Gray
      'basalt': '#2C3E50',       // Dark blue
      'schist': '#27AE60',       // Green
      'conglomerate': '#D35400', // Dark orange
      'default': '#3498DB'       // Light blue
    };

    const lines = [];

    erratics.forEach(erratic => {
      if (!erratic.location || !erratic.location.coordinates) return;

      const [lng, lat] = erratic.location.coordinates;
      
      // Generate source point with natural variation
      const sourcePoint = generateSourcePoint(sourceRegion, erratic);
      
      // Create elegant curved path
      const flowPath = createElegantPath(sourcePoint, { lat, lng }, erratic);
      
      // Visual properties based on actual data
      const rockType = normalizeRockType(erratic.rock_type);
      const color = rockTypeColors[rockType] || rockTypeColors.default;
      const size = erratic.size_meters || 1;
      
      // Simple, effective scaling
      const weight = Math.max(1, Math.min(5, 1 + Math.sqrt(size) * 0.3));
      const opacity = Math.max(0.4, Math.min(0.8, 0.3 + (size / 25)));

      lines.push({
        id: `flow-${erratic.id}`,
        path: flowPath,
        color: color,
        weight: weight,
        opacity: opacity
      });
    });

    return lines;
  }, [erratics]);

  if (!isVisible) return null;

  return (
    <>
      {flowLines.map((line) => (
        <Polyline
          key={line.id}
          positions={line.path}
          pathOptions={{
            color: line.color,
            weight: line.weight,
            opacity: line.opacity,
            lineCap: 'round',
            lineJoin: 'round'
          }}
        />
      ))}
    </>
  );
};

// Generate source point with natural variation based on erratic properties
function generateSourcePoint(sourceRegion, erratic) {
  // Use erratic properties to create natural variation in source location
  const rockType = normalizeRockType(erratic.rock_type);
  const size = erratic.size_meters || 1;
  const [lng, lat] = erratic.location.coordinates;
  
  // Create variation based on where the erratic ended up (eastern vs western erratics)
  const longitudinalOffset = (lng + 85) * 0.1; // Offset based on final longitude
  const latitudinalOffset = (lat - 45) * 0.05;  // Offset based on final latitude
  
  // Rock type creates slight clustering in source region
  const rockTypeOffsets = {
    'granite': { lat: 1.0, lng: -1.5 },
    'quartzite': { lat: -0.5, lng: 2.0 },
    'gneiss': { lat: 1.5, lng: 0.5 },
    'sandstone': { lat: -1.0, lng: -1.0 },
    'limestone': { lat: 0.5, lng: 1.5 },
    'basalt': { lat: 2.0, lng: -0.5 },
    'schist': { lat: -1.5, lng: 1.0 },
    'conglomerate': { lat: 0.0, lng: -2.0 },
    'default': { lat: 0.0, lng: 0.0 }
  };
  
  const rockOffset = rockTypeOffsets[rockType] || rockTypeOffsets.default;
  
  // Size creates slight variation (larger erratics from slightly different areas)
  const sizeOffset = {
    lat: (Math.log(size + 1) - 1) * 0.3,
    lng: (Math.log(size + 1) - 1) * 0.2
  };
  
  // Random natural variation
  const randomLat = (Math.random() - 0.5) * 3.0;
  const randomLng = (Math.random() - 0.5) * 4.0;
  
  return {
    lat: sourceRegion.centerLat + rockOffset.lat + sizeOffset.lat + longitudinalOffset + randomLat,
    lng: sourceRegion.centerLng + rockOffset.lng + sizeOffset.lng + latitudinalOffset + randomLng
  };
}

// Create smooth Bézier curve path
function createElegantPath(source, destination, erratic) {
  // Calculate control points for a smooth Bézier curve
  const latDiff = destination.lat - source.lat;
  const lngDiff = destination.lng - source.lng;
  const distance = Math.sqrt(latDiff * latDiff + lngDiff * lngDiff);
  
  // Size-based variation for larger erratics
  const size = erratic.size_meters || 1;
  const sizeInfluence = Math.log(size + 1) * 0.02;
  
  // Create control points for natural curve
  const controlPoint1 = {
    lat: source.lat + latDiff * 0.3 - distance * 0.08 + sizeInfluence,
    lng: source.lng + lngDiff * 0.3 + distance * 0.12 + sizeInfluence * 0.7
  };
  
  const controlPoint2 = {
    lat: source.lat + latDiff * 0.7 - distance * 0.06 + sizeInfluence * 0.5,
    lng: source.lng + lngDiff * 0.7 + distance * 0.10 + sizeInfluence * 0.5
  };
  
  // Generate smooth Bézier curve points
  const path = [];
  const numPoints = 25; // More points for smoother curves
  
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints;
    
    // Cubic Bézier curve formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    const oneMinusT = 1 - t;
    const oneMinusTSquared = oneMinusT * oneMinusT;
    const oneMinusTCubed = oneMinusTSquared * oneMinusT;
    const tSquared = t * t;
    const tCubed = tSquared * t;
    
    const lat = oneMinusTCubed * source.lat +
                3 * oneMinusTSquared * t * controlPoint1.lat +
                3 * oneMinusT * tSquared * controlPoint2.lat +
                tCubed * destination.lat;
                
    const lng = oneMinusTCubed * source.lng +
                3 * oneMinusTSquared * t * controlPoint1.lng +
                3 * oneMinusT * tSquared * controlPoint2.lng +
                tCubed * destination.lng;
    
    path.push([lat, lng]);
  }
  
  return path;
}

// Simple rock type normalization
function normalizeRockType(rockType) {
  if (!rockType || typeof rockType !== 'string') return 'default';
  
  const normalized = rockType.toLowerCase().trim();
  
  if (normalized.includes('granite')) return 'granite';
  if (normalized.includes('quartzite')) return 'quartzite';
  if (normalized.includes('gneiss')) return 'gneiss';
  if (normalized.includes('sandstone')) return 'sandstone';
  if (normalized.includes('limestone')) return 'limestone';
  if (normalized.includes('basalt')) return 'basalt';
  if (normalized.includes('schist')) return 'schist';
  if (normalized.includes('conglomerate')) return 'conglomerate';
  
  return 'default';
}

export default GlacialFlowLayer; 