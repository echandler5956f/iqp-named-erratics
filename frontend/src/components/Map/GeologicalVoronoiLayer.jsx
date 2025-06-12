import React, { useMemo } from 'react';
import { Polygon } from 'react-leaflet';

const GeologicalVoronoiLayer = ({ erratics, isVisible }) => {
  // Generate Voronoi diagram based on erratic locations and rock types
  const voronoiCells = useMemo(() => {
    if (!erratics || erratics.length === 0) return [];

    // Filter erratics with valid locations
    const validErratics = erratics.filter(e => 
      e.location && e.location.coordinates && e.location.coordinates.length >= 2
    );

    if (validErratics.length < 3) return []; // Need at least 3 points for Voronoi

    // Convert to points for Voronoi calculation
    const points = validErratics.map(erratic => ({
      x: erratic.location.coordinates[0], // longitude
      y: erratic.location.coordinates[1], // latitude
      erratic: erratic
    }));

    // Calculate Voronoi diagram
    const voronoiCells = calculateVoronoi(points);

    // Rock type colors (matching the flow layer)
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

    // Convert Voronoi cells to polygons with styling
    return voronoiCells.map((cell, index) => {
      const rockType = normalizeRockType(cell.erratic.rock_type);
      const color = rockTypeColors[rockType] || rockTypeColors.default;
      const size = cell.erratic.size_meters || 1;
      
      // Larger erratics get more prominent territories
      const opacity = Math.max(0.1, Math.min(0.4, 0.15 + (size / 50)));
      
      return {
        id: `voronoi-${cell.erratic.id}`,
        positions: cell.polygon,
        color: color,
        fillColor: color,
        fillOpacity: opacity,
        weight: 1,
        opacity: 0.6,
        rockType: rockType,
        erratic: cell.erratic
      };
    });
  }, [erratics]);

  if (!isVisible) return null;

  return (
    <>
      {voronoiCells.map((cell) => (
        <Polygon
          key={cell.id}
          positions={cell.positions}
          pathOptions={{
            color: cell.color,
            fillColor: cell.fillColor,
            fillOpacity: cell.fillOpacity,
            weight: cell.weight,
            opacity: cell.opacity
          }}
        />
      ))}
    </>
  );
};

// Calculate Voronoi diagram using Fortune's algorithm (simplified implementation)
function calculateVoronoi(points) {
  // For a production implementation, you'd use a library like d3-delaunay
  // This is a simplified version that creates approximate Voronoi cells
  
  const cells = [];
  const bounds = calculateBounds(points);
  
  // Expand bounds slightly for edge cells
  const margin = 2.0;
  bounds.minX -= margin;
  bounds.maxX += margin;
  bounds.minY -= margin;
  bounds.maxY += margin;
  
  points.forEach((point, index) => {
    const cell = createVoronoiCell(point, points, bounds);
    if (cell && cell.length >= 3) { // Valid polygon needs at least 3 points
      cells.push({
        erratic: point.erratic,
        polygon: cell
      });
    }
  });
  
  return cells;
}

// Calculate bounding box for all points
function calculateBounds(points) {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  
  points.forEach(point => {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  });
  
  return { minX, maxX, minY, maxY };
}

// Create a Voronoi cell for a given point (simplified approach)
function createVoronoiCell(centerPoint, allPoints, bounds) {
  // This creates an approximate Voronoi cell by finding the region closer to centerPoint
  // than to any other point. For production, use a proper Voronoi library.
  
  const cellVertices = [];
  const resolution = 20; // Number of test points around the perimeter
  
  // Create a rough approximation by testing points in a grid around the center
  const testRadius = 3.0; // degrees
  const angleStep = (2 * Math.PI) / resolution;
  
  for (let i = 0; i < resolution; i++) {
    const angle = i * angleStep;
    let radius = 0.1;
    let maxRadius = testRadius;
    let bestPoint = null;
    
    // Binary search to find the boundary of the Voronoi cell
    for (let step = 0; step < 10; step++) {
      const testRadius = (radius + maxRadius) / 2;
      const testX = centerPoint.x + Math.cos(angle) * testRadius;
      const testY = centerPoint.y + Math.sin(angle) * testRadius;
      
      // Check if this test point is closer to centerPoint than to any other point
      const distToCenter = distance(testX, testY, centerPoint.x, centerPoint.y);
      let isClosestToCenter = true;
      
      for (let j = 0; j < allPoints.length; j++) {
        if (j === allPoints.indexOf(centerPoint)) continue;
        
        const distToOther = distance(testX, testY, allPoints[j].x, allPoints[j].y);
        if (distToOther < distToCenter) {
          isClosestToCenter = false;
          break;
        }
      }
      
      if (isClosestToCenter) {
        radius = testRadius;
        bestPoint = [testY, testX]; // [lat, lng] for Leaflet
      } else {
        maxRadius = testRadius;
      }
    }
    
    if (bestPoint) {
      cellVertices.push(bestPoint);
    }
  }
  
  // Ensure the polygon is closed
  if (cellVertices.length > 0) {
    cellVertices.push(cellVertices[0]);
  }
  
  return cellVertices;
}

// Calculate distance between two points
function distance(x1, y1, x2, y2) {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
}

// Normalize rock type names (same as flow layer)
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

export default GeologicalVoronoiLayer; 