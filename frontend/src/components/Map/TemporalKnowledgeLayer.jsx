import React, { useMemo } from 'react';
import { Circle, Polyline, Polygon } from 'react-leaflet';

const TemporalKnowledgeLayer = ({ erratics, isVisible }) => {
  // Create temporal knowledge evolution visualization
  const temporalElements = useMemo(() => {
    if (!erratics || erratics.length === 0) return {
      discoveryWaves: [],
      knowledgeFrontiers: [],
      researchClusters: [],
      temporalConnections: []
    };

    console.log('TemporalKnowledgeLayer: Processing', erratics.length, 'erratics');

    // Parse discovery dates and create temporal groups
    const erraticsWithDates = erratics.filter(e => 
      e.discovery_date && 
      e.location && 
      e.location.coordinates
    ).map(e => ({
      ...e,
      discoveryYear: parseDiscoveryYear(e.discovery_date),
      hasGeologicalData: e.rock_type && !e.rock_type.toLowerCase().includes('placeholder')
    })).filter(e => e.discoveryYear > 0);

    console.log('TemporalKnowledgeLayer: Erratics with valid dates:', erraticsWithDates.length);

    if (erraticsWithDates.length === 0) {
      console.log('TemporalKnowledgeLayer: No erratics with valid discovery dates found');
      return {
        discoveryWaves: [],
        knowledgeFrontiers: [],
        researchClusters: [],
        temporalConnections: []
      };
    }

    // Create discovery waves showing temporal patterns
    const discoveryWaves = createDiscoveryWaves(erraticsWithDates);
    
    // Create knowledge frontiers - boundaries of geological understanding
    const knowledgeFrontiers = createKnowledgeFrontiers(erraticsWithDates);
    
    // Create research clusters - areas of concentrated study
    const researchClusters = createResearchClusters(erraticsWithDates);
    
    // Create temporal connections showing knowledge propagation
    const temporalConnections = createTemporalConnections(erraticsWithDates);

    console.log('TemporalKnowledgeLayer: Created', discoveryWaves.length, 'waves,', knowledgeFrontiers.length, 'frontiers,', researchClusters.length, 'clusters,', temporalConnections.length, 'connections');

    return {
      discoveryWaves,
      knowledgeFrontiers,
      researchClusters,
      temporalConnections
    };
  }, [erratics]);

  if (!isVisible) return null;

  console.log('TemporalKnowledgeLayer: Rendering temporal elements');

  return (
    <>
      {/* Discovery waves - showing temporal patterns of discovery - MUCH MORE VISIBLE */}
      {temporalElements.discoveryWaves.map((wave) => (
        <Circle
          key={wave.id}
          center={wave.center}
          radius={wave.radius}
          pathOptions={{
            color: wave.color,
            fillColor: wave.color,
            fillOpacity: 0.3, // Much more visible
            weight: 4, // Thicker border
            opacity: 0.9, // Much more visible
            dashArray: wave.dashArray
          }}
        />
      ))}

      {/* Knowledge frontiers - boundaries of understanding - MUCH MORE VISIBLE */}
      {temporalElements.knowledgeFrontiers.map((frontier) => (
        <Polygon
          key={frontier.id}
          positions={frontier.polygon}
          pathOptions={{
            color: frontier.color,
            fillColor: frontier.color,
            fillOpacity: 0.2, // Much more visible
            weight: 5, // Much thicker
            opacity: 0.9, // Much more visible
            dashArray: '15, 10' // More prominent dashing
          }}
        />
      ))}

      {/* Research clusters - areas of concentrated study - MUCH MORE VISIBLE */}
      {temporalElements.researchClusters.map((cluster) => (
        <Circle
          key={cluster.id}
          center={cluster.center}
          radius={cluster.radius}
          pathOptions={{
            color: cluster.color,
            fillColor: cluster.color,
            fillOpacity: 0.4, // Much more visible
            weight: 3,
            opacity: 0.9
          }}
        />
      ))}

      {/* Temporal connections - knowledge propagation - MUCH MORE VISIBLE */}
      {temporalElements.temporalConnections.map((connection) => (
        <Polyline
          key={connection.id}
          positions={connection.path}
          pathOptions={{
            color: connection.color,
            weight: Math.max(3, connection.weight), // Minimum 3px
            opacity: Math.max(0.7, connection.opacity), // Minimum 70% opacity
            dashArray: connection.dashArray
          }}
        />
      ))}
    </>
  );
};

// Parse discovery year from various date formats
function parseDiscoveryYear(dateString) {
  if (!dateString || typeof dateString !== 'string') return 0;
  
  // Handle various formats: "1492-1-1", "1857", "2011-1-1", etc.
  const yearMatch = dateString.match(/(\d{4})/);
  if (yearMatch) {
    const year = parseInt(yearMatch[1]);
    // Filter out obviously placeholder dates
    if (year === 1492) return 0; // Common placeholder
    if (year < 1600 || year > 2024) return 0; // Unrealistic range
    return year;
  }
  
  return 0;
}

// Create discovery waves showing temporal patterns
function createDiscoveryWaves(erraticsWithDates) {
  if (erraticsWithDates.length === 0) return [];

  // Group erratics by discovery periods
  const periods = {
    'Early Exploration (1600-1800)': { min: 1600, max: 1800, color: '#FF4444' }, // Bright red
    'Scientific Revolution (1800-1850)': { min: 1800, max: 1850, color: '#44FF44' }, // Bright green
    'Geological Survey Era (1850-1900)': { min: 1850, max: 1900, color: '#4444FF' }, // Bright blue
    'Modern Documentation (1900-1950)': { min: 1900, max: 1950, color: '#FFFF44' }, // Bright yellow
    'Contemporary Research (1950-2024)': { min: 1950, max: 2024, color: '#FF44FF' } // Bright magenta
  };

  const waves = [];

  Object.entries(periods).forEach(([periodName, period]) => {
    const periodErratics = erraticsWithDates.filter(e => 
      e.discoveryYear >= period.min && e.discoveryYear <= period.max
    );

    console.log(`TemporalKnowledgeLayer: Period ${periodName}: ${periodErratics.length} erratics`);

    if (periodErratics.length > 0) {
      // Calculate centroid and spread
      const centroid = calculateCentroid(periodErratics);
      const spread = calculateSpread(periodErratics, centroid);
      
      // Create wave visualization
      waves.push({
        id: `wave-${period.min}-${period.max}`,
        center: centroid,
        radius: Math.max(100000, spread * 2), // Much larger minimum radius (100km)
        color: period.color,
        period: periodName,
        count: periodErratics.length,
        dashArray: period.min < 1850 ? '20, 15' : null // More prominent dashing for early periods
      });
    }
  });

  console.log('TemporalKnowledgeLayer: Created', waves.length, 'discovery waves');
  return waves;
}

// Create knowledge frontiers - boundaries of geological understanding
function createKnowledgeFrontiers(erraticsWithDates) {
  const frontiers = [];
  
  // Separate erratics with and without geological data
  const withGeology = erraticsWithDates.filter(e => e.hasGeologicalData);
  const withoutGeology = erraticsWithDates.filter(e => !e.hasGeologicalData);
  
  console.log('TemporalKnowledgeLayer: With geology:', withGeology.length, 'Without geology:', withoutGeology.length);

  if (withGeology.length >= 3) {
    // Create convex hull around erratics with geological data
    const knowledgeHull = createConvexHull(withGeology);
    
    if (knowledgeHull.length >= 3) {
      frontiers.push({
        id: 'knowledge-frontier',
        polygon: knowledgeHull,
        color: '#00FF00', // Bright green
        type: 'Known Geology'
      });
    }
  }

  // Create frontier around areas needing research
  if (withoutGeology.length >= 3) {
    const researchHull = createConvexHull(withoutGeology);
    
    if (researchHull.length >= 3) {
      frontiers.push({
        id: 'research-frontier',
        polygon: researchHull,
        color: '#FF0000', // Bright red
        type: 'Research Needed'
      });
    }
  }

  console.log('TemporalKnowledgeLayer: Created', frontiers.length, 'knowledge frontiers');
  return frontiers;
}

// Create research clusters - areas of concentrated study
function createResearchClusters(erraticsWithDates) {
  const clusters = [];
  
  // Group by decade and location
  const decades = {};
  
  erraticsWithDates.forEach(erratic => {
    const decade = Math.floor(erratic.discoveryYear / 10) * 10;
    if (!decades[decade]) decades[decade] = [];
    decades[decade].push(erratic);
  });

  Object.entries(decades).forEach(([decade, erratics]) => {
    console.log(`TemporalKnowledgeLayer: Decade ${decade}: ${erratics.length} erratics`);
    
    if (erratics.length >= 2) { // Lowered minimum cluster size
      // Find spatial clusters within this decade
      const spatialClusters = findSpatialClusters(erratics, 200); // Increased threshold to 200km
      
      spatialClusters.forEach((cluster, index) => {
        if (cluster.length >= 2) {
          const centroid = calculateCentroid(cluster);
          const radius = Math.max(50000, calculateSpread(cluster, centroid)); // Minimum 50km
          
          // Color based on geological data completeness
          const geologicalRatio = cluster.filter(e => e.hasGeologicalData).length / cluster.length;
          const color = getResearchIntensityColor(geologicalRatio);
          
          clusters.push({
            id: `cluster-${decade}-${index}`,
            center: centroid,
            radius: radius,
            color: color,
            decade: decade,
            count: cluster.length,
            geologicalRatio: geologicalRatio
          });
        }
      });
    }
  });

  console.log('TemporalKnowledgeLayer: Created', clusters.length, 'research clusters');
  return clusters;
}

// Create temporal connections showing knowledge propagation
function createTemporalConnections(erraticsWithDates) {
  const connections = [];
  
  // Sort by discovery year
  const sortedErratics = [...erraticsWithDates].sort((a, b) => a.discoveryYear - b.discoveryYear);
  
  // Connect erratics discovered in sequence if they're nearby
  for (let i = 0; i < sortedErratics.length - 1; i++) {
    const current = sortedErratics[i];
    const next = sortedErratics[i + 1];
    
    const [lng1, lat1] = current.location.coordinates;
    const [lng2, lat2] = next.location.coordinates;
    
    const distance = calculateDistance(lat1, lng1, lat2, lng2);
    const timeDiff = next.discoveryYear - current.discoveryYear;
    
    // Connect if spatially close and temporally sequential
    if (distance < 500 && timeDiff <= 30) { // Increased thresholds
      const connectionStrength = Math.exp(-distance / 200) * Math.exp(-timeDiff / 15);
      
      if (connectionStrength > 0.05) { // Lowered threshold
        connections.push({
          id: `connection-${current.id}-${next.id}`,
          path: [[lat1, lng1], [lat2, lng2]],
          color: getTemporalConnectionColor(timeDiff),
          weight: Math.max(2, connectionStrength * 8), // Thicker lines
          opacity: Math.max(0.5, connectionStrength),
          dashArray: timeDiff > 15 ? '12, 8' : null,
          timeDiff: timeDiff,
          distance: distance
        });
      }
    }
  }

  console.log('TemporalKnowledgeLayer: Created', connections.length, 'temporal connections');
  return connections;
}

// Utility functions
function calculateCentroid(erratics) {
  let totalLat = 0, totalLng = 0;
  
  erratics.forEach(erratic => {
    const [lng, lat] = erratic.location.coordinates;
    totalLat += lat;
    totalLng += lng;
  });
  
  return [totalLat / erratics.length, totalLng / erratics.length];
}

function calculateSpread(erratics, centroid) {
  let maxDistance = 0;
  
  erratics.forEach(erratic => {
    const [lng, lat] = erratic.location.coordinates;
    const distance = calculateDistance(centroid[0], centroid[1], lat, lng);
    maxDistance = Math.max(maxDistance, distance);
  });
  
  return maxDistance * 1000; // Convert to meters
}

function findSpatialClusters(erratics, thresholdKm) {
  const clusters = [];
  const used = new Set();
  
  erratics.forEach((erratic, index) => {
    if (used.has(index)) return;
    
    const cluster = [erratic];
    used.add(index);
    
    const [lng1, lat1] = erratic.location.coordinates;
    
    // Find nearby erratics
    erratics.forEach((other, otherIndex) => {
      if (used.has(otherIndex)) return;
      
      const [lng2, lat2] = other.location.coordinates;
      const distance = calculateDistance(lat1, lng1, lat2, lng2);
      
      if (distance <= thresholdKm) {
        cluster.push(other);
        used.add(otherIndex);
      }
    });
    
    clusters.push(cluster);
  });
  
  return clusters;
}

function createConvexHull(erratics) {
  if (erratics.length < 3) return [];
  
  // Simple convex hull algorithm (gift wrapping)
  const points = erratics.map(e => {
    const [lng, lat] = e.location.coordinates;
    return [lat, lng]; // [lat, lng] for Leaflet
  });
  
  // Find leftmost point
  let leftmost = 0;
  for (let i = 1; i < points.length; i++) {
    if (points[i][1] < points[leftmost][1]) {
      leftmost = i;
    }
  }
  
  const hull = [];
  let current = leftmost;
  
  do {
    hull.push(points[current]);
    let next = (current + 1) % points.length;
    
    for (let i = 0; i < points.length; i++) {
      if (orientation(points[current], points[i], points[next]) === 2) {
        next = i;
      }
    }
    
    current = next;
  } while (current !== leftmost);
  
  return hull;
}

function orientation(p, q, r) {
  const val = (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
  if (val === 0) return 0; // Collinear
  return val > 0 ? 1 : 2; // Clockwise or Counterclockwise
}

function getResearchIntensityColor(geologicalRatio) {
  if (geologicalRatio > 0.8) return '#00FF00'; // Bright green - high knowledge
  if (geologicalRatio > 0.5) return '#FFAA00'; // Bright orange - medium knowledge
  if (geologicalRatio > 0.2) return '#FF6600'; // Orange - low knowledge
  return '#FF0000'; // Bright red - very low knowledge
}

function getTemporalConnectionColor(timeDiff) {
  if (timeDiff <= 5) return '#00AAFF'; // Bright blue - recent
  if (timeDiff <= 15) return '#AA00FF'; // Bright purple - medium
  return '#AAAAAA'; // Gray - distant
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

export default TemporalKnowledgeLayer; 