import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import ErraticsMap from '../components/Map/ErraticsMap';
import Header from '../components/Layout/Header';
import FilterPanel from '../components/FilterSystem/FilterPanel';
import { getVisibleErraticIds } from '../components/FilterSystem/filterUtils';
import { solveTsp, calculatePathDistance } from '../services/tspService';
import styles from './HomePage.module.css';

// Updated filter definitions to match refactored backend schema
const GLOBAL_FILTER_DEFINITIONS = {
  size: {
    label: 'Size (meters)',
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => ( 
      <div className={panelStyles.filterConfigRow}> 
        <label htmlFor="minSize">Min:</label>
        <input
          type="number"
          id="minSize"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Min size"
        />
        <label htmlFor="maxSize">Max:</label>
        <input
          type="number"
          id="maxSize"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max size"
        />
      </div>
    ),
  },
  proximity_water: {
    label: 'Proximity to Water (meters)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistWater">Max Distance:</label>
        <input
          type="number"
          id="maxDistWater"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  proximity_forest_trail: {
    label: 'Proximity to Forest Trail (meters)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistForestTrail">Max Distance:</label>
        <input
          type="number"
          id="maxDistForestTrail"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  proximity_settlement: {
    label: 'Proximity to Settlement (meters)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistSettlement">Max Distance:</label>
        <input
          type="number"
          id="maxDistSettlement"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  proximity_road: {
    label: 'Proximity to Road (meters)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistRoad">Max Distance:</label>
        <input
          type="number"
          id="maxDistRoad"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  elevation_category: {
    label: 'Elevation Category',
    defaultConfig: { category: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="elevationCategory">Category:</label>
        <select
          id="elevationCategory"
          value={config.category ?? ''}
          onChange={(e) => onChange({ ...config, category: e.target.value })}
        >
          <option value="">Any Elevation</option>
          <option value="lowland">Lowland</option>
          <option value="highland">Highland</option>
          <option value="mountain">Mountain</option>
          <option value="coastal">Coastal</option>
        </select>
      </div>
    ),
  },
  rock_type: {
    label: 'Rock Type',
    defaultConfig: { type: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="rockType">Type:</label>
        <input
          type="text"
          id="rockType"
          value={config.type ?? ''}
          onChange={(e) => onChange({ ...config, type: e.target.value })}
          placeholder="e.g., Granite (case insensitive)"
        />
      </div>
    ),
  },
  usage_type: {
    label: 'Usage Type Contains',
    defaultConfig: { tag: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="usageTag">Tag:</label>
        <input
          type="text"
          id="usageTag"
          value={config.tag ?? ''}
          onChange={(e) => onChange({ ...config, tag: e.target.value })}
          placeholder="e.g., Ceremonial (case insensitive)"
        />
      </div>
    ),
  },
  has_inscriptions: {
    label: 'Has Inscriptions',
    defaultConfig: { required: true },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label>
          <input
            type="checkbox"
            checked={config.required === true}
            onChange={(e) => onChange({ ...config, required: e.target.checked })} 
          />
          Must have inscriptions
        </label>
      </div>
    ),
  },
  accessibility_score: {
    label: 'Accessibility Score (1-10)',
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="minAccessibility">Min:</label>
        <input
          type="number"
          id="minAccessibility"
          min="1" max="10"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="1-10"
        />
        <label htmlFor="maxAccessibility">Max:</label>
        <input
          type="number"
          id="maxAccessibility"
          min="1" max="10"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="1-10"
        />
      </div>
    ),
  },
  terrain_landform: {
    label: 'Terrain Landform',
    defaultConfig: { type: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="terrainLandform">Type:</label>
        <select
          id="terrainLandform"
          value={config.type ?? ''}
          onChange={(e) => onChange({ ...config, type: e.target.value })}
        >
          <option value="">Any Landform</option>
          <option value="valley">Valley</option>
          <option value="ridge">Ridge</option>
          <option value="slope">Slope</option>
          <option value="plateau">Plateau</option>
          <option value="depression">Depression</option>
        </select>
      </div>
    ),
  },
  proximity_native_territory: {
    label: 'Proximity to Native Territory (meters)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistNativeTerritory">Max Distance:</label>
        <input
          type="number"
          id="maxDistNativeTerritory"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  proximity_natd_road: {
    label: 'Proximity to NATD Road (meters)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistNatdRoad">Max Distance:</label>
        <input
          type="number"
          id="maxDistNatdRoad"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  cultural_significance_score: {
    label: 'Cultural Significance Score (0-10)',
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="minCulturalSignificance">Min:</label>
        <input
          type="number"
          id="minCulturalSignificance"
          min="0" max="10"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="0-10"
        />
        <label htmlFor="maxCulturalSignificance">Max:</label>
        <input
          type="number"
          id="maxCulturalSignificance"
          min="0" max="10"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="0-10"
        />
      </div>
    ),
  },
  discovery_date: {
    label: 'Discovery Date',
    defaultConfig: { startYear: null, endYear: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="startYear">From Year:</label>
        <input
          type="number"
          id="startYear"
          min="1000" max={new Date().getFullYear()}
          value={config.startYear ?? ''}
          onChange={(e) => onChange({ ...config, startYear: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="e.g., 1620"
        />
        <label htmlFor="endYear">To Year:</label>
        <input
          type="number"
          id="endYear"
          min="1000" max={new Date().getFullYear()}
          value={config.endYear ?? ''}
          onChange={(e) => onChange({ ...config, endYear: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="e.g., 1900"
        />
      </div>
    ),
  },
  estimated_age: {
    label: 'Estimated Age',
    defaultConfig: { age: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="estimatedAge">Age:</label>
        <input
          type="text"
          id="estimatedAge"
          value={config.age ?? ''}
          onChange={(e) => onChange({ ...config, age: e.target.value })}
          placeholder="e.g., Pleistocene (case insensitive)"
        />
      </div>
    ),
  },
  elevation: {
    label: 'Elevation (meters)',
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="minElevation">Min:</label>
        <input
          type="number"
          id="minElevation"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Min elevation"
        />
        <label htmlFor="maxElevation">Max:</label>
        <input
          type="number"
          id="maxElevation"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max elevation"
        />
      </div>
    ),
  },
  size_category: {
    label: 'Size Category',
    defaultConfig: { category: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="sizeCategory">Category:</label>
        <select
          id="sizeCategory"
          value={config.category ?? ''}
          onChange={(e) => onChange({ ...config, category: e.target.value })}
        >
          <option value="">Any Size Category</option>
          {/* Options will be populated dynamically */}
        </select>
      </div>
    ),
  },
  geological_type: {
    label: 'Geological Type',
    defaultConfig: { type: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="geologicalType">Type:</label>
        <select
          id="geologicalType"
          value={config.type ?? ''}
          onChange={(e) => onChange({ ...config, type: e.target.value })}
        >
          <option value="">Any Geological Type</option>
          {/* Options will be populated dynamically */}
        </select>
      </div>
    ),
  },
  displacement_distance: {
    label: 'Estimated Displacement (meters)', // Assuming meters based on other distance fields
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="minDisplacement">Min:</label>
        <input
          type="number"
          id="minDisplacement"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Min distance"
        />
        <label htmlFor="maxDisplacement">Max:</label>
        <input
          type="number"
          id="maxDisplacement"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance"
        />
      </div>
    ),
  },
  ruggedness: {
    label: 'Terrain Ruggedness (TRI)',
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="minRuggedness">Min:</label>
        <input
          type="number"
          id="minRuggedness"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Min TRI"
        />
        <label htmlFor="maxRuggedness">Max:</label>
        <input
          type="number"
          id="maxRuggedness"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max TRI"
        />
      </div>
    ),
  },
  slope_position: {
    label: 'Terrain Slope Position',
    defaultConfig: { type: '' },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="slopePosition">Position:</label>
        <select
          id="slopePosition"
          value={config.type ?? ''}
          onChange={(e) => onChange({ ...config, type: e.target.value })}
        >
          <option value="">Any Slope Position</option>
          {/* Options will be populated dynamically */}
        </select>
      </div>
    ),
  },
};

function HomePage() {
  const [allErraticData, setAllErraticData] = useState([]);
  // -------------------------
  //  User-location & TSP state
  // -------------------------
  const [userLocation, setUserLocation] = useState(null);            // [lat, lng]
  const [locationError, setLocationError] = useState(null);

  const [isTspPathVisible, setIsTspPathVisible] = useState(false);
  const [tspPath, setTspPath] = useState([]);                       // Array<[lat,lng]>
  const [isCalculatingTsp, setIsCalculatingTsp] = useState(false);
  const [tspDistanceKm, setTspDistanceKm] = useState(0);
  const [isRequestingLocation, setIsRequestingLocation] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState([]);
  const [isFilterPanelCollapsed, setIsFilterPanelCollapsed] = useState(false);

  // ------------------------------------------------------------
  //  Acquire user location once on mount (high-accuracy attempt)
  // ------------------------------------------------------------
  // Helper: request geolocation with robust error handling
  const requestUserLocation = () => {
    if (!navigator.geolocation) {
      setLocationError('Geolocation API not available in this browser.');
      return;
    }

    setIsRequestingLocation(true);
    setLocationError(null);

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLocation([pos.coords.latitude, pos.coords.longitude]);
        setLocationError(null);
        setIsRequestingLocation(false);
      },
      (err) => {
        let msg = err.message || 'Unable to retrieve location.';
        if (err.code === 1) {
          // Permission denied
          msg = 'Location permission denied. Adjust browser site settings to enable.';
        }
        setLocationError(msg);
        setIsRequestingLocation(false);
      },
      { enableHighAccuracy: true, timeout: 15000 }
    );
  };

  // Request location once on mount if permission state is already granted
  useEffect(() => {
    if (!('permissions' in navigator)) {
      // Fallback: try once
      requestUserLocation();
      return;
    }
    navigator.permissions.query({ name: 'geolocation' }).then((status) => {
      if (status.state === 'granted') {
        requestUserLocation();
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const fetchErratics = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get('/api/erratics');
        if (Array.isArray(response.data)) {
          setAllErraticData(response.data);
        } else {
          console.warn('API did not return an array for erratics. Received:', response.data);
          setAllErraticData([]);
          setError('Data format error: Expected an array of erratics.');
        }
      } catch (err) {
        console.error('Error fetching erratics:', err);
        setError(err.message || 'Failed to fetch erratics from server.');
        setAllErraticData([]);
      }
      setIsLoading(false);
    };
    fetchErratics();
  }, []);

  // Calculate Distinct Values for Dropdowns
  const distinctRockTypes = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const types = new Set(allErraticData.map(e => e.rock_type).filter(Boolean));
    return Array.from(types).sort();
  }, [allErraticData]);

  const distinctUsageTags = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const tags = new Set();
    allErraticData.forEach(e => {
      if (Array.isArray(e.usage_type)) {
        e.usage_type.forEach(tag => {
          if (tag) tags.add(tag);
        });
      }
    });
    return Array.from(tags).sort();
  }, [allErraticData]);

  const distinctTerrainLandforms = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const landforms = new Set(allErraticData.map(e => e.terrain_landform).filter(Boolean)); 
    return Array.from(landforms).sort();
  }, [allErraticData]);

  const distinctSizeCategories = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const categories = new Set(allErraticData.map(e => e.size_category).filter(Boolean));
    // Expected values: "Small", "Medium", "Large", "Monumental" - sorting might be okay
    return Array.from(categories).sort(); 
  }, [allErraticData]);

  const distinctGeologicalTypes = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const types = new Set(allErraticData.map(e => e.geological_type).filter(Boolean));
    return Array.from(types).sort();
  }, [allErraticData]);
  
  const distinctSlopePositions = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const positions = new Set(allErraticData.map(e => e.terrain_slope_position).filter(Boolean));
    return Array.from(positions).sort();
  }, [allErraticData]);

  const distinctEstimatedAges = useMemo(() => {
    if (!Array.isArray(allErraticData)) return [];
    const ages = new Set(allErraticData.map(e => e.estimated_age).filter(Boolean));
    return Array.from(ages).sort();
  }, [allErraticData]);

  // Define GLOBAL_FILTER_DEFINITIONS with dynamic data
  const GLOBAL_FILTER_DEFINITIONS_WITH_DATA = useMemo(() => ({
    ...GLOBAL_FILTER_DEFINITIONS,
    rock_type: {
      ...GLOBAL_FILTER_DEFINITIONS.rock_type,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="rockType">Type:</label>
          <select
            id="rockType"
            value={config.type ?? ''}
            onChange={(e) => onChange({ ...config, type: e.target.value })}
          >
            <option value="">Any Rock Type</option>
            {distinctRockTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
      ),
    },
    usage_type: {
      ...GLOBAL_FILTER_DEFINITIONS.usage_type,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="usageTag">Tag:</label>
          <select
            id="usageTag"
            value={config.tag ?? ''}
            onChange={(e) => onChange({ ...config, tag: e.target.value })}
          >
            <option value="">Any Usage Tag</option>
            {distinctUsageTags.map(tag => (
              <option key={tag} value={tag}>{tag}</option>
            ))}
          </select>
        </div>
      ),
    },
    terrain_landform: {
      ...GLOBAL_FILTER_DEFINITIONS.terrain_landform,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="terrainLandform">Type:</label>
          <select
            id="terrainLandform"
            value={config.type ?? ''}
            onChange={(e) => onChange({ ...config, type: e.target.value })}
          >
            <option value="">Any Landform</option>
            {distinctTerrainLandforms.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
      ),
    },
    size_category: {
      ...GLOBAL_FILTER_DEFINITIONS.size_category,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="sizeCategory">Category:</label>
          <select
            id="sizeCategory"
            value={config.category ?? ''}
            onChange={(e) => onChange({ ...config, category: e.target.value })}
          >
            <option value="">Any Size Category</option>
            {distinctSizeCategories.map(category => (
              <option key={category} value={category}>{category}</option>
            ))}
          </select>
        </div>
      ),
    },
    geological_type: {
      ...GLOBAL_FILTER_DEFINITIONS.geological_type,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="geologicalType">Type:</label>
          <select
            id="geologicalType"
            value={config.type ?? ''}
            onChange={(e) => onChange({ ...config, type: e.target.value })}
          >
            <option value="">Any Geological Type</option>
            {distinctGeologicalTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
      ),
    },
    estimated_age: {
      ...GLOBAL_FILTER_DEFINITIONS.estimated_age,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="estimatedAge">Age:</label>
          <select
            id="estimatedAge"
            value={config.age ?? ''}
            onChange={(e) => onChange({ ...config, age: e.target.value })}
          >
            <option value="">Any Estimated Age</option>
            {distinctEstimatedAges.map(age => (
              <option key={age} value={age}>{age}</option>
            ))}
          </select>
        </div>
      ),
    },
    slope_position: {
      ...GLOBAL_FILTER_DEFINITIONS.slope_position,
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="slopePosition">Position:</label>
          <select
            id="slopePosition"
            value={config.type ?? ''}
            onChange={(e) => onChange({ ...config, type: e.target.value })}
          >
            <option value="">Any Slope Position</option>
            {distinctSlopePositions.map(position => (
              <option key={position} value={position}>{position}</option>
            ))}
          </select>
        </div>
      ),
    },
  }), [distinctRockTypes, distinctUsageTags, distinctTerrainLandforms, distinctSizeCategories, distinctGeologicalTypes, distinctEstimatedAges, distinctSlopePositions]);

  const activeFilters = useMemo(() => filters.filter(f => f.isActive), [filters]);

  const visibleErraticIds = useMemo(() => {
    if (!Array.isArray(allErraticData)) {
        console.error('visibleErraticIds: allErraticData is not an array!', allErraticData);
        return new Set();
    }
    return getVisibleErraticIds(allErraticData, activeFilters, GLOBAL_FILTER_DEFINITIONS_WITH_DATA);
  }, [allErraticData, activeFilters, GLOBAL_FILTER_DEFINITIONS_WITH_DATA]);

  const erraticsToDisplay = useMemo(() => {
    if (!Array.isArray(allErraticData)) {
        console.error('erraticsToDisplay: allErraticData is not an array!', allErraticData);
        return [];
    }
    if (activeFilters.length === 0) return allErraticData;
    return allErraticData.filter(e => visibleErraticIds.has(e.id));
  }, [allErraticData, visibleErraticIds, activeFilters.length]);

  const handleFiltersChange = (newFilters) => {
    setFilters(newFilters);
  };

  // ------------------------------------------------------------
  //  Recompute TSP path whenever visibility/userLocation/toggle changes
  // ------------------------------------------------------------
  useEffect(() => {
    const computeTsp = async () => {
      if (!isTspPathVisible) {
        setTspPath([]);
        setTspDistanceKm(0);
        return;
      }

      // Build node list: user first (if available) + visible erratics
      const nodes = [];
      if (userLocation && Array.isArray(userLocation)) {
        nodes.push({ id: 'user', lat: userLocation[0], lng: userLocation[1], name: 'You' });
      }

      erraticsToDisplay.forEach((e) => {
        if (e.location && Array.isArray(e.location.coordinates) && e.location.coordinates.length === 2) {
          nodes.push({ id: e.id, lat: e.location.coordinates[1], lng: e.location.coordinates[0], name: e.name });
        }
      });

      if (nodes.length < 2) {
        setTspPath([]);
        setTspDistanceKm(0);
        return;
      }

      setIsCalculatingTsp(true);
      try {
        const ordered = await solveTsp(nodes);
        setTspPath(ordered.map((p) => [p.lat, p.lng]));
        setTspDistanceKm(calculatePathDistance(ordered));
      } catch (err) {
        console.error('TSP calculation failed:', err);
        setTspPath([]);
        setTspDistanceKm(0);
      } finally {
        setIsCalculatingTsp(false);
      }
    };

    computeTsp();
  }, [isTspPathVisible, erraticsToDisplay, userLocation]);

  return (
    <div className={styles.homePage}>
      <Header />
      <div className={styles.mainContent}>
        <div className={`${styles.filterPanelContainer} ${isFilterPanelCollapsed ? styles.collapsed : ''}`}>
           <FilterPanel 
            filters={filters} 
            onFiltersChange={handleFiltersChange} 
            filterDefinitions={GLOBAL_FILTER_DEFINITIONS_WITH_DATA}
          />
        </div>
        
        {/* Collapse button - always visible */}
        <button 
          className={`${styles.filterCollapseButton} ${isFilterPanelCollapsed ? styles.collapsed : ''}`}
          onClick={() => setIsFilterPanelCollapsed(!isFilterPanelCollapsed)}
          aria-label={isFilterPanelCollapsed ? "Expand filters" : "Collapse filters"}
          title={isFilterPanelCollapsed ? "Expand filters" : "Collapse filters"}
        >
          <span className={styles.collapseIcon}>
            {isFilterPanelCollapsed ? '▶' : '◀'}
          </span>
        </button>
        <div className={styles.mapContainer}>
          {/* TSP Route Controls */}
          <div style={{ 
            position: 'absolute', 
            top: 'var(--spacing-3)', 
            left: isFilterPanelCollapsed ? 'var(--spacing-3)' : '15px',
            transition: 'var(--transition-all)', 
            display: 'flex', 
            gap: 'var(--spacing-3)', 
            alignItems: 'center',
            background: 'linear-gradient(135deg, var(--color-bg-secondary), var(--color-bg-tertiary))',
            padding: 'var(--spacing-3) var(--spacing-4)',
            borderRadius: 'var(--radius-lg)',
            border: 'var(--border-width-1) solid var(--color-neutral-600)',
            boxShadow: 'var(--shadow-xl)',
            backdropFilter: 'blur(12px)',
            zIndex: 'var(--z-index-overlay)',
            pointerEvents: 'auto',
            color: 'var(--color-text-primary)',
            fontFamily: 'var(--font-family-sans)',
            fontSize: 'var(--font-size-sm)',
            fontWeight: 'var(--font-weight-medium)',
            width: 'auto', 
            height: 'auto',
            minWidth: '320px'
          }}>
            <label style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 'var(--spacing-2)',
              cursor: 'pointer',
              color: 'var(--color-text-secondary)',
              transition: 'var(--transition-colors)'
            }}>
              <input
                type="checkbox"
                checked={isTspPathVisible}
                onChange={(e) => setIsTspPathVisible(e.target.checked)}
                style={{
                  accentColor: 'var(--color-primary-500)',
                  width: '16px',
                  height: '16px'
                }}
              />
              <span>Show Optimal Route</span>
            </label>
            <button
              style={{ 
                padding: 'var(--spacing-2) var(--spacing-3)', 
                cursor: 'pointer', 
                borderRadius: 'var(--radius-md)', 
                border: 'var(--border-width-1) solid var(--color-primary-600)', 
                background: 'linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700))',
                color: 'var(--color-text-primary)',
                fontSize: 'var(--font-size-xs)',
                fontWeight: 'var(--font-weight-medium)',
                fontFamily: 'var(--font-family-sans)',
                transition: 'var(--transition-all)',
                boxShadow: 'var(--shadow-sm)',
                minWidth: '80px'
              }}
              onClick={requestUserLocation}
              onMouseEnter={(e) => {
                e.target.style.background = 'linear-gradient(135deg, var(--color-primary-700), var(--color-primary-800))';
                e.target.style.transform = 'translateY(-1px)';
                e.target.style.boxShadow = 'var(--shadow-md)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = 'linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700))';
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = 'var(--shadow-sm)';
              }}
            >
              {isRequestingLocation ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-1)' }}>
                  <span style={{ 
                    display: 'inline-block',
                    width: '12px',
                    height: '12px',
                    border: '2px solid var(--color-text-primary)',
                    borderRadius: '50%',
                    borderTopColor: 'transparent',
                    animation: 'spin 1s linear infinite'
                  }}></span>
                  Locating…
                </span>
              ) : '📍 Locate Me'}
            </button>
            {isTspPathVisible && isCalculatingTsp && (
              <span style={{ 
                color: 'var(--color-text-muted)', 
                fontSize: 'var(--font-size-xs)',
                display: 'flex', 
                alignItems: 'center', 
                gap: 'var(--spacing-2)'
              }}>
                <span style={{ 
                  display: 'inline-block',
                  width: '14px',
                  height: '14px',
                  border: '2px solid var(--color-primary-500)',
                  borderRadius: '50%',
                  borderTopColor: 'transparent',
                  animation: 'spin 1s linear infinite'
                }}></span>
                Calculating route…
              </span>
            )}
            {isTspPathVisible && !isCalculatingTsp && tspPath.length > 1 && (
              <span style={{ 
                fontWeight: 'var(--font-weight-semibold)',
                color: 'var(--color-accent-gold)',
                fontSize: 'var(--font-size-sm)',
                padding: 'var(--spacing-1) var(--spacing-2)',
                background: 'var(--color-bg-primary)',
                borderRadius: 'var(--radius-md)',
                border: 'var(--border-width-1) solid var(--color-neutral-600)'
              }}>
                🛣️ {tspDistanceKm.toFixed(2)} km
              </span>
            )}
            {locationError && (
              <span style={{ 
                color: 'var(--color-error-300)', 
                maxWidth: '200px',
                fontSize: 'var(--font-size-xs)',
                padding: 'var(--spacing-1) var(--spacing-2)',
                background: 'var(--color-error-950)',
                borderRadius: 'var(--radius-md)',
                border: 'var(--border-width-1) solid var(--color-error-600)'
              }}>
                ⚠️ {locationError}
              </span>
            )}
          </div>
          {isLoading && (
            <div className={styles.loadingText}>
              Loading geological survey data...
            </div>
          )}
          {error && (
            <div className={styles.errorText}>
              Error: {error}
            </div>
          )}
          {!isLoading && !error && Array.isArray(allErraticData) && (
            <ErraticsMap 
              erratics={erraticsToDisplay} 
              userLocation={userLocation}
              tspPath={tspPath}
            />
          )}
          {!isLoading && !error && Array.isArray(allErraticData) && allErraticData.length > 0 && erraticsToDisplay.length === 0 && activeFilters.length > 0 && (
            <div className={styles.noResultsText}>
              No erratics match your current filter criteria.
            </div>
          )}
           {!isLoading && !error && Array.isArray(allErraticData) && allErraticData.length === 0 && (
            <div className={styles.noResultsText}>
              No erratics data found in database.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default HomePage; 