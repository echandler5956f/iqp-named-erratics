import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import ErraticsMap from '../components/Map/ErraticsMap';
import Header from '../components/Layout/Header';
import FilterPanel from '../components/FilterSystem/FilterPanel';
import { getVisibleErraticIds } from '../components/FilterSystem/filterUtils';
import styles from './HomePage.module.css'; // Using CSS Modules for HomePage as well

// Moved from FilterPanel.jsx to be managed at HomePage level or context
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
    label: 'Accessibility Score (1-5)',
    defaultConfig: { min: null, max: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="minAccessibility">Min:</label>
        <input
          type="number"
          id="minAccessibility"
          min="1" max="5"
          value={config.min ?? ''}
          onChange={(e) => onChange({ ...config, min: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="1-5"
        />
        <label htmlFor="maxAccessibility">Max:</label>
        <input
          type="number"
          id="maxAccessibility"
          min="1" max="5"
          value={config.max ?? ''}
          onChange={(e) => onChange({ ...config, max: e.target.value ? parseInt(e.target.value, 10) : null })}
          placeholder="1-5"
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
          {distinctTerrainLandforms.map(type => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>
    ),
  },
  nearest_colonial_road_dist: {
    label: 'Proximity to Colonial Road (m)',
    defaultConfig: { maxDist: null },
    Component: ({ config, onChange, styles: panelStyles }) => (
      <div className={panelStyles.filterConfigRow}>
        <label htmlFor="maxDistColonialRoad">Max Distance:</label>
        <input
          type="number"
          id="maxDistColonialRoad"
          value={config.maxDist ?? ''}
          onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
          placeholder="Max distance in meters"
        />
      </div>
    ),
  },
};

function HomePage() {
  const [allErraticData, setAllErraticData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState([]); // User-defined filter instances

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
        setError(err.message || 'Failed to fetch erratics.');
        setAllErraticData([]);
      }
      setIsLoading(false);
    };
    fetchErratics();
  }, []);

  // --- Calculate Distinct Values for Dropdowns ---
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
    // Assuming terrain_landform is a direct property on the erratic object, potentially from a join with ErraticAnalysis
    const landforms = new Set(allErraticData.map(e => e.terrain_landform).filter(Boolean)); 
    return Array.from(landforms).sort();
  }, [allErraticData]);
  // --- End Distinct Values ---

  // --- DEFINE GLOBAL_FILTER_DEFINITIONS FIRST --- 
  const GLOBAL_FILTER_DEFINITIONS = useMemo(() => ({
    size: {
      label: 'Size (meters)',
      defaultConfig: { min: null, max: null },
      Component: ({ config, onChange, styles: panelStyles }) => ( 
        <div className={panelStyles.filterConfigRow}> 
          <label htmlFor="minSize">Min:</label>
          <input type="number" id="minSize" value={config.min ?? ''} onChange={(e) => onChange({ ...config, min: e.target.value ? parseFloat(e.target.value) : null })} placeholder="Min size" />
          <label htmlFor="maxSize">Max:</label>
          <input type="number" id="maxSize" value={config.max ?? ''} onChange={(e) => onChange({ ...config, max: e.target.value ? parseFloat(e.target.value) : null })} placeholder="Max size" />
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
    rock_type: {
      label: 'Rock Type',
      defaultConfig: { type: '' },
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
      label: 'Usage Type Contains',
      defaultConfig: { tag: '' },
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
    has_inscriptions: {
      label: 'Has Inscriptions',
      defaultConfig: { required: false }, 
      Component: ({ config, onChange, styles: panelStyles }) => {
          const checkboxId = useMemo(() => `inscriptions-${Math.random().toString(36).substring(2, 9)}`, []);
          return (
              <div className={panelStyles.filterConfigRow} style={{ alignItems: 'center' }}>
                <input
                  type="checkbox"
                  id={checkboxId}
                  style={{ marginRight: '8px' }}
                  checked={config.required === true}
                  onChange={(e) => onChange({ ...config, required: e.target.checked })}
                />
                <label htmlFor={checkboxId} style={{ marginBottom: '0', fontWeight: 'normal' }}> 
                  Must have inscriptions
                </label>
              </div>
          );
        }
     },
     accessibility_score: {
      label: 'Accessibility Score (1-5)',
      defaultConfig: { min: null, max: null },
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="minAccessibility">Min:</label>
          <input
            type="number"
            id="minAccessibility"
            min="1" max="5"
            value={config.min ?? ''}
            onChange={(e) => onChange({ ...config, min: e.target.value ? parseInt(e.target.value, 10) : null })}
            placeholder="1-5"
          />
          <label htmlFor="maxAccessibility">Max:</label>
          <input
            type="number"
            id="maxAccessibility"
            min="1" max="5"
            value={config.max ?? ''}
            onChange={(e) => onChange({ ...config, max: e.target.value ? parseInt(e.target.value, 10) : null })}
            placeholder="1-5"
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
            {distinctTerrainLandforms.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
      ),
    },
    nearest_colonial_road_dist: {
      label: 'Proximity to Colonial Road (m)',
      defaultConfig: { maxDist: null },
      Component: ({ config, onChange, styles: panelStyles }) => (
        <div className={panelStyles.filterConfigRow}>
          <label htmlFor="maxDistColonialRoad">Max Distance:</label>
          <input
            type="number"
            id="maxDistColonialRoad"
            value={config.maxDist ?? ''}
            onChange={(e) => onChange({ ...config, maxDist: e.target.value ? parseFloat(e.target.value) : null })}
            placeholder="Max distance in meters"
          />
        </div>
      ),
    },
  }), [distinctRockTypes, distinctUsageTags, distinctTerrainLandforms]);

  // --- HOOKS USING GLOBAL_FILTER_DEFINITIONS --- 
  const activeFilters = useMemo(() => filters.filter(f => f.isActive), [filters]);

  const visibleErraticIds = useMemo(() => {
    if (!Array.isArray(allErraticData)) {
        console.error('visibleErraticIds: allErraticData is not an array!', allErraticData);
        return new Set();
    }
    return getVisibleErraticIds(allErraticData, activeFilters, GLOBAL_FILTER_DEFINITIONS);
  }, [allErraticData, activeFilters, GLOBAL_FILTER_DEFINITIONS]);

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

  return (
    <div className={styles.homePage}>
      <Header />
      <div className={styles.mainContent}>
        <div className={styles.filterPanelContainer}>
           <FilterPanel 
            filters={filters} 
            onFiltersChange={handleFiltersChange} 
            filterDefinitions={GLOBAL_FILTER_DEFINITIONS}
          />
        </div>
        <div className={styles.mapContainer}>
          {isLoading && <p className={styles.loadingText}>Loading map data...</p>}
          {error && <p className={styles.errorText}>Error: {error}</p>}
          {!isLoading && !error && Array.isArray(allErraticData) && (
            <ErraticsMap 
              erratics={erraticsToDisplay} 
            />
          )}
          {!isLoading && !error && Array.isArray(allErraticData) && allErraticData.length > 0 && erraticsToDisplay.length === 0 && activeFilters.length > 0 && (
            <p className={styles.noResultsText}>No erratics match your current filter criteria.</p>
          )}
           {!isLoading && !error && Array.isArray(allErraticData) && allErraticData.length === 0 && (
            <p className={styles.noResultsText}>No erratics data found on the server.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default HomePage; 