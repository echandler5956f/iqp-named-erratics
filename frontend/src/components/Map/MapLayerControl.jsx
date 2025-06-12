import React, { useState, useRef, useEffect } from 'react';
import './MapLayerControl.css';

const MapLayerControl = ({ 
  currentBaseLayer,
  onBaseLayerChange,
  overlayLayers = {}, 
  onOverlayToggle 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const controlRef = useRef(null);

  const baseLayers = [
    { id: 'osm', name: 'OpenStreetMap', description: 'Standard street map' },
    { id: 'satellite', name: 'Satellite', description: 'Aerial imagery' },
    { id: 'topographic', name: 'Topographic', description: 'Terrain and elevation' }
  ];

  const handleBaseLayerSelect = (layerId) => {
    onBaseLayerChange(layerId);
  };

  const handleOverlayToggle = (layerId, enabled) => {
    onOverlayToggle(layerId, enabled);
  };

  // Auto-close when clicking outside the panel
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isExpanded && controlRef.current && !controlRef.current.contains(event.target)) {
        setIsExpanded(false);
      }
    };

    if (isExpanded) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isExpanded]);

  return (
    <div ref={controlRef} className={`map-layer-control unified-control ${isExpanded ? 'expanded' : ''}`}>
      <button 
        className="layer-control-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-label="Toggle map controls"
        title="Map Controls"
      >
        <span className="toggle-icon">☰</span>
      </button>
      
      {isExpanded && (
        <div className="layer-control-panel">
          {/* Base Maps Section */}
          <div className="layer-section">
            <h3 className="section-title">Base Maps</h3>
            <div className="layer-options">
              {baseLayers.map(layer => (
                <label key={layer.id} className="layer-option">
                  <input
                    type="radio"
                    name="baseLayer"
                    value={layer.id}
                    checked={currentBaseLayer === layer.id}
                    onChange={() => handleBaseLayerSelect(layer.id)}
                  />
                  <div className="layer-info">
                    <span className="layer-name">{layer.name}</span>
                    <span className="layer-description">{layer.description}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Overlay Layers Section */}
          <div className="section-separator"></div>
          <div className="layer-section">
            <h3 className="section-title">Visualizations</h3>
            <div className="layer-options">
              <label className="layer-option">
                <input
                  type="checkbox"
                  checked={overlayLayers.glacialFlow || false}
                  onChange={(e) => handleOverlayToggle('glacialFlow', e.target.checked)}
                />
                <div className="layer-info">
                  <span className="layer-name">🌊 Glacial Flow</span>
                  <span className="layer-description">Inferred ice flow patterns and directions</span>
                </div>
              </label>
              
              <label className="layer-option">
                <input
                  type="checkbox"
                  checked={overlayLayers.geologicalTerritories || false}
                  onChange={(e) => handleOverlayToggle('geologicalTerritories', e.target.checked)}
                />
                <div className="layer-info">
                  <span className="layer-name">🗿 Geological Territories</span>
                  <span className="layer-description">Rock type influence zones and dominance</span>
                </div>
              </label>

              <label className="layer-option">
                <input
                  type="checkbox"
                  checked={overlayLayers.uncertaintyVisualization || false}
                  onChange={(e) => handleOverlayToggle('uncertaintyVisualization', e.target.checked)}
                />
                <div className="layer-info">
                  <span className="layer-name">❓ Data Uncertainty</span>
                  <span className="layer-description">Elegant visualization of data quality and research gaps</span>
                </div>
              </label>
            </div>
          </div>

          {/* Data Quality Notice */}
          <div className="section-separator"></div>
          <div className="data-quality-notice">
            <h4>📊 Data Sparsity Solutions</h4>
            <p>89% of erratics lack geological data. The uncertainty visualization transforms this limitation into valuable insights:</p>
            <ul>
              <li>Knowledge gaps show areas needing research</li>
              <li>Confidence connections reveal data relationships</li>
              <li>Quality indicators highlight reliable information</li>
              <li>Uncertainty fields quantify data reliability</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapLayerControl; 