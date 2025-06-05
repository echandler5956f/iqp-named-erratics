import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, LayersControl, ZoomControl, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './ErraticsMap.css';

// Fix the default icon issue in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png'
});

// Define a default custom erratic icon (ensure /erratic-icon.png is in the public folder)
const defaultErraticIcon = new L.Icon({
  iconUrl: '/erratic-icon.png', 
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  shadowSize: [41, 41]
});

// Custom erratic icon
// const erraticIcon = new L.Icon({
//   iconUrl: '/erratic-icon.png',
//   iconSize: [25, 41],
//   iconAnchor: [12, 41],
//   popupAnchor: [1, -34],
//   shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
//   shadowSize: [41, 41]
// });

// Component to handle user's current location
const LocationMarker = ({ setUserLocation }) => {
  const map = useMap();

  useEffect(() => {
    map.locate({ setView: false });
    
    map.on('locationfound', (e) => {
      setUserLocation([e.latlng.lat, e.latlng.lng]);
      console.log('User location found:', e.latlng);
    });
    
    map.on('locationerror', (err) => {
      console.warn('Location access error:', err.message);
    });

    // Cleanup
    return () => {
      map.off('locationfound');
      map.off('locationerror');
    };
  }, [map, setUserLocation]);

  return null;
};

function ErraticsMap({ erratics: erraticsToDisplay }) {
  const [userLocation, setUserLocation] = useState(null);
  const [selectedErratic, setSelectedErratic] = useState(null);

  const mapCenter = useMemo(() => {
    if (erraticsToDisplay && erraticsToDisplay.length > 0) {
      const firstErratic = erraticsToDisplay[0];
      if (firstErratic.location && firstErratic.location.coordinates) {
        return [
          firstErratic.location.coordinates[1], 
          firstErratic.location.coordinates[0]
        ];
      }
    }
    return [40, -100]; // Default center of USA if no erratics or first has no location
  }, [erraticsToDisplay]);

  // --- ESC Key Handler for Sidebar ---
  useEffect(() => {
    const handleEsc = (event) => {
      if (event.key === 'Escape') {
        setSelectedErratic(null); // Close sidebar
      }
    };
    // Add listener only when sidebar is open
    if (selectedErratic) {
      document.addEventListener('keydown', handleEsc);
    }
    // Cleanup function to remove listener
    return () => {
      document.removeEventListener('keydown', handleEsc);
    };
  }, [selectedErratic]); // Re-run effect when selectedErratic changes
  // --- End ESC Key Handler ---

  return (
    <div className="map-container">
      <MapContainer 
        key={mapCenter.join('-') + '-' + (erraticsToDisplay ? erraticsToDisplay.length : 0)} // Add erratic length to key for re-render on data change
        center={mapCenter} 
        zoom={5} 
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        attributionControl={false} // Hide attribution control
      >
        <ZoomControl position="bottomright" />
        <LocationMarker setUserLocation={setUserLocation} />
        
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution=""
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution=""
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer name="Topographic">
            <TileLayer
              url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
              attribution=""
              maxZoom={17}
            />
          </LayersControl.BaseLayer>
        </LayersControl>
        
        {/* Render erratics directly on map without layers control overlay */}
        {erraticsToDisplay && erraticsToDisplay.map(erratic => {
          let markerIcon = defaultErraticIcon; // Default to custom icon

          if (erratic.image_url) {
            try {
              // Attempt to create an icon from the erratic's image_url
              // Basic validation: check if it's a string and looks like a URL (very basic check)
              if (typeof erratic.image_url === 'string' && erratic.image_url.startsWith('http')) {
                markerIcon = new L.Icon({
                  iconUrl: erratic.image_url,
                  iconSize: [25, 41],
                  iconAnchor: [12, 41],
                  popupAnchor: [1, -34],
                  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
                  shadowSize: [41, 41]
                });
              } else if (typeof erratic.image_url === 'string' && erratic.image_url) {
                // If it's a string but not a full URL, it might be a local path, but less safe
                // For now, only http/https URLs are used for per-erratic icons to avoid issues
                // console.warn(`Erratic ${erratic.id} has an image_url that is not a full URL: ${erratic.image_url}. Using default icon.`);
              }
            } catch (e) {
              console.warn(`Error creating icon for erratic ${erratic.id} with image_url: ${erratic.image_url}`, e);
              // Fallback to defaultErraticIcon is already handled by initial assignment
            }
          }

          if (!erratic.location || !erratic.location.coordinates || erratic.location.coordinates.length < 2) {
            console.warn('Erratic with missing or invalid location:', erratic.id, erratic.name);
            return null;
          }

          return (
            <Marker 
              key={erratic.id} 
              position={[
                erratic.location.coordinates[1], 
                erratic.location.coordinates[0]
              ]}
              icon={markerIcon} // This should always be a valid L.Icon instance now
              eventHandlers={{
                click: () => {
                  setSelectedErratic(erratic);
                }
              }}
            >
              <Popup>
                <div className="erratic-popup">
                  <h3>{erratic.name}</h3>
                  {/* Show image in popup if URL exists, regardless of icon source */}
                  {erratic.image_url && (typeof erratic.image_url === 'string' && erratic.image_url.startsWith('http')) && (
                    <img 
                      src={erratic.image_url} 
                      alt={erratic.name} 
                      className="erratic-image"
                      onError={(e) => { e.target.style.display = 'none'; }} // Hide if image fails to load
                    />
                  )}
                  <div className="erratic-details">
                    {erratic.rock_type && <p><strong>Rock Type:</strong> {erratic.rock_type}</p>}
                    {erratic.size_meters && <p><strong>Size:</strong> {erratic.size_meters} meters</p>}
                    {erratic.elevation && <p><strong>Elevation:</strong> {erratic.elevation} meters</p>}
                    {erratic.estimated_age && <p><strong>Age:</strong> {erratic.estimated_age}</p>}
                    {/* Removed short description from popup; full details in sidebar */}
                    {userLocation && erratic.location && erratic.location.coordinates && (
                        <p><strong>Distance:</strong> 
                        { (L.latLng(userLocation).distanceTo(L.latLng(erratic.location.coordinates[1], erratic.location.coordinates[0])) / 1000).toFixed(1) } km
                        </p>
                    )}
                  </div>
                  <button className="view-details-btn" onClick={() => setSelectedErratic(erratic)}>
                    View Details
                  </button>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
      
      {selectedErratic && (
        <div className="erratic-sidebar">
          <button className="close-btn" onClick={() => setSelectedErratic(null)}>×</button>
          <div key={selectedErratic.id} className="sidebar-content-wrapper">
            <h2>{selectedErratic.name}</h2>
            {selectedErratic.image_url && (typeof selectedErratic.image_url === 'string' && selectedErratic.image_url.startsWith('http')) && (
              <img 
                src={selectedErratic.image_url} 
                alt={selectedErratic.name} 
                className="erratic-full-image"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            )}
            <div className="erratic-full-details">
              {selectedErratic.rock_type && <p><strong>Rock Type:</strong> {selectedErratic.rock_type}</p>}
              {selectedErratic.size_meters && <p><strong>Size:</strong> {selectedErratic.size_meters} meters</p>}
              {selectedErratic.elevation && <p><strong>Elevation:</strong> {selectedErratic.elevation} meters</p>}
              {selectedErratic.estimated_age && <p><strong>Estimated Age:</strong> {selectedErratic.estimated_age}</p>}
              {selectedErratic.discovery_date && <p><strong>Discovered:</strong> {new Date(selectedErratic.discovery_date).toLocaleDateString()}</p>}
              {selectedErratic.description && <div className="detail-section"><h3>Description</h3><p>{selectedErratic.description}</p></div>}
              {selectedErratic.cultural_significance && <div className="detail-section"><h3>Cultural Significance</h3><p>{selectedErratic.cultural_significance}</p></div>}
              {selectedErratic.historical_notes && <div className="detail-section"><h3>Historical Notes</h3><p>{selectedErratic.historical_notes}</p></div>}
              
              { /* --- Analysis Fields --- */ }
              { typeof selectedErratic.accessibility_score === 'number' && (
                <div className="detail-section"><h3>Accessibility Score</h3><p>{selectedErratic.accessibility_score} / 5</p></div>
              )}
              { selectedErratic.terrain_landform && (
                <div className="detail-section"><h3>Terrain Landform</h3><p>{selectedErratic.terrain_landform}</p></div>
              )}
              { selectedErratic.usage_type && Array.isArray(selectedErratic.usage_type) && selectedErratic.usage_type.length > 0 && (
                <div className="detail-section"><h3>Usage Types</h3><p>{selectedErratic.usage_type.join(', ')}</p></div>
              )}
              { typeof selectedErratic.has_inscriptions === 'boolean' && (
                <div className="detail-section"><h3>Has Inscriptions</h3><p>{selectedErratic.has_inscriptions ? 'Yes' : 'No'}</p></div>
              )}
              { /* --- End Analysis Fields --- */ }

              {selectedErratic.location && selectedErratic.location.coordinates && (
                <div className="coordinates">
                  <p><strong>Coordinates:</strong> {selectedErratic.location.coordinates[1].toFixed(4)}, {selectedErratic.location.coordinates[0].toFixed(4)}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ErraticsMap; 