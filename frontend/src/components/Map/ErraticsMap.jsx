import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, LayersControl, ZoomControl, useMap } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import './ErraticsMap.css';

// Fix the default icon issue in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png'
});

// Custom erratic icon
const erraticIcon = new L.Icon({
  iconUrl: '/erratic-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  shadowSize: [41, 41]
});

// Component to handle user's current location
const LocationMarker = ({ setUserLocation }) => {
  const map = useMap();

  useEffect(() => {
    map.locate({ setView: false });
    
    map.on('locationfound', (e) => {
      setUserLocation([e.latlng.lat, e.latlng.lng]);
      map.flyTo(e.latlng, 12);
    });
    
    map.on('locationerror', () => {
      console.log('Location access denied or not available.');
    });
  }, [map, setUserLocation]);

  return null;
};

function ErraticsMap() {
  const [erratics, setErratics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  const [mapCenter, setMapCenter] = useState([40, -100]); // Default center of USA
  const [selectedErratic, setSelectedErratic] = useState(null);

  // Fetch erratics data from the API
  useEffect(() => {
    const fetchErratics = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:3001/api/erratics');
        setErratics(response.data);
        
        // If we have data, center map on the first erratic
        if (response.data.length > 0) {
          const firstErratic = response.data[0];
          setMapCenter([
            firstErratic.location.coordinates[1], 
            firstErratic.location.coordinates[0]
          ]);
        }
      } catch (err) {
        setError('Failed to fetch erratic data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchErratics();
  }, []);

  // Fetch nearby erratics when user location is available
  useEffect(() => {
    if (!userLocation) return;
    
    const fetchNearbyErratics = async () => {
      try {
        const [lat, lng] = userLocation;
        const response = await axios.get(`http://localhost:3001/api/erratics/nearby?lat=${lat}&lng=${lng}&radius=100`);
        setErratics(response.data);
      } catch (err) {
        console.error('Failed to fetch nearby erratics', err);
      }
    };
    
    fetchNearbyErratics();
  }, [userLocation]);

  if (loading) return <div className="loading">Loading map data...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="map-container">
      <MapContainer 
        center={mapCenter} 
        zoom={5} 
        style={{ height: '100vh', width: '100%' }}
        zoomControl={false}
      >
        <ZoomControl position="bottomright" />
        <LocationMarker setUserLocation={setUserLocation} />
        
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution='&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer name="Terrain">
            <TileLayer
              url="https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png"
              attribution='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.Overlay checked name="Erratics">
            {erratics.map(erratic => (
              <Marker 
                key={erratic.id} 
                position={[
                  erratic.location.coordinates[1], 
                  erratic.location.coordinates[0]
                ]}
                icon={erraticIcon}
                eventHandlers={{
                  click: () => {
                    setSelectedErratic(erratic);
                  }
                }}
              >
                <Popup>
                  <div className="erratic-popup">
                    <h3>{erratic.name}</h3>
                    {erratic.image_url && (
                      <img 
                        src={erratic.image_url} 
                        alt={erratic.name} 
                        className="erratic-image"
                      />
                    )}
                    <div className="erratic-details">
                      {erratic.rock_type && (
                        <p><strong>Rock Type:</strong> {erratic.rock_type}</p>
                      )}
                      {erratic.size_meters && (
                        <p><strong>Size:</strong> {erratic.size_meters} meters</p>
                      )}
                      {erratic.elevation && (
                        <p><strong>Elevation:</strong> {erratic.elevation} meters</p>
                      )}
                      {erratic.estimated_age && (
                        <p><strong>Age:</strong> {erratic.estimated_age}</p>
                      )}
                      {erratic.description && (
                        <p>{erratic.description}</p>
                      )}
                      {erratic.distance && (
                        <p><strong>Distance from you:</strong> {erratic.distance.toFixed(1)} km</p>
                      )}
                    </div>
                    <button className="view-details-btn" onClick={() => setSelectedErratic(erratic)}>
                      View Details
                    </button>
                  </div>
                </Popup>
              </Marker>
            ))}
          </LayersControl.Overlay>
        </LayersControl>
      </MapContainer>
      
      {selectedErratic && (
        <div className="erratic-sidebar">
          <button className="close-btn" onClick={() => setSelectedErratic(null)}>×</button>
          <h2>{selectedErratic.name}</h2>
          
          {selectedErratic.image_url && (
            <img 
              src={selectedErratic.image_url} 
              alt={selectedErratic.name} 
              className="erratic-full-image"
            />
          )}
          
          <div className="erratic-full-details">
            {selectedErratic.rock_type && (
              <p><strong>Rock Type:</strong> {selectedErratic.rock_type}</p>
            )}
            {selectedErratic.size_meters && (
              <p><strong>Size:</strong> {selectedErratic.size_meters} meters</p>
            )}
            {selectedErratic.elevation && (
              <p><strong>Elevation:</strong> {selectedErratic.elevation} meters</p>
            )}
            {selectedErratic.estimated_age && (
              <p><strong>Estimated Age:</strong> {selectedErratic.estimated_age}</p>
            )}
            {selectedErratic.discovery_date && (
              <p><strong>Discovered:</strong> {new Date(selectedErratic.discovery_date).toLocaleDateString()}</p>
            )}
            
            {selectedErratic.description && (
              <div className="detail-section">
                <h3>Description</h3>
                <p>{selectedErratic.description}</p>
              </div>
            )}
            
            {selectedErratic.cultural_significance && (
              <div className="detail-section">
                <h3>Cultural Significance</h3>
                <p>{selectedErratic.cultural_significance}</p>
              </div>
            )}
            
            {selectedErratic.historical_notes && (
              <div className="detail-section">
                <h3>Historical Notes</h3>
                <p>{selectedErratic.historical_notes}</p>
              </div>
            )}
            
            {selectedErratic.distance && (
              <p><strong>Distance from your location:</strong> {selectedErratic.distance.toFixed(1)} km</p>
            )}
            
            <div className="coordinates">
              <p>
                <strong>Coordinates:</strong> {selectedErratic.location.coordinates[1]}, {selectedErratic.location.coordinates[0]}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ErraticsMap; 