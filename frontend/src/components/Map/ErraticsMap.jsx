import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, LayersControl, ZoomControl, Polyline, CircleMarker, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './ErraticsMap.css';
import MapLayerControl from './MapLayerControl';
import GlacialFlowLayer from './GlacialFlowLayer';
import GeologicalVoronoiLayer from './GeologicalVoronoiLayer';
import UncertaintyVisualizationLayer from './UncertaintyVisualizationLayer';

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

// Utility function to create circular, vignetted erratic icons
const createCircularVignettedIcon = (imageUrl, size = 32) => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    // Set canvas size
    canvas.width = size;
    canvas.height = size;
    
    // Timeout for loading images
    const timeout = setTimeout(() => {
      console.warn(`Image load timeout for: ${imageUrl}`);
      reject(new Error('Image load timeout'));
    }, 10000); // 10 second timeout
    
    img.onload = () => {
      clearTimeout(timeout);
      try {
        // Validate image dimensions
        if (img.width === 0 || img.height === 0) {
          console.warn(`Invalid image dimensions for: ${imageUrl}`);
          reject(new Error('Invalid image dimensions'));
          return;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, size, size);
        
        // Create circular clipping path
        ctx.save();
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
        ctx.clip();
        
        // Draw the image to fill the circle
        const aspectRatio = img.width / img.height;
        let drawWidth, drawHeight, drawX, drawY;
        
        if (aspectRatio > 1) {
          // Image is wider than tall
          drawHeight = size;
          drawWidth = size * aspectRatio;
          drawX = -(drawWidth - size) / 2;
          drawY = 0;
        } else {
          // Image is taller than wide
          drawWidth = size;
          drawHeight = size / aspectRatio;
          drawX = 0;
          drawY = -(drawHeight - size) / 2;
        }
        
        ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
        ctx.restore();
        
        // Apply subtle vignette effect
        const gradient = ctx.createRadialGradient(
          size / 2, size / 2, 0,
          size / 2, size / 2, size / 2
        );
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        gradient.addColorStop(0.7, 'rgba(0, 0, 0, 0)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.3)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);
        
        // Add subtle border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
        ctx.stroke();
        
        // Try to convert canvas to data URL, handle tainted canvas
        let dataUrl;
        try {
          dataUrl = canvas.toDataURL('image/png');
        } catch (securityError) {
          // Canvas is tainted, we can't export it
          // Fall back to using the original image with a CSS-based circular mask
          console.warn(`Canvas tainted for ${imageUrl}, using CSS fallback`);
          
          // Create a CSS-masked icon instead
          const icon = new L.DivIcon({
            html: `<div style="
              width: ${size}px;
              height: ${size}px;
              background-image: url('${imageUrl}');
              background-size: cover;
              background-position: center;
              border-radius: 50%;
              border: 2px solid rgba(255, 255, 255, 0.8);
              box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.3);
            "></div>`,
            iconSize: [size, size],
            iconAnchor: [size / 2, size / 2],
            popupAnchor: [0, -size / 2],
            className: 'circular-erratic-icon-css'
          });
          
          console.log(`Successfully created CSS circular icon for: ${imageUrl}`);
          resolve(icon);
          return;
        }
        
        // Create Leaflet icon with canvas data
        const icon = new L.Icon({
          iconUrl: dataUrl,
          iconSize: [size, size],
          iconAnchor: [size / 2, size / 2],
          popupAnchor: [0, -size / 2],
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
          shadowSize: [size * 1.2, size * 0.8],
          shadowAnchor: [size * 0.6, size * 0.4]
        });
        
        console.log(`Successfully created circular icon for: ${imageUrl}`);
        resolve(icon);
      } catch (error) {
        clearTimeout(timeout);
        console.error(`Canvas processing error for ${imageUrl}:`, error);
        reject(error);
      }
    };
    
    img.onerror = (error) => {
      clearTimeout(timeout);
      console.warn(`Failed to load image: ${imageUrl}`, error);
      reject(new Error(`Failed to load image: ${imageUrl}`));
    };
    
    // Try loading with CORS first, then without
    const attemptLoad = (useCors) => {
      if (useCors) {
        img.crossOrigin = 'anonymous';
      } else {
        img.crossOrigin = null;
      }
      img.src = imageUrl;
    };

    // Start with CORS attempt
    let corsAttempted = false;
    
    const originalOnError = img.onerror;
    img.onerror = (error) => {
      if (!corsAttempted) {
        corsAttempted = true;
        console.warn(`CORS load failed for ${imageUrl}, trying without CORS`);
        // Try again without CORS
        attemptLoad(false);
      } else {
        // Both attempts failed
        originalOnError(error);
      }
    };

    // Start with CORS
    attemptLoad(true);
  });
};

// Cache for processed icons to avoid reprocessing
const iconCache = new Map();

// Function to validate image URL
const isValidImageUrl = (url) => {
  if (!url || typeof url !== 'string') return false;
  
  // Check if it's a valid HTTP/HTTPS URL
  try {
    const urlObj = new URL(url);
    if (!['http:', 'https:'].includes(urlObj.protocol)) return false;
  } catch {
    return false;
  }
  
  // Check for common image extensions in the URL path or query params
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'];
  const lowercaseUrl = url.toLowerCase();
  const hasImageExtension = imageExtensions.some(ext => lowercaseUrl.includes(ext));
  
  // Check for common image hosting patterns and services
  const imageHostPatterns = [
    'imgur.com',
    'flickr.com',
    'wikimedia.org',
    'wikipedia.org',
    'googleusercontent.com',
    'amazonaws.com',
    'bing.com/th/id', // Bing image service
    'geocaching.com',
    'photoshelter.com',
    'staticflickr.com',
    'istockphoto.com',
    'shutterstock.com',
    'alamy.com',
    'gettyimages.com',
    'pinimg.com', // Pinterest images
    'ytimg.com', // YouTube thumbnails
    'blogspot.com',
    'wordpress.com',
    'squarespace-cdn.com',
    'fineartamerica.com',
    'kiddle.co',
    'naturallyamazing.com',
    'newengland.com',
    'oceanlight.com',
    'wanderer.com',
    'njhiking.com',
    'blockislandguide.com',
    'berkeleyca.gov',
    'smokymountains.com',
    'wsu.edu',
    'bidsquare.com',
    'producer.com',
    'editorial01.shutterstock.com'
  ];
  const hasImageHost = imageHostPatterns.some(pattern => lowercaseUrl.includes(pattern));
  
  // If it has an image extension or is from a known image host, it's likely valid
  // Also accept URLs that look like they might be images even without explicit extensions
  return hasImageExtension || hasImageHost || lowercaseUrl.includes('image') || lowercaseUrl.includes('photo');
};

// Function to get or create a circular vignetted icon
const getCircularIcon = async (imageUrl) => {
  if (iconCache.has(imageUrl)) {
    return iconCache.get(imageUrl);
  }
  
  // Validate URL first
  if (!isValidImageUrl(imageUrl)) {
    console.warn('Invalid image URL format:', imageUrl);
    iconCache.set(imageUrl, defaultErraticIcon);
    return defaultErraticIcon;
  }
  
  try {
    const icon = await createCircularVignettedIcon(imageUrl, 32);
    iconCache.set(imageUrl, icon);
    return icon;
  } catch (error) {
    console.warn('Failed to create circular icon for:', imageUrl, error);
    // Cache the failure to avoid retrying
    iconCache.set(imageUrl, defaultErraticIcon);
    return defaultErraticIcon;
  }
};

// ------------------------------------------------------------
//  Map helper – fit bounds to path & user location
// ------------------------------------------------------------
const FitBoundsToPath = ({ path, userLocation }) => {
  const map = useMap();
  useEffect(() => {
    if ((!path || path.length === 0) && !userLocation) return;

    const bounds = [];
    if (Array.isArray(path)) {
      path.forEach((coord) => bounds.push(coord));
    }
    if (userLocation) bounds.push(userLocation);

    if (bounds.length > 0) {
      map.fitBounds(bounds, { padding: [40, 40] });
    }
  }, [path, userLocation, map]);
  return null;
};

function ErraticsMap({ erratics: erraticsToDisplay, userLocation, tspPath }) {
  const [selectedErratic, setSelectedErratic] = useState(null);
  const [currentBaseLayer, setCurrentBaseLayer] = useState('osm');
  const [overlayLayers, setOverlayLayers] = useState({
    glacialFlow: false,
    geologicalTerritories: false,
    uncertaintyVisualization: false
  });
  const [erraticIcons, setErraticIcons] = useState(new Map());

  const handleBaseLayerChange = (layerId) => {
    setCurrentBaseLayer(layerId);
  };

  const handleOverlayToggle = (layerId, enabled) => {
    setOverlayLayers(prev => ({
      ...prev,
      [layerId]: enabled
    }));
  };

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

  // Get theme colors directly from CSS custom properties
  const themePrimaryColor = useMemo(() => {
    return getComputedStyle(document.documentElement).getPropertyValue('--color-accent-navy').trim() || '#1e3a8a';
  }, []);

  const themeAccentBurgundy = useMemo(() => {
    return getComputedStyle(document.documentElement).getPropertyValue('--color-accent-burgundy').trim() || '#800020';
  }, []);

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

  // --- Preload Circular Icons ---
  useEffect(() => {
    if (!erraticsToDisplay) return;

    const loadIcons = async () => {
      const newIcons = new Map(erraticIcons);
      let iconsProcessed = 0;
      let iconsSuccessful = 0;
      let iconsFailed = 0;
      
      console.log(`Starting to process ${erraticsToDisplay.length} erratic icons...`);
      
      // Process icons in batches to avoid overwhelming the browser
      const batchSize = 5;
      for (let i = 0; i < erraticsToDisplay.length; i += batchSize) {
        const batch = erraticsToDisplay.slice(i, i + batchSize);
        
        await Promise.allSettled(
          batch.map(async (erratic) => {
            if (newIcons.has(erratic.id)) {
              return; // Already processed
            }
            
            iconsProcessed++;
            
            if (erratic.image_url && 
                typeof erratic.image_url === 'string' && 
                erratic.image_url.startsWith('http')) {
              
              try {
                console.log(`Processing icon ${iconsProcessed}/${erraticsToDisplay.length}: ${erratic.name}`);
                const circularIcon = await getCircularIcon(erratic.image_url);
                newIcons.set(erratic.id, circularIcon);
                iconsSuccessful++;
              } catch (error) {
                console.warn(`Failed to create circular icon for erratic ${erratic.id} (${erratic.name}):`, error);
                newIcons.set(erratic.id, defaultErraticIcon);
                iconsFailed++;
              }
            } else {
              // No image URL or invalid format
              newIcons.set(erratic.id, defaultErraticIcon);
            }
          })
        );
        
        // Update state after each batch
        setErraticIcons(new Map(newIcons));
        
        // Small delay between batches to prevent browser freezing
        if (i + batchSize < erraticsToDisplay.length) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
      
      console.log(`Icon processing complete: ${iconsSuccessful} successful, ${iconsFailed} failed, ${iconsProcessed} total`);
    };

    loadIcons();
  }, [erraticsToDisplay]);
  // --- End Icon Preloading ---

  return (
    <div className="map-container">
      {/* Unified Map Layer Control */}
      <MapLayerControl 
        currentBaseLayer={currentBaseLayer}
        onBaseLayerChange={handleBaseLayerChange}
        overlayLayers={overlayLayers}
        onOverlayToggle={handleOverlayToggle}
      />
      
      <MapContainer
        key={mapCenter.join('-') + '-' + (erraticsToDisplay ? erraticsToDisplay.length : 0)}
        center={mapCenter}
        zoom={5}
        style={{ height: '100%', width: '100%' }}
        zoomControl={false}
        attributionControl={false}
      >
        <ZoomControl position="bottomright" />

        {/* Fit bounds helper */}
        <FitBoundsToPath path={tspPath} userLocation={userLocation} />

        {/* Render TSP path - lines first, then points for proper z-order */}
        {tspPath && tspPath.length > 1 && (
          <>
            {/* Background glow for the path */}
            <Polyline 
              positions={tspPath} 
              pathOptions={{ 
                color: themePrimaryColor, 
                weight: 6, 
                opacity: 0.25,
                lineCap: 'round',
                lineJoin: 'round'
              }} 
            />
            {/* Main path line */}
            <Polyline 
              positions={tspPath} 
              pathOptions={{ 
                color: themePrimaryColor, 
                weight: 3, 
                opacity: 0.9,
                lineCap: 'round',
                lineJoin: 'round'
              }} 
            />
          </>
        )}

        {/* Render waypoint markers AFTER lines so they appear on top */}
        {tspPath && tspPath.length > 1 && tspPath.map((coord, idx) => (
          <CircleMarker 
            key={`path-waypoint-${idx}`} 
            center={coord} 
            radius={4} 
            pathOptions={{ 
              color: themeAccentBurgundy, 
              fillColor: themeAccentBurgundy, 
              fillOpacity: 1,
              weight: 2,
              opacity: 1
            }}
          >
            <Popup>
              <div style={{ 
                color: 'var(--color-text-primary)', 
                fontFamily: 'var(--font-family-sans)',
                textAlign: 'center'
              }}>
                <strong style={{ color: 'var(--color-accent-burgundy)' }}>Stop {idx + 1}</strong>
                <br />
                <small style={{ color: 'var(--color-text-secondary)' }}>
                  Waypoint {coord[0].toFixed(4)}, {coord[1].toFixed(4)}
                </small>
              </div>
            </Popup>
          </CircleMarker>
        ))}

        {/* User location marker with distinctive styling */}
        {userLocation && (
          <>
            {/* Outer glow ring */}
            <CircleMarker
              center={userLocation}
              radius={12}
              pathOptions={{ 
                color: themeAccentBurgundy, 
                fillColor: themeAccentBurgundy, 
                fillOpacity: 0.2,
                weight: 1,
                opacity: 0.6
              }}
            />
            {/* Main marker */}
            <CircleMarker
              center={userLocation}
              radius={7}
              pathOptions={{ 
                color: themeAccentBurgundy, 
                fillColor: themeAccentBurgundy, 
                fillOpacity: 1,
                weight: 3,
                opacity: 1
              }}
            >
              <Popup>
                <div style={{ 
                  color: 'var(--color-text-primary)', 
                  fontFamily: 'var(--font-family-sans)',
                  textAlign: 'center'
                }}>
                  <strong style={{ color: 'var(--color-accent-burgundy)' }}>📍 Your Location</strong>
                  <br />
                  <small style={{ color: 'var(--color-text-secondary)' }}>
                    {userLocation[0].toFixed(4)}, {userLocation[1].toFixed(4)}
                  </small>
                </div>
              </Popup>
            </CircleMarker>
          </>
        )}

        {/* Base Layer - controlled by our unified control */}
        {currentBaseLayer === 'osm' && (
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution=""
            />
        )}
        {currentBaseLayer === 'satellite' && (
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution=""
            />
        )}
        {currentBaseLayer === 'topographic' && (
            <TileLayer
              url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
              attribution=""
              maxZoom={17}
            />
        )}

        {/* Visualization Layers */}
        <GlacialFlowLayer 
          erratics={erraticsToDisplay}
          isVisible={overlayLayers.glacialFlow}
        />
        <GeologicalVoronoiLayer 
          erratics={erraticsToDisplay}
          isVisible={overlayLayers.geologicalTerritories}
        />
        <UncertaintyVisualizationLayer 
          erratics={erraticsToDisplay}
          isVisible={overlayLayers.uncertaintyVisualization}
        />

        {/* Render erratics with circular, vignetted icons */}
        {erraticsToDisplay && erraticsToDisplay.map(erratic => {
          // Get the preloaded circular icon or use default
          const markerIcon = erraticIcons.get(erratic.id) || defaultErraticIcon;

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
              icon={markerIcon}
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
                    {erratic.rock_type && <p><strong>Rock Type: </strong> {erratic.rock_type}</p>}
                    {erratic.size_meters && <p><strong>Size: </strong> {erratic.size_meters} meters</p>}
                    {erratic.elevation && <p><strong>Elevation: </strong> {erratic.elevation} meters</p>}
                    {erratic.estimated_age && <p><strong>Age: </strong> {erratic.estimated_age}</p>}
                    {/* Removed short description from popup; full details in sidebar */}
                    {userLocation && erratic.location && erratic.location.coordinates && (
                        <p><strong>Distance: </strong> 
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
                <div className="detail-section"><h3>Accessibility Score</h3><p>{selectedErratic.accessibility_score} / 10</p></div>
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