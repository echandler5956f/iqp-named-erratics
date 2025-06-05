# Glacial Erratics Map - Frontend

A modern React application providing an interactive geospatial interface for exploring named glacial erratics across North America, featuring real-time route optimization, advanced filtering, and comprehensive spatial data visualization.

## Overview

The frontend is built with React and Vite, leveraging Leaflet.js for interactive mapping capabilities. It features a sophisticated filtering system, real-time TSP route optimization, and a responsive design that works across desktop and mobile devices.

## Architecture

### Technology Stack

- **Framework**: React 18+ with functional components and hooks
- **Build Tool**: Vite for fast development and optimized builds
- **Mapping**: Leaflet.js via React-Leaflet for interactive maps
- **HTTP Client**: Axios for API communication
- **Styling**: CSS Modules with custom design tokens
- **State Management**: React built-in state (useState, useEffect, useMemo)
- **Geolocation**: Browser Geolocation API
- **Route Optimization**: Custom TSP solver with Haversine distance calculations

### Project Structure

```
frontend/
├── public/                     # Static assets
│   ├── index.html             # HTML entry point
│   └── erratic-icon.png       # Default erratic marker icon
├── src/
│   ├── assets/                # Application assets
│   ├── components/            # Reusable React components
│   │   ├── Auth/             # Authentication components
│   │   ├── FilterSystem/     # Dynamic filtering system
│   │   │   ├── FilterPanel.jsx
│   │   │   ├── FilterItem.jsx
│   │   │   ├── AddEditFilterModal.jsx
│   │   │   └── filterUtils.js
│   │   ├── Layout/           # Layout components
│   │   │   └── Header.jsx
│   │   ├── Map/              # Map-related components
│   │   │   ├── ErraticsMap.jsx
│   │   │   └── ErraticsMap.css
│   │   ├── ThematicMap/      # Thematic mapping components
│   │   └── ui/               # Base UI components
│   │       ├── Button/
│   │       └── Card/
│   ├── pages/                # Page-level components
│   │   ├── HomePage.jsx
│   │   └── HomePage.module.css
│   ├── services/             # Business logic and API services
│   │   └── tspService.js     # TSP algorithm implementation
│   ├── styles/               # Global styles and design system
│   │   ├── design-tokens.css
│   │   ├── global.css
│   │   └── README.md
│   ├── App.jsx               # Root component
│   ├── App.css              # Root styles
│   └── main.jsx             # Application entry point
├── package.json             # Dependencies and scripts
└── vite.config.js          # Vite configuration
```

## Core Components

### HomePage (`src/pages/HomePage.jsx`)

The main application container that orchestrates data fetching, filtering, and route optimization.

**Key Responsibilities:**
- Fetches erratic data from the backend API
- Manages filtering state and filter definitions
- Handles user geolocation
- Coordinates TSP route calculation
- Provides data to child components

**State Management:**
```javascript
// Data state
const [allErraticData, setAllErraticData] = useState([]);
const [isLoading, setIsLoading] = useState(true);
const [error, setError] = useState(null);

// Filtering state
const [filters, setFilters] = useState([]);

// User location & TSP state
const [userLocation, setUserLocation] = useState(null);
const [locationError, setLocationError] = useState(null);
const [isTspPathVisible, setIsTspPathVisible] = useState(false);
const [tspPath, setTspPath] = useState([]);
const [isCalculatingTsp, setIsCalculatingTsp] = useState(false);
const [tspDistanceKm, setTspDistanceKm] = useState(0);
const [isRequestingLocation, setIsRequestingLocation] = useState(false);
```

**Filter Definitions:**
The component defines comprehensive filter types supporting various data attributes:
- Size (range filtering)
- Proximity to water bodies, settlements, roads, trails, native territories
- Rock type, usage type, terrain characteristics
- Cultural significance scores
- Discovery dates and geological ages
- Elevation and displacement distances

### ErraticsMap (`src/components/Map/ErraticsMap.jsx`)

Interactive Leaflet map component displaying erratics, user location, and TSP routes.

**Features:**
- Multiple base layers (OpenStreetMap, Satellite, Topographic)
- Custom erratic markers with popup information
- User location marker
- TSP route visualization with polylines and waypoints
- Detailed sidebar for selected erratics
- Responsive design with mobile support

**Key Props:**
```javascript
ErraticsMap.propTypes = {
  erratics: PropTypes.array.isRequired,     // Filtered erratic data
  userLocation: PropTypes.array,            // [lat, lng] or null
  tspPath: PropTypes.array                  // Array of [lat, lng] coordinates
};
```

**Visual Elements:**
- **Erratic Markers**: Custom icons, click for popup/sidebar
- **TSP Route**: Blue polyline with circular waypoint markers
- **User Location**: Distinct marker showing current position
- **Auto-fit Bounds**: Automatically adjusts view to show route
- **Layer Controls**: Toggle between different map tile layers

### FilterPanel (`src/components/FilterSystem/FilterPanel.jsx`)

Dynamic filtering interface allowing users to add, edit, and manage multiple filters.

**Architecture:**
- **Filter Definitions**: Centralized in `HomePage.jsx`
- **Filter Items**: Individual filter displays with toggle/edit/delete
- **Modal Interface**: Add/edit filters via modal dialog
- **Dynamic Components**: Filter configuration UIs are dynamically rendered

**Supported Filter Types:**
- Range filters (min/max values)
- Category selections (dropdowns)
- Boolean toggles
- Multi-select options
- Date range filters

### TSP Service (`src/services/tspService.js`)

Pure JavaScript implementation of a Traveling Salesman Problem solver optimized for geospatial data.

**Algorithm:**
1. **Nearest Neighbor Construction**: Creates initial tour visiting closest unvisited points
2. **2-opt Improvement**: Iteratively improves tour by swapping edges that reduce total distance
3. **Haversine Distance**: Uses great-circle distance calculations for geographic accuracy

**Performance:**
- Handles 200+ points in sub-second time
- O(n²) complexity with practical optimizations
- Asynchronous interface for UI responsiveness

**API:**
```javascript
// Solve TSP for array of points
const orderedPath = await solveTsp(points);

// Calculate total path distance
const distanceKm = calculatePathDistance(orderedPath);
```

## Filtering System

### Architecture

The filtering system provides a flexible, extensible framework for data filtering with real-time TSP integration.

**Flow:**
1. **Filter Definitions** (`GLOBAL_FILTER_DEFINITIONS`) define available filter types
2. **Active Filters** stored in component state as configuration objects
3. **Filter Logic** in `filterUtils.js` applies filters to erratic data
4. **Visible Erratics** computed via `useMemo` for performance
5. **TSP Recalculation** triggered automatically when visible set changes

### Filter Types

**Range Filters:**
```javascript
size: {
  label: 'Size (meters)',
  defaultConfig: { min: null, max: null },
  Component: RangeFilterComponent
}
```

**Category Filters:**
```javascript
rock_type: {
  label: 'Rock Type',
  defaultConfig: { type: '' },
  Component: SelectFilterComponent
}
```

**Boolean Filters:**
```javascript
has_inscriptions: {
  label: 'Has Inscriptions',
  defaultConfig: { required: true },
  Component: BooleanFilterComponent
}
```

### Adding New Filters

1. **Define Filter** in `GLOBAL_FILTER_DEFINITIONS`:
```javascript
new_attribute: {
  label: 'Display Name',
  defaultConfig: { /* default values */ },
  Component: ({ config, onChange, styles }) => (
    // Filter UI component
  )
}
```

2. **Add Filter Logic** in `filterUtils.js`:
```javascript
case 'new_attribute':
  // Implement filtering logic
  break;
```

3. **Ensure Data Availability** in API responses and component state.

## State Management

### Data Flow

```
API Response → allErraticData (HomePage)
              ↓
Filter Definitions + Active Filters → Visible Erratics
              ↓
User Location + Visible Erratics → TSP Calculation
              ↓
TSP Path → Map Visualization
```

### Performance Optimizations

**Memoization:**
- `useMemo` for expensive computations (filtering, distinct values)
- `useCallback` for stable event handlers
- Computed properties cached until dependencies change

**Debouncing:**
- Filter changes trigger immediate UI updates
- TSP recalculation happens on next render cycle
- Loading states prevent UI blocking

## Responsive Design

### Breakpoints

- **Desktop**: `> 1024px` - Full sidebar layout
- **Tablet**: `768px - 1024px` - Compressed sidebar
- **Mobile**: `< 768px` - Stacked layout with collapsible filter panel

### Mobile Adaptations

- **Touch-friendly Controls**: Larger buttons and touch targets
- **Swipe Gestures**: Map navigation optimized for touch
- **Compact Filter Panel**: Collapsible with reduced height
- **Route Controls**: Persistent overlay for easy access

## API Integration

### Endpoints Used

```javascript
// Fetch all erratics with analysis data
GET /api/erratics

// Example response structure
{
  "id": 1,
  "name": "Plymouth Rock",
  "location": {
    "type": "Point",
    "coordinates": [-70.6620, 41.9584]
  },
  "size_meters": 3.0,
  "rock_type": "Dedham Granite",
  "elevation": 2.0,
  "description": "...",
  // Analysis data joined
  "nearest_water_body_dist": 15.2,
  "accessibility_score": 5,
  "usage_type": ["ceremonial", "landmark"],
  "has_inscriptions": false
  // ... other fields
}
```

### Error Handling

- **Network Errors**: Graceful fallback with user feedback
- **Data Validation**: Client-side validation of API responses
- **Loading States**: Visual indicators during API calls
- **Retry Logic**: Automatic retry for transient failures

## Build and Deployment

### Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Configuration

```javascript
// vite.config.js
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:3001'  // Backend proxy
    }
  }
});
```

### Production Optimizations

- **Code Splitting**: Automatic route-based splitting
- **Tree Shaking**: Unused code elimination
- **Asset Optimization**: Image and CSS optimization
- **Bundle Analysis**: Size monitoring and optimization

## Performance Considerations

### Bundle Size

Current build produces:
- **JavaScript**: ~651KB (gzipped: ~185KB)
- **CSS**: ~65KB (gzipped: ~15KB)
- **Recommendation**: Consider code splitting for larger datasets

### Runtime Performance

**TSP Algorithm:**
- **Small datasets** (< 50 points): < 10ms
- **Medium datasets** (50-150 points): 10-100ms
- **Large datasets** (150-300 points): 100ms-1s
- **Very large datasets** (> 300 points): Consider Web Worker implementation

**Rendering Performance:**
- **Map markers**: Efficient clustering for large datasets
- **Filter updates**: Debounced to prevent excessive re-renders
- **Route visualization**: Optimized polyline rendering

## Browser Support

### Minimum Requirements

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

### Feature Dependencies

- **Geolocation API**: Required for user location features
- **ES6+ Features**: Async/await, destructuring, arrow functions
- **CSS Grid/Flexbox**: Layout requirements
- **SVG Support**: Icon rendering

## Testing Strategy

### Recommended Testing Approach

```bash
# Component testing with React Testing Library
npm install --save-dev @testing-library/react @testing-library/jest-dom

# TSP algorithm unit tests
npm install --save-dev vitest

# E2E testing
npm install --save-dev cypress
```

### Key Test Areas

1. **TSP Algorithm**: Distance calculations, route optimization
2. **Filter Logic**: Filter application, edge cases
3. **Map Interactions**: Marker clicks, layer switching
4. **Geolocation**: Permission handling, error states
5. **Responsive Design**: Mobile/desktop layouts

## Security Considerations

### Data Protection

- **No sensitive data** stored in frontend state
- **API authentication** handled via secure HTTP-only cookies
- **HTTPS required** for geolocation in production
- **Input validation** on all user inputs

### Third-party Dependencies

- **Regular updates** for security patches
- **Minimal dependencies** to reduce attack surface
- **Trusted sources** for map tiles and external resources

## Troubleshooting

### Common Issues

**Map not loading:**
- Check network connectivity
- Verify tile server accessibility
- Ensure container has proper dimensions

**Geolocation not working:**
- Requires HTTPS in production
- Check browser permissions
- Handle permission denied gracefully

**TSP calculation slow:**
- Monitor dataset size
- Consider Web Worker for > 300 points
- Check for infinite loops in 2-opt

**Filters not applying:**
- Verify filter logic in `filterUtils.js`
- Check data format consistency
- Ensure proper state updates

### Debug Mode

Enable detailed logging:
```javascript
// In development
localStorage.setItem('debug', 'glacial-erratics:*');
```

## Contributing

### Code Style

- **ESLint configuration** enforced
- **Prettier formatting** required
- **Component patterns** follow established conventions
- **Performance considerations** documented

### Adding Features

1. **Plan component structure** and data flow
2. **Update type definitions** if using TypeScript
3. **Add comprehensive tests** for new functionality
4. **Update documentation** including this README
5. **Consider mobile impact** for new UI elements

## Future Enhancements

### Planned Features

- **Offline Support**: Service worker for offline map tiles
- **Advanced Clustering**: Hierarchical erratic clustering visualization
- **Export Functionality**: Route export to GPX/KML formats
- **Progressive Web App**: Native app-like experience
- **Advanced Analytics**: Route optimization analytics and statistics

### Performance Improvements

- **Web Workers**: Move TSP calculation to background thread
- **Virtual Scrolling**: For large filter lists
- **Intersection Observer**: Lazy loading for map markers
- **Request Debouncing**: Optimize API call frequency
