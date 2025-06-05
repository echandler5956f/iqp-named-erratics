#!/usr/bin/env python3
"""
Geographic utility functions for the erratics spatial analysis pipeline.
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import nearest_points
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import psycopg2 # Added for direct DB query
from psycopg2.extras import RealDictCursor # To get results as dicts
import os # For path operations
import sys # Added for sys.path manipulation

# Attempt rasterio import once and flag availability for tests
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Add the 'python' directory (which contains 'utils' and 'data_pipeline') to sys.path
# to allow importing sibling packages like data_pipeline.
script_dir = os.path.dirname(os.path.abspath(__file__)) # .../utils
python_dir = os.path.dirname(script_dir) # .../python
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

# Attempt to import pipeline components
PIPELINE_AVAILABLE = False
try:
    # Import the global registry directly
    from data_pipeline.registry import REGISTRY as GlobalDataRegistry
    # Ensure data_sources are loaded once to populate the registry
    import data_pipeline.data_sources
    from data_pipeline.sources import DataSource
    PIPELINE_AVAILABLE = True
    # Configure logging here after basic imports are successful
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported DataPipeline global REGISTRY. Pipeline available.")
except ImportError as e:
    # Configure logging even on import error to log the failure
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import DataPipeline REGISTRY: {e}. DEM loading will fail.")


# Earth radius in meters
EARTH_RADIUS = 6371000

class Point: # This is the GeoPoint class used by the pipeline now if imported successfully
    """
    Simple class to represent a geographic point with lon/lat coordinates.
    """
    def __init__(self, longitude: float, latitude: float):
        self.longitude = float(longitude)
        self.latitude = float(latitude)
    
    def to_shapely(self) -> ShapelyPoint:
        """Convert to Shapely Point object"""
        return ShapelyPoint(self.longitude, self.latitude)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lon, lat) tuple"""
        return (self.longitude, self.latitude)
    
    def __str__(self) -> str:
        return f"Point(lon={self.longitude}, lat={self.latitude})"
    
    def __repr__(self) -> str:
        return self.__str__()

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth's surface using the Haversine formula.
    
    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees
        
    Returns:
        Distance between points in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Return distance in meters
    return EARTH_RADIUS * c

def calculate_distances_to_features(point: Point, features_gdf: gpd.GeoDataFrame) -> List[Dict]:
    """
    Calculate distances from a point to all features in a GeoDataFrame.
    
    Args:
        point: Point to calculate distances from
        features_gdf: GeoDataFrame with features to calculate distances to
        
    Returns:
        List of dictionaries with feature information and distances
    """
    if features_gdf.empty:
        return []
    
    # Ensure correct CRS (EPSG:4326 - WGS84)
    if features_gdf.crs is None:
        features_gdf.set_crs(epsg=4326, inplace=True)
    elif features_gdf.crs != "EPSG:4326":
        features_gdf = features_gdf.to_crs(epsg=4326)
    
    # Convert point to Shapely Point
    shapely_point = ShapelyPoint(point.longitude, point.latitude)
    
    # Calculate distances using Haversine formula
    results = []
    for idx, feature in features_gdf.iterrows():
        try:
            # For point features
            if feature.geometry.geom_type == 'Point':
                distance = haversine_distance(
                    point.latitude, point.longitude,
                    feature.geometry.y, feature.geometry.x
                )
            
            # For line or polygon features, find nearest point
            else:
                nearest_geom = nearest_points(shapely_point, feature.geometry)[1]
                distance = haversine_distance(
                    point.latitude, point.longitude,
                    nearest_geom.y, nearest_geom.x
                )
            
            # Create result entry
            result = {
                'feature_id': getattr(feature, 'id', idx),
                'distance': distance,  # in meters
                'geometry_type': feature.geometry.geom_type
            }
            
            # Add common attributes if they exist
            for attr in ['name', 'type', 'class', 'category']:
                if hasattr(feature, attr) and getattr(feature, attr) is not None:
                    result[attr] = getattr(feature, attr)
            
            results.append(result)
        except Exception as e:
            logger.error(f"Error calculating distance to feature {idx}: {e}")
    
    # Sort by distance
    results.sort(key=lambda x: x['distance'])
    
    return results

def categorize_elevation(elevation: float) -> str:
    """
    Categorize elevation into standard categories.
    
    Args:
        elevation: Elevation in meters
        
    Returns:
        Elevation category as string
    """
    if elevation < 0:
        return "below_sea_level"
    elif elevation < 200:
        return "lowland"
    elif elevation < 500:
        return "upland"
    elif elevation < 1000:
        return "hill"
    elif elevation < 2000:
        return "low_mountain"
    elif elevation < 3000:
        return "mid_mountain"
    elif elevation < 5000:
        return "high_mountain"
    else:
        return "extreme_elevation"

def get_elevation_category(elevation: float) -> str:
    """
    Get the elevation category for a given elevation.
    
    Args:
        elevation: Elevation in meters
        
    Returns:
        Elevation category
    """
    return categorize_elevation(elevation)

def load_dem_data(point: Optional[Point] = None) -> Optional[str]:
    """
    Finds the local file path for the DEM tile that covers the given point.
    It iterates through registered tiled DEM sources (specifically 'gmted_elevation_tiled')
    and checks if the point falls within any tile's bounding box.

    Args:
        point: geo_utils.Point object (lon, lat) to find the corresponding DEM tile for.
               If None, this function cannot determine a specific tile and will return None.

    Returns:
        File path to the DEM TIF file for the specific tile,
        or None if the global registry is unavailable, point not provided, no covering tile found,
        or the 'gmted_elevation_tiled' source is not configured correctly.
    """
    if not PIPELINE_AVAILABLE or GlobalDataRegistry is None:
        logger.error("DataPipeline REGISTRY is not available. Cannot load DEM data.")
        return None
    if point is None:
        logger.warning("load_dem_data called without a specific point. Cannot determine DEM tile path.")
        return None

    try:
        target_tiled_source_name = "gmted_elevation_tiled"
        gmted_source: Optional[DataSource] = None
        for source_name, source_obj in GlobalDataRegistry.get_all_sources().items():
            if source_name == target_tiled_source_name and source_obj.is_tiled:
                gmted_source = source_obj
                break
        
        if not gmted_source:
            logger.error(f"Tiled DEM source '{target_tiled_source_name}' not found or not configured as tiled in the registry.")
            return None

        if not gmted_source.tile_centers or not gmted_source.tile_paths or not gmted_source.tile_size_degrees:
            logger.error(f"Tiled DEM source '{target_tiled_source_name}' is missing tile_centers, tile_paths, or tile_size_degrees.")
            return None
        
        if len(gmted_source.tile_paths) != len(gmted_source.tile_centers):
            logger.error(f"Mismatch in lengths of tile_paths and tile_centers for '{target_tiled_source_name}'.")
            return None

        logger.debug(f"Searching for DEM tile for point {point} in '{target_tiled_source_name}'.")
        tile_size = gmted_source.tile_size_degrees
        half_size = tile_size / 2.0

        for i, center_coord in enumerate(gmted_source.tile_centers):
            if center_coord is None:
                continue
            lon_c, lat_c = center_coord
            
            lat_min = lat_c - half_size
            lat_max = lat_c + half_size
            lon_min = lon_c - half_size
            lon_max = lon_c + half_size

            # Check if the point is within the tile bounds (inclusive min, exclusive max for standard geo queries)
            if (lat_min <= point.latitude < lat_max and
                lon_min <= point.longitude < lon_max):
                tile_path = gmted_source.tile_paths[i]
                if os.path.exists(tile_path):
                    logger.info(f"Found DEM tile for point {point}: {tile_path}")
                    return tile_path
                else:
                    logger.error(f"DEM tile path {tile_path} for point {point} does not exist on filesystem.")
                    # Continue searching in case of misconfiguration, but this is an error.
        
        logger.warning(f"No covering DEM tile found in '{target_tiled_source_name}' for point {point}.")
        return None

    except Exception as e:
        logger.error(f"Error loading DEM data for point {point}: {e}", exc_info=True)
        return None

def find_nearest_feature(point: Point, features_gdf: gpd.GeoDataFrame) -> Tuple[Optional[Dict], float]:
    """
    Find the nearest feature to a point from an in-memory GeoDataFrame.
    
    Args:
        point: Point to find the nearest feature to
        features_gdf: GeoDataFrame with features
        
    Returns:
        Tuple of (feature_data, distance)
    """
    if features_gdf.empty:
        return None, float('inf')
    
    distances = calculate_distances_to_features(point, features_gdf)
    if not distances:
        return None, float('inf')
    
    # Get nearest feature
    nearest = distances[0]
    
    # Get full feature data
    # Correct handling for index or custom ID
    feature_identifier = nearest.get('feature_id') 
    feature_data = None
    if feature_identifier is not None:
        try:
            # Priority 1: Check if 'id' column exists and feature_identifier matches a value in it
            if 'id' in features_gdf.columns and feature_identifier in features_gdf['id'].values:
                 matching_features = features_gdf[features_gdf['id'] == feature_identifier]
                 if not matching_features.empty:
                     # Assume first match if IDs are not strictly unique, though they should be.
                     feature_data = matching_features.iloc[0].to_dict()
            # Priority 2: If not found by 'id' column, check if feature_identifier corresponds to the GeoDataFrame index
            elif isinstance(feature_identifier, int) and feature_identifier in features_gdf.index:
                 feature_data = features_gdf.loc[feature_identifier].to_dict()
            else:
                 logger.warning(f"Could not reliably map feature_id {feature_identifier} back to GeoDataFrame 'id' column or index.")

        except Exception as e:
             logger.error(f"Error retrieving feature data for identifier {feature_identifier}: {e}")

    # Use standard dict to avoid serialization issues with geometry
    # Convert geometry to WKT or remove if present
    if feature_data:
        if 'geometry' in feature_data and hasattr(feature_data['geometry'], 'wkt'):
            try:
                # Store WKT representation instead of Shapely object for JSON compatibility
                feature_data['geometry_wkt'] = feature_data['geometry'].wkt 
            except Exception:
                 pass # Ignore if conversion fails
            del feature_data['geometry']
        # Convert any other non-serializable types if necessary (e.g., timestamps)
        for key, value in feature_data.items():
             if isinstance(value, pd.Timestamp):
                 feature_data[key] = value.isoformat()
             elif isinstance(value, np.generic):
                 feature_data[key] = value.item() # Convert numpy types to native Python types

    return feature_data, nearest.get('distance', float('inf'))

def find_nearest_feature_db(point: Point, conn, table_name: str, geom_col: str, 
                              feature_id_col: str = 'id', attrs_to_select: Optional[List[str]] = None) -> Tuple[Optional[Dict], float]:
    """
    Find the nearest feature to a point using a direct PostGIS query.
    Assumes the table geometry is in EPSG:4326.

    Args:
        point: Point object to find the nearest feature to.
        conn: Active psycopg2 database connection.
        table_name: Name of the database table (including schema if needed).
        geom_col: Name of the geometry column in the table.
        feature_id_col: Name of the primary key or unique ID column.
        attrs_to_select: Optional list of other attribute columns to select.

    Returns:
        Tuple of (feature_data_dict, distance_meters) or (None, float('inf')).
    """
    if not conn:
        logger.error("Database connection not provided for find_nearest_feature_db")
        return None, float('inf')

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            lon = float(point.longitude)
            lat = float(point.latitude)

            # Build SELECT clause carefully
            select_cols = [feature_id_col]
            if attrs_to_select:
                # Ensure geometry and ID columns are not duplicated if requested
                safe_attrs = [attr for attr in attrs_to_select if attr != geom_col and attr != feature_id_col]
                select_cols.extend(safe_attrs)
            
            # Ensure unique columns and quote them
            # Quote with f'"{{c}}"' if c is not already quoted, else c
            quoted_select_cols = []
            for c in list(dict.fromkeys(select_cols)):
                if not (c.startswith('"') and c.endswith('"')):
                    quoted_select_cols.append(f'"{c}"')
                else:
                    quoted_select_cols.append(c)
            select_clause = ", ".join(quoted_select_cols)

            # Use ST_MakePoint for the input point and ST_DistanceSphere for distance
            # Order by the KNN operator (<->) for efficiency with a spatial index
            # Ensure table_name and geom_col are properly quoted if they might contain special chars or be case sensitive
            # For simplicity, assuming they are standard identifiers here or already quoted if needed passed in.
            query = f""" 
                SELECT 
                    {select_clause},
                    ST_DistanceSphere(
                        "{geom_col}", 
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                    ) AS distance_meters
                FROM "{table_name}"
                ORDER BY 
                    "{geom_col}" <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                LIMIT 1;
            """
            
            # Execute the query with parameters (lon, lat passed twice for point and KNN center)
            cursor.execute(query, (lon, lat, lon, lat))
            nearest = cursor.fetchone()
            
            if nearest:
                distance = nearest.pop('distance_meters', float('inf')) # Extract distance
                return dict(nearest), float(distance) # Return feature attrs and distance
            else:
                logger.info(f"No features found in table '{table_name}'.")
                return None, float('inf')

    except (Exception, psycopg2.Error) as e:
        logger.error(f"Error querying nearest feature in '{table_name}': {e}")
        try:
            if conn and not conn.closed: # Check if connection is valid before rollback
                 conn.rollback()
                 logger.info("Database transaction rolled back due to error.")
        except Exception as rb_err:
            logger.error(f"Error during rollback: {rb_err}")
        return None, float('inf')

def get_elevation_at_point(point: Point, dem_path: str) -> Optional[float]:
    """
    Get the elevation value from a DEM raster at a specific point.

    Args:
        point: Point object with longitude and latitude.
        dem_path: Path to the DEM raster file (e.g., GeoTIFF).

    Returns:
        Elevation value in the units of the DEM, or None if error or outside bounds.
    """
    if not RASTERIO_AVAILABLE:
        logger.warning("rasterio not available, cannot get elevation.")
        return None
    if not dem_path or not os.path.exists(dem_path):
        logger.error(f"DEM file not found at {dem_path}")
        return None

    try:
        with rasterio.open(dem_path) as src:
            # Ensure point is in the same CRS as the raster
            # Assuming point is WGS84 (EPSG:4326) and DEM might be different
            point_coords = [(point.longitude, point.latitude)]
            
            # Sample the raster at the given coordinates
            # Note: rasterio.sample expects coordinates in the raster's CRS.
            # If the DEM is not EPSG:4326, we need to transform the point coords.
            # For simplicity here, we assume the DEM is EPSG:4326 or compatible.
            # A robust implementation would check src.crs and transform if needed.
            if src.crs and str(src.crs).lower() != 'epsg:4326': # Check if src.crs is not None
                logger.warning(f"DEM CRS ({src.crs}) is not EPSG:4326. Elevation sampling might be inaccurate without coordinate transformation.")
                # Add coordinate transformation logic here if necessary
            
            # Use sample method to get value(s)
            # It returns a generator, get the first value
            value_generator = src.sample(point_coords)
            try:
                elevation_value = next(value_generator)[0] # Get value from the first band
            except StopIteration:
                logger.warning(f"Point {point} seems to be outside the DEM bounds of {dem_path}.")
                return None

            # Check for nodata value
            if src.nodata is not None and elevation_value == src.nodata:
                logger.warning(f"Point {point} falls on a nodata value in the DEM {dem_path}.")
                return None
                
            return float(elevation_value)

    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio IO error reading {dem_path} at point {point}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting elevation from DEM {dem_path} at point {point}: {e}")
        return None

def calculate_terrain_ruggedness_index(window_data: np.ndarray) -> float:
    """
    Calculate Terrain Ruggedness Index (TRI) for a window of elevation data.
    TRI = Mean of the absolute differences between the center cell and its 8 neighbors.
    """
    if window_data.shape != (3, 3):
        # Need a 3x3 window for standard TRI
        # Could also calculate over larger windows, but this is typical
        logger.warning(f"TRI calculation requires a 3x3 window, got {window_data.shape}. Cannot calculate TRI.")
        return np.nan
    
    center_value = window_data[1, 1]
    if np.isnan(center_value):
        return np.nan
        
    diff_sum = 0
    valid_neighbors = 0
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue # Skip center cell
            neighbor_value = window_data[i, j]
            if not np.isnan(neighbor_value):
                diff_sum += abs(neighbor_value - center_value)
                valid_neighbors += 1
    
    if valid_neighbors == 0:
        return np.nan
        
    return diff_sum / valid_neighbors # Return mean difference

def calculate_landscape_metrics(point: Point, dem_path: str, radius_m: float = 1000) -> Dict[str, float]:
    """
    Calculate basic landscape metrics within a radius of the point using a DEM.
    Requires rasterio.
    
    Args:
        point: Center point for landscape analysis (lon, lat)
        dem_path: Path to the DEM raster file.
        radius_m: Radius in meters around the point.
        
    Returns:
        Dictionary with calculated landscape metrics (e.g., elevation stats, ruggedness).
        Returns empty dict or dict with NaNs if calculation fails.
    """
    if not RASTERIO_AVAILABLE:
        logger.warning("rasterio not available, cannot calculate landscape metrics.")
        return {}
    if not dem_path or not os.path.exists(dem_path):
        logger.error(f"DEM file not found at {dem_path} for landscape metrics.")
        return {}

    metrics = {
        'elevation_mean': np.nan,
        'elevation_stddev': np.nan,
        'ruggedness_tri': np.nan, 
        # Add more metrics like slope, aspect if needed (requires more complex calculation)
    }

    try:
        with rasterio.open(dem_path) as src:
            # Convert radius in meters to pixels
            # This is approximate if the CRS is geographic (like EPSG:4326)
            # A more accurate way uses the transform and CRS properties.
            pixel_size_x = src.res[0]
            pixel_size_y = src.res[1]
            if src.crs and src.crs.is_geographic: # Check if src.crs is not None
                # Approximation: degrees per meter varies with latitude
                m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(point.latitude)) + 1.175 * math.cos(4 * math.radians(point.latitude))
                m_per_deg_lon = 111320 * math.cos(math.radians(point.latitude))
                radius_deg_x = radius_m / m_per_deg_lon
                radius_deg_y = radius_m / m_per_deg_lat
                radius_px_x = int(radius_deg_x / abs(pixel_size_x))
                radius_px_y = int(radius_deg_y / abs(pixel_size_y))
            elif src.crs: # Projected CRS, check if src.crs is not None
                radius_px_x = int(radius_m / abs(pixel_size_x))
                radius_px_y = int(radius_m / abs(pixel_size_y))
            else: # CRS is None
                logger.warning(f"DEM {dem_path} has no CRS. Cannot reliably convert radius to pixels.")
                return metrics
                
            # Get pixel coordinates of the point
            try:
                row, col = src.index(point.longitude, point.latitude)
            except rasterio.errors.RasterioIOError: # Handles point outside bounds
                logger.warning(f"Point {point} is outside the bounds of DEM {dem_path}. Cannot calculate landscape metrics.")
                return metrics
            
            # Define the window
            # Ensure window stays within raster bounds
            win_row_off = max(0, row - radius_px_y)
            win_col_off = max(0, col - radius_px_x)
            win_height = min(src.height - win_row_off, 2 * radius_px_y + 1)
            win_width = min(src.width - win_col_off, 2 * radius_px_x + 1)
            
            window = rasterio.windows.Window(win_col_off, win_row_off, win_width, win_height)
            
            # Read data within the window
            window_data = src.read(1, window=window)
            
            # Handle nodata values
            nodata_val = src.nodata
            if nodata_val is not None:
                valid_data = window_data[window_data != nodata_val]
            else:
                valid_data = window_data.flatten() # Assume all data is valid if nodata not defined
                
            if valid_data.size == 0:
                logger.warning(f"No valid DEM data found in window around {point} in {dem_path}")
                return metrics # Return NaNs

            # Calculate basic stats
            metrics['elevation_mean'] = float(np.mean(valid_data))
            metrics['elevation_stddev'] = float(np.std(valid_data))

            # Calculate TRI (Terrain Ruggedness Index) for the center pixel (requires 3x3 window)
            # Read a 3x3 window centered on the point
            if row > 0 and row < src.height - 1 and col > 0 and col < src.width - 1:
                tri_window_data = src.read(1, window=rasterio.windows.Window(col-1, row-1, 3, 3))
                if nodata_val is not None:
                    tri_window_data = np.where(tri_window_data == nodata_val, np.nan, tri_window_data).astype(float)
                else:
                     tri_window_data = tri_window_data.astype(float)
                metrics['ruggedness_tri'] = calculate_terrain_ruggedness_index(tri_window_data)
            else:
                logger.warning(f"Point {point} is too close to DEM edge ({dem_path}) to calculate 3x3 TRI.")

            # Placeholder: More complex metrics like slope, aspect would require dedicated libraries or algorithms
            # e.g., using numpy.gradient or richdem

    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio IO error processing DEM {dem_path} for metrics: {e}")
    except Exception as e:
        logger.error(f"Error calculating landscape metrics from DEM {dem_path}: {e}")
        
    return metrics

def determine_geomorphological_context(point: Point, dem_path: str) -> Dict[str, str]:
    """
    Determine a simplified geomorphological context of a point using DEM.
    Requires rasterio.
    
    Args:
        point: Point to analyze
        dem_path: Path to the DEM raster file.
        
    Returns:
        Dictionary with simplified geomorphological context information.
    """
    if not RASTERIO_AVAILABLE:
        logger.warning("rasterio not available, cannot determine geomorph context.")
        return {}
    if not dem_path or not os.path.exists(dem_path):
        logger.error(f"DEM file not found at {dem_path} for geomorphological context.")
        return {}

    context = {
        'landform': 'unknown',
        'slope_position': 'unknown',
        # 'curvature_type': 'unknown',
        # 'hydrological_context': 'unknown' 
    }

    try:
        # Get elevation at the point
        elevation = get_elevation_at_point(point, dem_path)
        if elevation is None:
            return context # Cannot proceed without elevation
            
        # Use landscape metrics (basic version for now)
        # A small radius might be better for local context
        metrics = calculate_landscape_metrics(point, dem_path, radius_m=100) # Use a smaller radius for local context
        
        elevation_stddev = metrics.get('elevation_stddev', np.nan)
        ruggedness = metrics.get('ruggedness_tri', np.nan)

        # Simple rules based on local variation / ruggedness
        if not np.isnan(elevation_stddev):
            if elevation_stddev < 5:
                context['landform'] = 'flat_plain'
                context['slope_position'] = 'level'
            elif elevation_stddev < 20:
                context['landform'] = 'undulating_terrain'
                # Need slope/aspect for better slope position
            else:
                context['landform'] = 'hilly_or_mountainous'
        
        if not np.isnan(ruggedness):
             if ruggedness < 10:
                  # Potentially refine landform if ruggedness is low
                  if context['landform'] == 'hilly_or_mountainous': context['landform'] = 'rolling_hills'
             elif ruggedness > 50:
                  # High ruggedness indicates more complex terrain
                  if context['landform'] != 'hilly_or_mountainous': context['landform'] = 'rugged_hills'
                  
        # Placeholder: Slope position and curvature require slope/aspect calculation
        # which is more involved (e.g., using numpy.gradient or dedicated libraries)
        # Example (conceptual):
        # slope, aspect = calculate_slope_aspect(dem_window)
        # if slope < 5: context['slope_position'] = 'flat'
        # elif slope > 30: context['slope_position'] = 'steep_slope'
        # else: context['slope_position'] = 'gentle_slope'

    except Exception as e:
        logger.error(f"Error determining geomorphological context: {e}")

    return context

def direction_to_cardinal(degrees: float) -> str:
    """
    Convert degrees to cardinal direction.
    
    Args:
        degrees: Direction in degrees (0-360, 0=North)
        
    Returns:
        Cardinal direction string
    """
    # Normalize to 0-360 range
    degrees = degrees % 360
    
    # Define direction brackets
    directions = [
        'N', 'NNE', 'NE', 'ENE', 
        'E', 'ESE', 'SE', 'SSE', 
        'S', 'SSW', 'SW', 'WSW', 
        'W', 'WNW', 'NW', 'NNW'
    ]
    
    # Calculate index (each slot is 22.5 degrees)
    idx = round(degrees / 22.5) % 16
    
    return directions[idx]

def calculate_relative_position(point: Point, reference_point: Point) -> Dict[str, Any]:
    """
    Calculate relative position from a point to a reference point.
    
    Args:
        point: Source point
        reference_point: Reference point
        
    Returns:
        Dictionary with relative position information
    """
    # Calculate distance
    distance = haversine_distance(
        point.latitude, point.longitude,
        reference_point.latitude, reference_point.longitude
    )
    
    # Calculate bearing
    y = math.sin(math.radians(reference_point.longitude - point.longitude)) * math.cos(math.radians(reference_point.latitude))
    x = (
        math.cos(math.radians(point.latitude)) * math.sin(math.radians(reference_point.latitude)) -
        math.sin(math.radians(point.latitude)) * math.cos(math.radians(reference_point.latitude)) *
        math.cos(math.radians(reference_point.longitude - point.longitude))
    )
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    
    # Get cardinal direction
    cardinal = direction_to_cardinal(bearing)
    
    return {
        'distance_m': distance,
        'bearing_degrees': bearing,
        'direction': cardinal
    }

def estimate_displacement_distance(latitude: float) -> float:
    """Very rough proxy for glacial displacement distance based on latitude.
    This is just a placeholder for testing – real logic would use ice sheet models.
    """
    try:
        lat = float(latitude)
    except (ValueError, TypeError):
        return 0.0
    # Simple heuristic: colder (higher latitude) potentially longer transport
    return max(0.0, (90 - abs(lat)) * 10)  # e.g., 0-900 km

def calculate_accessibility_score(road_dist_m: float, settlement_dist_m: float) -> float:
    """Calculate an accessibility score given road and settlement distances.

    Lower distances yield higher accessibility. Score scaled 0 (inaccessible) – 10 (very accessible).
    """
    try:
        road = float(road_dist_m)
        settle = float(settlement_dist_m)
    except (ValueError, TypeError):
        return 0.0

    # Simple inverse relationship with cap
    max_dist = 10000  # 10 km as reference for minimal accessibility
    norm = max(0.0, 1 - min((road + settle) / (2 * max_dist), 1))
    return round(norm * 10, 2)

def find_nearest_water(point: Point, rivers_gdf: gpd.GeoDataFrame, lakes_gdf: gpd.GeoDataFrame) -> Tuple[float, str, str]:
    """
    Find the nearest water feature to a point from in-memory GeoDataFrames.
    
    Args:
        point: Point to find the nearest water feature to
        rivers_gdf: GeoDataFrame with rivers
        lakes_gdf: GeoDataFrame with lakes
        
    Returns:
        Tuple of (distance, water_name, water_type)
    """
    if rivers_gdf.empty and lakes_gdf.empty:
        return float('inf'), '', ''
    
    best_distance = float('inf')
    best_name = ''
    best_type = ''
    
    # Check rivers
    if not rivers_gdf.empty:
        rivers_gdf = rivers_gdf.copy()
        river_feature, river_dist = find_nearest_feature(point, rivers_gdf)
        if river_dist < best_distance:
            best_distance = river_dist
            best_name = river_feature.get('HYR_NAME', '') if river_feature else ''
            best_type = 'river'
    
    # Check lakes
    if not lakes_gdf.empty:
        lakes_gdf = lakes_gdf.copy()
        lake_feature, lake_dist = find_nearest_feature(point, lakes_gdf)
        if lake_dist < best_distance:
            best_distance = lake_dist
            best_name = lake_feature.get('Lake_name', '') if lake_feature else ''
            best_type = 'lake'
    
    return best_distance, best_name, best_type