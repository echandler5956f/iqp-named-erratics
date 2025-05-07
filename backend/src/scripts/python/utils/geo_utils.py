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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Earth radius in meters
EARTH_RADIUS = 6371000

class Point:
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
        return f"Point({self.longitude}, {self.latitude})"
    
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

def load_dem_data() -> Optional[Dict]:
    """
    Load Digital Elevation Model data - placeholder for full implementation.
    
    Returns:
        Dictionary with DEM data or None if failed
    """
    # In a real implementation, this would load DEM data from a raster file or service
    # For now, return None to indicate no data available
    return None

def find_nearest_feature(point: Point, features_gdf: gpd.GeoDataFrame) -> Tuple[Optional[Dict], float]:
    """
    Find the nearest feature to a point.
    
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
            # Check if the identifier likely corresponds to the GeoDataFrame index
            if isinstance(feature_identifier, int) and feature_identifier in features_gdf.index:
                 feature_data = features_gdf.loc[feature_identifier].to_dict()
            # Otherwise, maybe it corresponds to an 'id' column or similar
            elif 'id' in features_gdf.columns and feature_identifier in features_gdf['id'].values:
                 feature_data = features_gdf[features_gdf['id'] == feature_identifier].iloc[0].to_dict()
            else:
                 # Fallback if ID doesn't match index or 'id' column
                 logger.warning(f"Could not reliably map feature_id {feature_identifier} back to GeoDataFrame.")
                 # Try using the index from the original loop if distance list order is preserved
                 # This relies on the implementation detail of calculate_distances_to_features
                 # which is less robust. Only use as a last resort if needed.
                 pass # Or implement a safer fallback if possible

        except Exception as e:
             logger.error(f"Error retrieving feature data for identifier {feature_identifier}: {e}")

    # Use standard dict to avoid serialization issues with geometry
    if feature_data and 'geometry' in feature_data:
        del feature_data['geometry']
    
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
                safe_attrs = [attr for attr in attrs_to_select if attr != geom_col and attr != feature_id_col]
                select_cols.extend(safe_attrs)
            
            # Ensure unique columns
            select_cols = list(dict.fromkeys(select_cols)) 
            
            # Quote column names correctly for the query
            select_clause = ", ".join([f'"{c}"' for c in select_cols]) 

            # Construct the PostGIS query using placeholders
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
            
            # Execute the query with parameters
            cursor.execute(query, (lon, lat, lon, lat))
            nearest = cursor.fetchone()
            
            if nearest:
                distance = nearest.pop('distance_meters', float('inf'))
                return dict(nearest), float(distance)
            else:
                logger.info(f"No features found in table '{table_name}'.")
                return None, float('inf')

    except (Exception, psycopg2.Error) as e:
        logger.error(f"Error querying nearest feature in '{table_name}': {e}")
        try:
            if conn and not conn.closed: # Check if connection is valid before rollback
                 conn.rollback()
        except Exception as rb_err:
            logger.error(f"Error during rollback: {rb_err}")
        return None, float('inf')

def calculate_watershed(point: Point, dem_gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate watershed for a point based on DEM data.
    This is a placeholder for a more complex implementation.
    
    Args:
        point: Point to calculate watershed for
        dem_gdf: DEM data as GeoDataFrame
        
    Returns:
        GeoDataFrame with watershed polygon or None if not possible
    """
    # This is a complex calculation that would require specialized libraries
    # In a full implementation, you might use tools like Whitebox, GRASS GIS, or custom algorithms
    logger.warning("Watershed calculation not fully implemented. Requires specialized GIS processing.")
    return None

def calculate_landscape_metrics(point: Point, radius_m: float = 1000) -> Dict[str, float]:
    """
    Calculate various landscape metrics within a radius of the point.
    This is a placeholder for a more complex implementation.
    
    Args:
        point: Center point for landscape analysis
        radius_m: Radius in meters around point
        
    Returns:
        Dictionary with landscape metrics
    """
    # This would calculate various landscape metrics using real data
    # For now, return placeholder values
    return {
        'ruggedness_index': 35.2,  # Higher values = more rugged
        'elevation_variability': 120.5,  # Standard deviation of elevation in the area
        'slope_mean': 12.8,  # Average slope in degrees
        'aspect_mean': 215.6,  # Average aspect in degrees (south-southwest)
        'curvature_mean': 0.05,  # Average curvature (slightly convex)
        'forest_cover_percent': 65.3,  # Percent of forest cover
        'water_cover_percent': 8.2  # Percent of water cover
    }

def determine_geomorphological_context(point: Point) -> Dict[str, str]:
    """
    Determine the geomorphological context of a point.
    This is a placeholder for a more complex implementation.
    
    Args:
        point: Point to analyze
        
    Returns:
        Dictionary with geomorphological context information
    """
    # This would analyze terrain, slope, etc. to determine landform type
    # For now, return placeholder values
    return {
        'landform': 'hillside',
        'slope_position': 'mid-slope',
        'curvature_type': 'slightly concave',
        'hydrological_context': 'distant from streams',
        'geomorphological_unit': 'glacial till plain'
    }

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