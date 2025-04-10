#!/usr/bin/env python3
"""
Geographic utility functions for spatial analysis.
Provides functions for distance calculations, terrain analysis, and other geospatial operations.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional, Union
import math

def haversine_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        point1: Tuple of (longitude, latitude) in degrees
        point2: Tuple of (longitude, latitude) in degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1 = map(math.radians, point1)
    lon2, lat2 = map(math.radians, point2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

def nearest_feature_distance(point: Point, features_gdf: gpd.GeoDataFrame) -> Dict[str, Union[int, float]]:
    """
    Find the nearest feature to a point and calculate the distance.
    
    Args:
        point: Shapely Point geometry
        features_gdf: GeoDataFrame of features
        
    Returns:
        Dictionary with 'id' of nearest feature and 'distance' in meters
    """
    if features_gdf.empty:
        return {'id': None, 'distance': None}
    
    # Ensure CRS is the same
    if features_gdf.crs and point.has_z:
        # For 3D points, compute 2D distance
        p2d = Point(point.x, point.y)
        distances = features_gdf.geometry.apply(lambda g: p2d.distance(g))
    else:
        distances = features_gdf.geometry.apply(lambda g: point.distance(g))
    
    # Find index of minimum distance
    min_idx = distances.idxmin()
    
    return {
        'id': features_gdf.iloc[min_idx].get('id', min_idx),
        'distance': distances.iloc[min_idx]
    }

def categorize_elevation(elevation: float) -> str:
    """
    Categorize elevation into descriptive categories.
    
    Args:
        elevation: Elevation in meters
        
    Returns:
        Category string (lowland, mid-elevation, highland, mountain)
    """
    if elevation is None:
        return 'unknown'
    
    if elevation < 100:
        return 'lowland'
    elif elevation < 500:
        return 'mid-elevation'
    elif elevation < 1000:
        return 'highland'
    else:
        return 'mountain'

def categorize_size(size_meters: float) -> str:
    """
    Categorize erratic size into classes.
    
    Args:
        size_meters: Size in meters (diameter or longest axis)
        
    Returns:
        Size category string
    """
    if size_meters is None:
        return 'unknown'
    
    if size_meters < 1.0:
        return 'small'
    elif size_meters < 3.0:
        return 'medium'
    elif size_meters < 10.0:
        return 'large'
    else:
        return 'very_large'

def calculate_distances_to_features(
    point: Point, 
    water_bodies_gdf: Optional[gpd.GeoDataFrame] = None,
    settlements_gdf: Optional[gpd.GeoDataFrame] = None
) -> Dict[str, float]:
    """
    Calculate distances from a point to nearest water body and settlement.
    
    Args:
        point: Shapely Point geometry
        water_bodies_gdf: GeoDataFrame of water bodies (optional)
        settlements_gdf: GeoDataFrame of settlements (optional)
        
    Returns:
        Dictionary with distance calculations
    """
    result = {}
    
    # Calculate distance to nearest water body if available
    if water_bodies_gdf is not None and not water_bodies_gdf.empty:
        nearest_water = nearest_feature_distance(point, water_bodies_gdf)
        result['nearest_water_body_dist'] = nearest_water['distance']
    
    # Calculate distance to nearest settlement if available
    if settlements_gdf is not None and not settlements_gdf.empty:
        nearest_settlement = nearest_feature_distance(point, settlements_gdf)
        result['nearest_settlement_dist'] = nearest_settlement['distance']
    
    return result

def estimate_displacement_distance(
    point: Point, 
    bedrock_gdf: Optional[gpd.GeoDataFrame] = None,
    rock_type: Optional[str] = None
) -> Optional[float]:
    """
    Estimate how far an erratic might have been transported based on rock type and bedrock geology.
    This is a simplified estimation and would need real geological data for accuracy.
    
    Args:
        point: Shapely Point geometry of erratic location
        bedrock_gdf: GeoDataFrame of bedrock geology
        rock_type: Rock type of the erratic
        
    Returns:
        Estimated displacement distance in kilometers, or None if can't be estimated
    """
    # This is a placeholder - actual implementation would require detailed geological data
    # and more sophisticated modeling
    if bedrock_gdf is None or bedrock_gdf.empty or not rock_type:
        return None
    
    # Find the closest bedrock of the same type
    matching_bedrock = bedrock_gdf[bedrock_gdf['rock_type'] == rock_type]
    
    if matching_bedrock.empty:
        return None
    
    nearest = nearest_feature_distance(point, matching_bedrock)
    return nearest['distance']