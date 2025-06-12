#!/usr/bin/env python3
"""
Proximity Analysis Script for Glacial Erratics

This script calculates distances from a specified erratic to various geographic features
with a focus on North American native and colonial historical context.
It uses real-world GIS data sources via the data_pipeline.
"""

import sys
import os
import json
import argparse
import math
from typing import Dict, List, Optional, Union, Tuple, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from math import radians

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to sys.path to ensure correct relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if PYTHON_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PYTHON_SCRIPTS_DIR)

# Ensure data_sources is imported to register sources before geo_utils attempts to access them
import data_pipeline.data_sources  # Explicitly import to trigger registration (same pattern as geo_utils)
from data_pipeline import load_data # Main entry point for new data pipeline
from utils import db_utils
from utils.db_utils import update_erratic_elevation
from utils import file_utils
from utils import geo_utils # Uses refactored load_dem_data
from utils.geo_utils import Point # Ensure Point class is available

# North American specific context (can be refined or made configurable)
NORTH_AMERICA_BOUNDS = {
    "xmin": -170.0,
    "ymin": 15.0,
    "xmax": -50.0,
    "ymax": 75.0
}

# --- Session-Level Caching with R-tree Spatial Indices ---

class ProximityAnalyzer:
    """
    Session-level analyzer that loads datasets once and uses spatial indices
    for efficient proximity queries. Replaces the memory-intensive _adaptive_load approach.
    """
    
    def __init__(self):
        """Load all datasets once at session start"""
        self.datasets = {}
        self._load_session_datasets()
    
    def _load_session_datasets(self):
        """Load all required datasets into memory with spatial indices"""
        logger.info("Loading datasets for session-level caching...")
        
        # Track memory usage
        initial_memory = None
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        except ImportError:
            logger.debug("psutil not available, skipping memory tracking")
        except Exception as e:
            logger.debug(f"Memory tracking failed: {e}")
        
        dataset_configs = [
            ('hydrosheds_lakes', ['Lake_name']),
            ('hydrosheds_rivers', []),  # No name field
            ('native_territories', ['Name']),
            ('osm_north_america', ['name', 'place']),
            ('natd_roads', ['ROADNAME', 'CLASS']),
            ('forest_trails', ['TRAIL_NAME']),
        ]
        
        for source_name, keep_cols in dataset_configs:
            try:
                logger.info(f"Loading {source_name}...")
                if keep_cols:
                    gdf = load_data(source_name, keep_cols=keep_cols)
                else:
                    gdf = load_data(source_name)
                
                if gdf is not None and not gdf.empty:
                    # Ensure consistent CRS - convert everything to EPSG:4326 (WGS84)
                    if gdf.crs is None:
                        logger.warning(f"{source_name} has no CRS, assuming EPSG:4326")
                        gdf.set_crs(epsg=4326, inplace=True)
                    elif gdf.crs != "EPSG:4326":
                        logger.info(f"Reprojecting {source_name} from {gdf.crs} to EPSG:4326")
                        gdf = gdf.to_crs(epsg=4326)
                    
                    # Ensure spatial index is built
                    _ = gdf.sindex  # Access triggers index construction
                    self.datasets[source_name] = gdf
                    logger.info(f"Loaded {len(gdf)} features for {source_name}")
                else:
                    logger.warning(f"No data loaded for {source_name}")
                    self.datasets[source_name] = None
                    
            except Exception as e:
                logger.error(f"Failed to load {source_name}: {e}")
                self.datasets[source_name] = None
        
        total_features = sum(len(gdf) if gdf is not None else 0 for gdf in self.datasets.values())
        
        # Log final memory if tracking was successful
        if initial_memory is not None:
            try:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Session dataset loading complete: {total_features} total features, {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
            except Exception as e:
                logger.debug(f"Final memory tracking failed: {e}")
                logger.info(f"Session dataset loading complete: {total_features} total features")
        else:
            logger.info(f"Session dataset loading complete: {total_features} total features")
    
    def find_nearby_features(self, point: Point, dataset_name: str, 
                           initial_km: float = 25.0, max_km: float = 500.0) -> Optional[gpd.GeoDataFrame]:
        """
        Find features near a point using spatial index, expanding search radius as needed.
        
        Args:
            point: Point to search around
            dataset_name: Name of dataset to search
            initial_km: Initial search radius in kilometers
            max_km: Maximum search radius in kilometers
            
        Returns:
            GeoDataFrame with nearby features, or None if none found
        """
        gdf = self.datasets.get(dataset_name)
        if gdf is None or gdf.empty:
            logger.debug(f"Dataset {dataset_name} not available or empty")
            return None
        
        radius_km = initial_km
        while radius_km <= max_km:
            # Convert km to degrees (rough approximation)
            delta_deg = radius_km / 111.32
            
            # Create bounding box
            minx = point.longitude - delta_deg
            miny = point.latitude - delta_deg  
            maxx = point.longitude + delta_deg
            maxy = point.latitude + delta_deg
            
            try:
                # Use spatial index for fast lookup
                if hasattr(gdf, 'sindex') and len(gdf) > 10:
                    # Get candidate indices from spatial index
                    possible_matches_idx = list(gdf.sindex.intersection((minx, miny, maxx, maxy)))
                    
                    if possible_matches_idx:
                        # Get candidate features
                        candidates = gdf.iloc[possible_matches_idx]
                        
                        # Filter out null geometries
                        valid_candidates = candidates[candidates.geometry.notna()]
                        
                        if not valid_candidates.empty:
                            logger.debug(f"Found {len(valid_candidates)} features for {dataset_name} within {radius_km}km")
                            return valid_candidates
                
                # Fallback for small datasets or no spatial index
                elif not gdf.empty:
                    from shapely.geometry import box
                    bbox_geom = box(minx, miny, maxx, maxy)
                    intersecting = gdf[gdf.geometry.notna() & gdf.intersects(bbox_geom)]
                    
                    if not intersecting.empty:
                        logger.debug(f"Found {len(intersecting)} features for {dataset_name} within {radius_km}km (fallback)")
                        return intersecting
                        
            except Exception as e:
                logger.warning(f"Error querying {dataset_name} at {radius_km}km: {e}")
            
            # Expand search radius
            radius_km *= 2
        
        logger.debug(f"No features found for {dataset_name} within {max_km}km")
        return None

# Global analyzer instance (singleton pattern)
_global_analyzer = None

def get_proximity_analyzer() -> ProximityAnalyzer:
    """Get or create the global proximity analyzer"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = ProximityAnalyzer()
    return _global_analyzer

# --- Helper functions for proximity calculations ---

def _process_single_feature_proximity(
    erratic_point: Point,
    analyzer: ProximityAnalyzer,
    results_dict: Dict,
    dataset_name: str,
    primary_result_key_dist: str,
    primary_result_key_name: Optional[str] = None,
    primary_result_key_type: Optional[str] = None,
    name_attribute_field: Optional[str] = None,
    type_attribute_field: Optional[str] = None,
    initial_search_km: float = 25.0,
    max_search_km: float = 500.0, # Max search for general features
    feature_description_log: str = "feature",
    use_analyzer_spatial_filter: bool = True
) -> None:
    """
    Helper to find nearest feature, log, and update results dictionary.
    """
    distance = float('inf')
    name = None
    feature_type = None
    features_gdf = None

    try:
        if use_analyzer_spatial_filter:
            features_gdf = analyzer.find_nearby_features(erratic_point, dataset_name, initial_km=initial_search_km, max_km=max_search_km)
        else:
            # Directly get the GDF if spatial filtering is to be bypassed for small/preloaded datasets
            features_gdf = analyzer.datasets.get(dataset_name)

        if features_gdf is not None and not features_gdf.empty:
            feature_data, dist_val = geo_utils.find_nearest_feature(erratic_point, features_gdf)
            
            if dist_val < float('inf') and feature_data: # Ensure a feature was found
                distance = dist_val
                if name_attribute_field:
                    name = feature_data.get(name_attribute_field)
                if type_attribute_field:
                    feature_type = feature_data.get(type_attribute_field)
        
        results_dict[primary_result_key_dist] = distance if distance != float('inf') else None
        if primary_result_key_name:
            results_dict[primary_result_key_name] = name
        if primary_result_key_type:
            results_dict[primary_result_key_type] = feature_type

        # Logging
        if distance != float('inf'):
            name_display = name if name is not None else "[unnamed]"
            type_display = f" ({feature_type})" if feature_type is not None else ""
            logger.info(f"Nearest {feature_description_log}: {name_display}{type_display} at {distance:.1f}m")
        else:
            logger.info(f"No {feature_description_log} found within search radius for {dataset_name}")

    except Exception as e:
        logger.error(f"Error processing {feature_description_log} for {dataset_name}: {e}", exc_info=True)
        results_dict[primary_result_key_dist] = None
        if primary_result_key_name:
            results_dict[primary_result_key_name] = None
        if primary_result_key_type:
            results_dict[primary_result_key_type] = None


def _calculate_hydro_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating hydro proximity for {erratic_point}")
    analyzer = get_proximity_analyzer()
    
    lake_results: Dict[str, Any] = {}
    river_results: Dict[str, Any] = {}

    _process_single_feature_proximity(
        erratic_point=erratic_point,
        analyzer=analyzer,
        results_dict=lake_results,
        dataset_name='hydrosheds_lakes',
        primary_result_key_dist='dist',
        primary_result_key_name='name',
        name_attribute_field='Lake_name',
        feature_description_log="lake"
    )

    _process_single_feature_proximity(
        erratic_point=erratic_point,
        analyzer=analyzer,
        results_dict=river_results,
        dataset_name='hydrosheds_rivers',
        primary_result_key_dist='dist',
        # Rivers in HydroRIVERS don't have a consistent name field that's useful here
        primary_result_key_name=None, 
        name_attribute_field=None, 
        feature_description_log="river"
    )

    lake_dist = lake_results.get('dist') if lake_results.get('dist') is not None else float('inf')
    lake_name = lake_results.get('name')
    
    river_dist = river_results.get('dist') if river_results.get('dist') is not None else float('inf')
    # river_name is effectively None

    if lake_dist < river_dist:
        results["nearest_water_body_dist"] = lake_dist
        results["nearest_water_body_name"] = lake_name
        results["nearest_water_body_type"] = 'lake'
    elif river_dist < float('inf'):
        results["nearest_water_body_dist"] = river_dist
        results["nearest_water_body_name"] = None # River name is None
        results["nearest_water_body_type"] = 'river'
    else:
        results["nearest_water_body_dist"] = None
        results["nearest_water_body_name"] = None
        results["nearest_water_body_type"] = None
            
    # Combined logging for nearest water (already done by _process_single_feature_proximity individually for lake/river if found)
    # This specific combined log might be redundant or could be enhanced
    water_name_log = results.get('nearest_water_body_name')
    water_type_log = results.get('nearest_water_body_type')
    water_dist_log = results.get('nearest_water_body_dist')
    
    if water_dist_log is not None:
        name_display_log = water_name_log if water_name_log is not None else "[unnamed]"
        logger.info(f"Overall nearest water body: {name_display_log} ({water_type_log}) at {water_dist_log:.1f}m")
    else:
        logger.info("No water bodies (lakes or rivers) found within search radius for combined hydro proximity.")


def _calculate_native_territory_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating native territory proximity for {erratic_point}")
    analyzer = get_proximity_analyzer()
    _process_single_feature_proximity(
        erratic_point=erratic_point,
        analyzer=analyzer,
        results_dict=results,
        dataset_name='native_territories',
        primary_result_key_dist='nearest_native_territory_dist',
        primary_result_key_name='nearest_native_territory_name',
        name_attribute_field='Name', # Capital N as per original
        # initial_search_km and max_search_km are effectively ignored when use_analyzer_spatial_filter is False,
        # as the full GDF from analyzer.datasets is used directly.
        # They are kept here for signature consistency if the flag were to be toggled.
        initial_search_km=25.0, 
        max_search_km=500.0,   
        feature_description_log="native territory",
        use_analyzer_spatial_filter=False # Key change for performance
    )

def _calculate_osm_settlement_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating OSM settlement proximity for {erratic_point}")
    analyzer = get_proximity_analyzer()
    _process_single_feature_proximity(
        erratic_point=erratic_point,
        analyzer=analyzer,
        results_dict=results,
        dataset_name='osm_north_america', # This is the pre-filtered OSM data source
        primary_result_key_dist='nearest_settlement_dist',
        primary_result_key_name='nearest_settlement_name',
        primary_result_key_type='nearest_settlement_type',
        name_attribute_field='name',
        type_attribute_field='place', # 'place' is typical OSM key
        initial_search_km=50, # As per original
        max_search_km=200,    # As per original
        feature_description_log="OSM settlement"
    )

def _calculate_natd_road_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating NATD road proximity for {erratic_point}")
    analyzer = get_proximity_analyzer()
    
    temp_road_results: Dict[str, Any] = {}
    _process_single_feature_proximity(
        erratic_point=erratic_point,
        analyzer=analyzer,
        results_dict=temp_road_results,
        dataset_name='natd_roads',
        primary_result_key_dist='dist',
        primary_result_key_name='name',
        primary_result_key_type='type',
        name_attribute_field='ROADNAME', # NATD uses ROADNAME
        type_attribute_field='CLASS',    # NATD uses CLASS
        initial_search_km=25, # As per original
        feature_description_log="NATD road"
    )
    
    results["nearest_natd_road_dist"] = temp_road_results.get('dist')
    # 'nearest_road_dist' will be populated by this, as it's our primary "modern" road source for now
    results["nearest_road_dist"] = temp_road_results.get('dist') 
    results["nearest_road_name"] = temp_road_results.get('name')
    results["nearest_road_type"] = temp_road_results.get('type')
    
    # Original code had specific logging structure after this call, which is now handled by _process_single_feature_proximity if a road is found.
    # If no road is found, the generic "No NATD road found..." message from the helper will be logged.

def _calculate_forest_trail_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating forest trail proximity for {erratic_point}")
    analyzer = get_proximity_analyzer()
    _process_single_feature_proximity(
        erratic_point=erratic_point,
        analyzer=analyzer,
        results_dict=results,
        dataset_name='forest_trails',
        primary_result_key_dist='nearest_forest_trail_dist',
        primary_result_key_name='nearest_forest_trail_name',
        name_attribute_field='TRAIL_NAME', # Correct field name
        initial_search_km=25, # As per original
        feature_description_log="forest trail"
    )

def _calculate_terrain_and_context(erratic_point: Point, erratic_db_data: Dict, results: Dict) -> None:
    erratic_id_for_log = erratic_db_data.get('id', 'UnknownID') # For logging
    logger.debug(f"Calculating terrain and context for erratic ID {erratic_id_for_log} at {erratic_point}")
    
    # Initial elevation category from DB if available
    db_elevation_str = erratic_db_data.get('elevation')
    if db_elevation_str is not None:
        try:
            db_elevation = float(db_elevation_str)
            results['elevation_category'] = geo_utils.get_elevation_category(db_elevation)
        except (ValueError, TypeError):
            logger.warning(f"Invalid elevation value '{db_elevation_str}' in DB for erratic ID: {erratic_id_for_log}")

    # DEM-dependent calculations
    try:
        dem_tile_path = geo_utils.load_dem_data(point=erratic_point)
        if dem_tile_path:
            logger.debug(f"Using DEM tile {dem_tile_path} for erratic ID {erratic_id_for_log}")
            point_elevation_from_dem = geo_utils.get_elevation_at_point(erratic_point, dem_tile_path)
            
            if point_elevation_from_dem is not None:
                results['elevation_dem'] = point_elevation_from_dem
                results['elevation_category'] = geo_utils.get_elevation_category(point_elevation_from_dem) # Override with DEM based
                # Update the main Erratics table with the accurate elevation from the DEM
                update_erratic_elevation(erratic_id_for_log, point_elevation_from_dem)
            else:
                logger.debug(f"Could not retrieve DEM elevation for erratic ID {erratic_id_for_log}")

            landscape_metrics = geo_utils.calculate_landscape_metrics(erratic_point, dem_tile_path, radius_m=500)
            geomorph_context = geo_utils.determine_geomorphological_context(erratic_point, dem_tile_path)
            
            ruggedness_tri = landscape_metrics.get('ruggedness_tri')
            if ruggedness_tri is not None and not np.isnan(ruggedness_tri):
                results["ruggedness_tri"] = float(ruggedness_tri) # Ensure float

            landform = geomorph_context.get('landform')
            if landform and landform != 'unknown': # Check for None/empty and 'unknown'
                results["terrain_landform"] = landform
            
            slope_position = geomorph_context.get('slope_position')
            if slope_position and slope_position != 'unknown': # Check for None/empty and 'unknown'
                results["terrain_slope_position"] = slope_position
            
            log_tri = results.get('ruggedness_tri', 'N/A')
            log_landform = results.get('terrain_landform', 'N/A')
            log_slope_pos = results.get('terrain_slope_position', 'N/A')
            logger.info(f"Terrain context for erratic ID {erratic_id_for_log}: Landform='{log_landform}', SlopePos='{log_slope_pos}', TRI={log_tri}")
        else:
            logger.warning(f"DEM data not available for detailed terrain analysis for erratic ID {erratic_id_for_log}. Using DB elevation if available.")
    except Exception as e:
        logger.error(f"Error during DEM-based terrain analysis for erratic ID {erratic_id_for_log}: {e}", exc_info=True)

    # Simplified displacement (independent of DEM)
    if 'estimated_displacement_dist' not in results: # Check if already processed
        latitude_str = erratic_db_data.get('latitude')
        if latitude_str is not None:
            try:
                latitude_val = float(latitude_str)
                results["estimated_displacement_dist"] = geo_utils.estimate_displacement_distance(latitude_val)
            except (ValueError, TypeError):
                logger.warning(f"Invalid latitude value '{latitude_str}' in DB for erratic ID {erratic_id_for_log} for displacement calculation.")
        else:
            logger.debug(f"Missing latitude for erratic ID {erratic_id_for_log}, cannot estimate displacement.")
            
    # Accessibility score (depends on proximity results, independent of DEM processing here)
    if 'accessibility_score' not in results: # Check if already processed
        road_dist_val = results.get("nearest_road_dist")
        settlement_dist_val = results.get("nearest_settlement_dist")
        
        if road_dist_val is not None and settlement_dist_val is not None:
            try:
                # Ensure these are floats before passing to calculation
                road_dist_float = float(road_dist_val)
                settlement_dist_float = float(settlement_dist_val)
                results["accessibility_score"] = geo_utils.calculate_accessibility_score(road_dist_float, settlement_dist_float)
            except (ValueError, TypeError):
                logger.warning(f"Invalid road or settlement distance values for accessibility score calculation for erratic ID {erratic_id_for_log}. Road: '{road_dist_val}', Settlement: '{settlement_dist_val}'.")
        else:
            missing_data_for_accessibility = []
            if road_dist_val is None: missing_data_for_accessibility.append("road distance")
            if settlement_dist_val is None: missing_data_for_accessibility.append("settlement distance")
            if missing_data_for_accessibility: # Only log if something is actually missing
                 logger.warning(f"Cannot calculate accessibility_score for erratic ID {erratic_id_for_log} due to missing: {', '.join(missing_data_for_accessibility)}.")


# --- Main Proximity Calculation Function ---

def calculate_proximity(erratic_id: int) -> Dict:
    logger.info(f"Starting proximity analysis for erratic ID: {erratic_id}")
    erratic_db_data = db_utils.load_erratic_details_by_id(erratic_id)
    if not erratic_db_data:
        logger.error(f"Erratic with ID {erratic_id} not found by db_utils.")
        return {"error": f"Erratic with ID {erratic_id} not found"}
    
    longitude = erratic_db_data.get('longitude')
    latitude = erratic_db_data.get('latitude')
    if longitude is None or latitude is None:
        logger.error(f"Missing longitude/latitude for erratic ID {erratic_id}.")
        return {"error": "Missing location data for erratic"}
    
    try:
        erratic_point = Point(float(longitude), float(latitude))
        logger.info(f"Analyzing erratic at Lon: {longitude}, Lat: {latitude}")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid location data for erratic ID {erratic_id}: {e}", exc_info=True)
        return {"error": "Invalid location data for erratic"}
    
    results_payload = {
        "in_north_america": (NORTH_AMERICA_BOUNDS["xmin"] <= longitude <= NORTH_AMERICA_BOUNDS["xmax"] and
                             NORTH_AMERICA_BOUNDS["ymin"] <= latitude <= NORTH_AMERICA_BOUNDS["ymax"])
    }
    if not results_payload["in_north_america"]:
        logger.warning(f"Erratic ID {erratic_id} appears outside North America. Proximity results might be limited or N/A.")

    # Call helper functions for each proximity type
    _calculate_hydro_proximity(erratic_point, results_payload)
    _calculate_native_territory_proximity(erratic_point, results_payload)
    _calculate_osm_settlement_proximity(erratic_point, results_payload)
    _calculate_natd_road_proximity(erratic_point, results_payload) # This will also set 'nearest_road_dist'
    _calculate_forest_trail_proximity(erratic_point, results_payload)
    _calculate_terrain_and_context(erratic_point, erratic_db_data, results_payload)
    
    # Clean up results: Convert any remaining float('inf') to None for JSON serialization
    for key, value in results_payload.items():
        if value == float('inf'):
            results_payload[key] = None

    final_output = {
        "erratic_id": erratic_id,
        "erratic_name": erratic_db_data.get('name', 'Unknown'),
        "location": {"longitude": longitude, "latitude": latitude},
        "proximity_analysis": results_payload
    }
    logger.info(f"Proximity analysis completed for erratic ID {erratic_id}")
    return final_output

def main():
    parser = argparse.ArgumentParser(description='Calculate proximity for a glacial erratic using data_pipeline.')
    parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze')
    # Removed --features argument as we now process a fixed set of relevant features
    parser.add_argument('--update-db', action='store_true', help='Update database with results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG) # Ensure this module's logger is also DEBUG
    
    analysis_results = calculate_proximity(args.erratic_id)
    
    if 'error' in analysis_results:
        logger.error(f"Analysis failed: {analysis_results['error']}")
        # Ensure JSON output for Node.js service compatibility on error
        print(json.dumps({"error": analysis_results['error'], "erratic_id": args.erratic_id}))
        sys.exit(1)
        return
    
    # Prepare payload for DB update - this is the 'proximity_analysis' sub-dictionary
    db_update_payload = analysis_results.get('proximity_analysis', {})
    
    if args.update_db:
        logger.info(f"Updating database for erratic {args.erratic_id}...")
        # Ensure only relevant fields for ErraticAnalyses are passed to db_utils
        # db_utils.update_erratic_analysis_results handles filtering based on its allowed_fields
        success = db_utils.update_erratic_analysis_results(args.erratic_id, db_update_payload)
        analysis_results['database_updated'] = success # Add status to final output
        logger.info(f"Database update for {args.erratic_id} {'succeeded' if success else 'failed'}")
    
    if args.output:
        file_utils.json_to_file(analysis_results, args.output)
        logger.info(f"Results saved to {args.output}")
    
    # Print complete results to stdout for Node.js service
    print(json.dumps(analysis_results, indent=2))
    logger.info("Proximity_analysis.py script finished.")
    # Exit cleanly for CLI and tests
    sys.exit(None)

if __name__ == "__main__":
    # sys.path manipulation is done at the top of the file
    main() 