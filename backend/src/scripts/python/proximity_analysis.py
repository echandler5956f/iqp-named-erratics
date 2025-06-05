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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to sys.path to ensure correct relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if PYTHON_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PYTHON_SCRIPTS_DIR)

from data_pipeline import load_data # Main entry point for new data pipeline
from utils import db_utils
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

# --- Helper functions for proximity calculations ---

def _calculate_hydro_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating hydro proximity for {erratic_point}")
    nearest_water_dist = float('inf')
    nearest_water_name = None
    nearest_water_type = None
    try:
        lakes_gdf = load_data('hydrosheds_lakes')
        if lakes_gdf is not None and not lakes_gdf.empty:
            lake_feature, lake_dist = geo_utils.find_nearest_feature(erratic_point, lakes_gdf)
            if lake_dist < nearest_water_dist:
                nearest_water_dist, nearest_water_name, nearest_water_type = lake_dist, lake_feature.get('Lake_name'), 'lake'
        
        rivers_gdf = load_data('hydrosheds_rivers')
        if rivers_gdf is not None and not rivers_gdf.empty:
            river_feature, river_dist = geo_utils.find_nearest_feature(erratic_point, rivers_gdf)
            if river_dist < nearest_water_dist:
                nearest_water_dist, nearest_water_name, nearest_water_type = river_dist, river_feature.get('HYR_NAME'), 'river'
        
        results["nearest_water_body_dist"] = nearest_water_dist if nearest_water_dist != float('inf') else None
        results["nearest_water_body_name"] = nearest_water_name
        results["nearest_water_body_type"] = nearest_water_type
        logger.info(f"Nearest water: {nearest_water_name} ({nearest_water_type}) at {results.get('nearest_water_body_dist')}m")
    except Exception as e:
        logger.error(f"Error processing hydro features: {e}", exc_info=True)

def _calculate_native_territory_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating native territory proximity for {erratic_point}")
    try:
        territories_gdf = load_data('native_territories')
        if territories_gdf is not None and not territories_gdf.empty:
            territory, dist = geo_utils.find_nearest_feature(erratic_point, territories_gdf)
            results["nearest_native_territory_dist"] = dist if dist != float('inf') else None
            if territory is not None: results["nearest_native_territory_name"] = territory.get('Name') # Adjust attribute if needed
            logger.info(f"Nearest native territory: {results.get('nearest_native_territory_name')} at {results.get('nearest_native_territory_dist')}m")
    except Exception as e:
        logger.error(f"Error processing native territories: {e}", exc_info=True)

def _calculate_osm_settlement_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating OSM settlement proximity for {erratic_point}")
    try:
        # Assuming 'osm_north_america' is configured in data_sources.py with params to extract settlements
        settlements_gdf = load_data('osm_north_america') 
        if settlements_gdf is not None and not settlements_gdf.empty:
            settlement, dist = geo_utils.find_nearest_feature(erratic_point, settlements_gdf)
            results["nearest_settlement_dist"] = dist if dist != float('inf') else None
            if settlement is not None:
                results["nearest_settlement_name"] = settlement.get('name')
                results["nearest_settlement_type"] = settlement.get('place') # 'place' is typical OSM key
            logger.info(f"Nearest OSM settlement: {results.get('nearest_settlement_name')} at {results.get('nearest_settlement_dist')}m")
    except Exception as e:
        logger.error(f"Error processing OSM settlements: {e}", exc_info=True)

def _calculate_natd_road_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating NATD road proximity for {erratic_point}")
    try:
        roads_gdf = load_data('natd_roads')
        if roads_gdf is not None and not roads_gdf.empty:
            road, dist = geo_utils.find_nearest_feature(erratic_point, roads_gdf)
            results["nearest_natd_road_dist"] = dist if dist != float('inf') else None
            # 'nearest_road_dist' will be populated by this, as it's our primary "modern" road source for now
            results["nearest_road_dist"] = results["nearest_natd_road_dist"] 
            if road is not None:
                results["nearest_road_name"] = road.get('FULLNAME') # Example attribute from NATD roads
                results["nearest_road_type"] = road.get('RTTYP')    # Example attribute
            logger.info(f"Nearest NATD road: {results.get('nearest_road_name')} ({results.get('nearest_road_type')}) at {results.get('nearest_natd_road_dist')}m")
    except Exception as e:
        logger.error(f"Error processing NATD roads: {e}", exc_info=True)

def _calculate_forest_trail_proximity(erratic_point: Point, results: Dict) -> None:
    logger.debug(f"Calculating forest trail proximity for {erratic_point}")
    try:
        trails_gdf = load_data('forest_trails')
        if trails_gdf is not None and not trails_gdf.empty:
            trail, dist = geo_utils.find_nearest_feature(erratic_point, trails_gdf)
            results["nearest_forest_trail_dist"] = dist if dist != float('inf') else None
            if trail is not None:
                results["nearest_forest_trail_name"] = trail.get('TRAIL_NAME') # Example attribute
            logger.info(f"Nearest forest trail: {results.get('nearest_forest_trail_name')} at {results.get('nearest_forest_trail_dist')}m")
    except Exception as e:
        logger.error(f"Error processing forest trails: {e}", exc_info=True)

def _calculate_terrain_and_context(erratic_point: Point, erratic_db_data: Dict, results: Dict) -> None:
    logger.debug(f"Calculating terrain and context for {erratic_point}")
    # Initial elevation category from DB if available
    if erratic_db_data.get('elevation') is not None:
        try:
            results['elevation_category'] = geo_utils.get_elevation_category(float(erratic_db_data['elevation']))
        except ValueError:
            logger.warning(f"Invalid elevation value in DB for erratic: {erratic_db_data.get('id')}")

    try:
        # Use refactored geo_utils.load_dem_data()
        dem_tile_path = geo_utils.load_dem_data(point=erratic_point)
        if dem_tile_path:
            point_elevation_from_dem = geo_utils.get_elevation_at_point(erratic_point, dem_tile_path)
            if point_elevation_from_dem is not None:
                results['elevation_dem'] = point_elevation_from_dem
                results['elevation_category'] = geo_utils.get_elevation_category(point_elevation_from_dem) # Override with DEM based
            
            landscape_metrics = geo_utils.calculate_landscape_metrics(erratic_point, dem_tile_path, radius_m=500)
            geomorph_context = geo_utils.determine_geomorphological_context(erratic_point, dem_tile_path)
            
            if not np.isnan(landscape_metrics.get('ruggedness_tri', np.nan)):
                results["ruggedness_tri"] = landscape_metrics['ruggedness_tri']
            if geomorph_context.get('landform') != 'unknown':
                results["terrain_landform"] = geomorph_context['landform']
            if geomorph_context.get('slope_position') != 'unknown':
                results["terrain_slope_position"] = geomorph_context['slope_position']
            logger.info(f"Terrain context from DEM: {results.get('terrain_landform')}, {results.get('terrain_slope_position')}, TRI: {results.get('ruggedness_tri')}")
        else:
            logger.warning(f"DEM data not available for detailed terrain analysis for erratic ID {erratic_db_data.get('id')}. Using estimates if available.")
    except Exception as e:
        logger.error(f"Error during terrain analysis: {e}", exc_info=True)

    # Simplified displacement and accessibility
    if 'estimated_displacement_dist' not in results and 'latitude' in erratic_db_data:
        results["estimated_displacement_dist"] = geo_utils.estimate_displacement_distance(erratic_db_data['latitude'])
    
    # Accessibility score depends on nearest_road_dist (now from natd_roads) and nearest_settlement_dist
    if 'accessibility_score' not in results:
        road_dist = results.get("nearest_road_dist") # This should be populated by _calculate_natd_road_proximity
        settlement_dist = results.get("nearest_settlement_dist")
        if road_dist is not None and settlement_dist is not None:
             results["accessibility_score"] = geo_utils.calculate_accessibility_score(road_dist, settlement_dist)
        else:
            logger.warning("Cannot calculate accessibility_score due to missing road or settlement distance.")


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