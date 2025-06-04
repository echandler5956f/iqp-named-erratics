#!/usr/bin/env python3
"""
Proximity Analysis Script for Glacial Erratics

This script calculates distances from a specified erratic to various geographic features
with a focus on North American native and colonial historical context.
It uses real-world GIS data sources.
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
# Assumes this script is in backend/src/scripts/python/
# and project root is backend/src/scripts/
# This needs to be robust to allow utils and data_pipeline to be found.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR) # up to /python
# PROJECT_ROOT_FOR_PYTHON = os.path.dirname(PYTHON_SCRIPTS_DIR) # up to /scripts
# sys.path.insert(0, PROJECT_ROOT_FOR_PYTHON)
# A more common pattern if utils/data_pipeline are sibling packages to the current script's package:
# sys.path.append(os.path.join(SCRIPT_DIR, '..')) # Add parent (python/) to find utils & data_pipeline

# Updated imports
from data_pipeline import load_data # Main entry point for new data pipeline
from utils import db_utils # For application-specific DB operations
from utils import file_utils # For json_to_file
from utils import geo_utils # Existing geo_utils, to be refactored later

# North American specific context
NORTH_AMERICA_BOUNDS = {
    "xmin": -170.0,  # Western limit (including Alaska)
    "ymin": 15.0,    # Southern limit (including southern Mexico)
    "xmax": -50.0,   # Eastern limit (including eastern Canada)
    "ymax": 75.0     # Northern limit (including northern Canada)
}

def calculate_proximity(erratic_id: int, feature_layers: Optional[List[str]] = None) -> Dict:
    logger.info(f"Starting proximity analysis for erratic ID: {erratic_id}")
    # Load erratic data using new db_utils
    erratic = db_utils.load_erratic_details_by_id(erratic_id)
    if not erratic:
        logger.error(f"Erratic with ID {erratic_id} not found by db_utils.")
        return {"error": f"Erratic with ID {erratic_id} not found"}
    
    if feature_layers is None:
        feature_layers = [
            "lakes", "rivers", # from hydrosheds
            "native_territories", 
            "osm_settlements", # from OSM (e.g., osm_north_america_settlements)
            "nhgis_settlements", # from NHGIS (colonial)
            "osm_roads", # from OSM (e.g., osm_north_america_roads)
            "colonial_roads"
        ]
    
    try:
        longitude = erratic.get('longitude')
        latitude = erratic.get('latitude')
        if longitude is None or latitude is None:
            logger.error(f"Missing longitude/latitude for erratic ID {erratic_id}.")
            return {"error": "Missing location data for erratic"}
        erratic_point = geo_utils.Point(longitude, latitude)
        logger.info(f"Analyzing erratic at Lon: {longitude}, Lat: {latitude}")
    except Exception as e:
        logger.error(f"Invalid location data for erratic ID {erratic_id}: {e}", exc_info=True)
        return {"error": "Invalid location data for erratic"}
    
    results = {
        "in_north_america": (NORTH_AMERICA_BOUNDS["xmin"] <= longitude <= NORTH_AMERICA_BOUNDS["xmax"] and 
                             NORTH_AMERICA_BOUNDS["ymin"] <= latitude <= NORTH_AMERICA_BOUNDS["ymax"])
    }
    if not results["in_north_america"]:
        logger.warning(f"Erratic ID {erratic_id} appears outside North America. Proximity results might be limited.")

    if erratic.get('elevation') is not None:
        try:
            results['elevation_category'] = geo_utils.get_elevation_category(float(erratic['elevation']))
        except Exception: pass # Handled by DEM later or if elevation is invalid
    
    # For HydroSHEDS features, potentially use DB if that utility is refined in geo_utils
    # For now, assume file loading via data_pipeline for simplicity here
    if 'lakes' in feature_layers or 'rivers' in feature_layers:
        nearest_water_dist = float('inf')
        nearest_water_name = None
        nearest_water_type = None
        try:
            if 'lakes' in feature_layers:
                lakes_gdf = load_data('hydrosheds_lakes')
                if not lakes_gdf.empty:
                    lake_feature, lake_dist = geo_utils.find_nearest_feature(erratic_point, lakes_gdf)
                    if lake_dist < nearest_water_dist:
                        nearest_water_dist, nearest_water_name, nearest_water_type = lake_dist, lake_feature.get('Lake_name'), 'lake'
            if 'rivers' in feature_layers: # Assuming rivers are similar to lakes in source/attributes
                rivers_gdf = load_data('hydrosheds_rivers') # You'll need 'hydrosheds_rivers' in data_sources.py
                if not rivers_gdf.empty:
                    river_feature, river_dist = geo_utils.find_nearest_feature(erratic_point, rivers_gdf)
                    if river_dist < nearest_water_dist:
                        nearest_water_dist, nearest_water_name, nearest_water_type = river_dist, river_feature.get('HYR_NAME'), 'river' # Example name col
            results["nearest_water_body_dist"] = nearest_water_dist if nearest_water_dist != float('inf') else None
            results["nearest_water_body_name"] = nearest_water_name
            results["nearest_water_body_type"] = nearest_water_type
            logger.info(f"Nearest water: {nearest_water_name} ({nearest_water_type}) at {nearest_water_dist if nearest_water_dist != float('inf') else 'N/A'}m")
        except Exception as e:
            logger.error(f"Error processing hydro features for erratic ID {erratic_id}: {e}", exc_info=True)

    if "native_territories" in feature_layers:
        try:
            territories_gdf = load_data('native_territories')
            if not territories_gdf.empty:
                territory, dist = geo_utils.find_nearest_feature(erratic_point, territories_gdf)
                results["nearest_native_territory_dist"] = dist if dist != float('inf') else None
                if territory is not None: results["nearest_native_territory_name"] = territory.get('Name')
                logger.info(f"Nearest native territory: {results.get('nearest_native_territory_name')} at {results.get('nearest_native_territory_dist')}m")
        except Exception as e: logger.error(f"Error processing native territories for {erratic_id}: {e}", exc_info=True)

    if "osm_settlements" in feature_layers:
        try:
            # Using North America wide data, could be parameterized by region later if needed.
            settlements_gdf = load_data('osm_north_america_settlements') 
            if not settlements_gdf.empty:
                settlement, dist = geo_utils.find_nearest_feature(erratic_point, settlements_gdf)
                results["nearest_settlement_dist"] = dist if dist != float('inf') else None
                if settlement is not None:
                    results["nearest_settlement_name"] = settlement.get('name')
                    results["nearest_settlement_type"] = settlement.get('place_type') # Assuming PBFProcessor standardizes this
                logger.info(f"Nearest OSM settlement: {results.get('nearest_settlement_name')} at {results.get('nearest_settlement_dist')}m")
        except Exception as e: logger.error(f"Error processing OSM settlements for {erratic_id}: {e}", exc_info=True)
    
    if "nhgis_settlements" in feature_layers: # Colonial settlements
        try:
            colonial_settlements_gdf = load_data('nhgis_historical_settlements')
            if not colonial_settlements_gdf.empty:
                settlement, dist = geo_utils.find_nearest_feature(erratic_point, colonial_settlements_gdf)
                results["nearest_colonial_settlement_dist"] = dist if dist != float('inf') else None
                if settlement is not None: results["nearest_colonial_settlement_name"] = settlement.get('NHGISNAM') # Example name col
                logger.info(f"Nearest colonial settlement: {results.get('nearest_colonial_settlement_name')} at {results.get('nearest_colonial_settlement_dist')}m")
        except Exception as e: logger.error(f"Error processing NHGIS settlements for {erratic_id}: {e}", exc_info=True)

    if "osm_roads" in feature_layers: # Modern roads
        try:
            roads_gdf = load_data('osm_north_america_roads') # Ensure this source name exists and is configured for roads
            if not roads_gdf.empty:
                road, dist = geo_utils.find_nearest_feature(erratic_point, roads_gdf)
                results["nearest_road_dist"] = dist if dist != float('inf') else None
                if road is not None:
                    results["nearest_road_name"] = road.get('name')
                    results["nearest_road_type"] = road.get('highway') # Standard OSM tag for road type
                logger.info(f"Nearest OSM road: {results.get('nearest_road_name')} ({results.get('nearest_road_type')}) at {results.get('nearest_road_dist')}m")
        except Exception as e: logger.error(f"Error processing OSM roads for {erratic_id}: {e}", exc_info=True)

    if "colonial_roads" in feature_layers:
        try:
            colonial_roads_gdf = load_data('colonial_roads')
            if not colonial_roads_gdf.empty:
                road, dist = geo_utils.find_nearest_feature(erratic_point, colonial_roads_gdf)
                results["nearest_colonial_road_dist"] = dist if dist != float('inf') else None
                if road is not None: results["nearest_colonial_road_name"] = road.get('name') # Or other relevant attribute
                logger.info(f"Nearest colonial road: {results.get('nearest_colonial_road_name')} at {results.get('nearest_colonial_road_dist')}m")
        except Exception as e: logger.error(f"Error processing colonial roads for {erratic_id}: {e}", exc_info=True)
    
    # --- Terrain Analysis (DEM related, geo_utils.load_dem_data will need to be robust or use data_pipeline if possible) ---
    # This part remains dependent on how geo_utils.load_dem_data is refactored.
    # For now, we assume it correctly finds/loads the necessary DEM tile(s), perhaps using `data_pipeline` internally
    # or by expecting manually placed files for sources like 'srtm90_csi_elevation' or specific GMTED tiles.
    try:
        dem_data_info = geo_utils.load_dem_data(point=erratic_point) # This needs to return path or rasterio object
        if dem_data_info and dem_data_info.get('raster_path'): # Assuming it returns a dict with path
            raster_path = dem_data_info['raster_path']
            point_elevation = geo_utils.get_elevation_at_point(erratic_point, raster_path)
            if point_elevation is not None: results['elevation_dem'] = point_elevation
            # Update elevation_category if DEM elevation is available and different
            if 'elevation_dem' in results: results['elevation_category'] = geo_utils.get_elevation_category(results['elevation_dem'])
            
            landscape_metrics = geo_utils.calculate_landscape_metrics(erratic_point, raster_path, radius_m=500)
            geomorph_context = geo_utils.determine_geomorphological_context(erratic_point, raster_path)
            if not np.isnan(landscape_metrics.get('ruggedness_tri', np.nan)): results["ruggedness_tri"] = landscape_metrics['ruggedness_tri']
            if geomorph_context.get('landform') != 'unknown': results["terrain_landform"] = geomorph_context['landform']
            if geomorph_context.get('slope_position') != 'unknown': results["terrain_slope_position"] = geomorph_context['slope_position']
            logger.info(f"Terrain context from DEM: {results.get('terrain_landform')}, {results.get('terrain_slope_position')}, TRI: {results.get('ruggedness_tri')}")
        else:
            logger.warning(f"DEM data not available for detailed terrain analysis for erratic ID {erratic_id}. Using estimates.")
            # Fallback logic from original script can be here if needed (based on lat/lon or existing elevation)
            if 'elevation_category' not in results and erratic.get('elevation') is not None:
                 try: results['elevation_category'] = geo_utils.get_elevation_category(float(erratic['elevation']))
                 except: pass
    except Exception as e: logger.error(f"Error during terrain analysis for {erratic_id}: {e}", exc_info=True)

    # Simplified displacement and accessibility (can be refined in geo_utils or specific modules)
    if 'estimated_displacement_dist' not in results: # Example of a field that might be calculated differently
        results["estimated_displacement_dist"] = geo_utils.estimate_displacement_distance(latitude) # Assumes geo_utils has this
    if 'accessibility_score' not in results:
        results["accessibility_score"] = geo_utils.calculate_accessibility_score(results.get("nearest_road_dist"), results.get("nearest_settlement_dist"))

    final_results = {
        "erratic_id": erratic_id,
        "erratic_name": erratic.get('name', 'Unknown'),
        "location": {"longitude": longitude, "latitude": latitude},
        "proximity_analysis": results
    }
    logger.info(f"Proximity analysis completed for erratic ID {erratic_id}")
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Calculate proximity for a glacial erratic')
    parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze')
    parser.add_argument('--features', nargs='+', help='Feature layers to analyze (e.g., lakes, osm_settlements)')
    parser.add_argument('--update-db', action='store_true', help='Update database with results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    
    results = calculate_proximity(args.erratic_id, args.features)
    
    if 'error' in results:
        logger.error(f"Analysis failed: {results['error']}")
        print(json.dumps({"error": results['error']})) # Keep JSON output for Node.js service
        sys.exit(1)
    
    if args.update_db:
        logger.info(f"Updating database for erratic {args.erratic_id}...")
        # Ensure only relevant fields for ErraticAnalyses are passed
        analysis_payload = results.get('proximity_analysis', {})
        success = db_utils.update_erratic_analysis_results(args.erratic_id, analysis_payload)
        results['database_updated'] = success
        logger.info(f"Database update for {args.erratic_id} {'succeeded' if success else 'failed'}")
    
    if args.output:
        file_utils.json_to_file(results, args.output)
    
    print(json.dumps(results, indent=2)) # Keep JSON output for Node.js service
    logger.info("Proximity_analysis.py script finished.")

if __name__ == "__main__":
    # Adjust sys.path for direct execution if utils/data_pipeline are not installed as packages
    # This ensures that when run directly, it can find sibling packages `utils` and `data_pipeline`
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_python_root = os.path.dirname(current_file_dir) # This should be 'python' directory
    if project_python_root not in sys.path:
        sys.path.insert(0, project_python_root)
    # Re-import after path adjustment if necessary, or ensure imports are relative if run as module
    # For simplicity, assuming direct execution might rely on PYTHONPATH or this adjustment.
    from data_pipeline import load_data 
    from utils import db_utils 
    from utils import file_utils 
    from utils import geo_utils 

    sys.exit(main()) 