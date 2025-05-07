#!/usr/bin/env python3
"""
Proximity Analysis Script for Glacial Erratics

This script calculates distances from a specified erratic to various geographic features
with a focus on North American native and colonial historical context.
It uses real-world GIS data sources from HydroSHEDS, Native Land Digital, and historical data.
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

# Add the parent directory to sys.path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from python.utils.data_loader import (
    load_erratic_by_id, 
    update_erratic_analysis_data, 
    json_to_file, 
    load_hydro_features,
    load_settlements,
    load_roads,
    load_native_territories,
    load_colonial_settlements,
    load_colonial_roads,
    load_dem_data,
    get_db_connection
)
from python.utils.geo_utils import (
    Point, 
    haversine_distance, 
    calculate_distances_to_features,
    find_nearest_feature,
    categorize_elevation,
    get_elevation_category,
    calculate_landscape_metrics,
    determine_geomorphological_context,
    find_nearest_feature_db,
    get_elevation_at_point
)

# North American specific context
NORTH_AMERICA_BOUNDS = {
    "xmin": -170.0,  # Western limit (including Alaska)
    "ymin": 15.0,    # Southern limit (including southern Mexico)
    "xmax": -50.0,   # Eastern limit (including eastern Canada)
    "ymax": 75.0     # Northern limit (including northern Canada)
}

def calculate_proximity(erratic_id: int, feature_layers: Optional[List[str]] = None) -> Dict:
    """
    Calculate proximity metrics for an erratic using real North American GIS data.
    
    Args:
        erratic_id: ID of the erratic to analyze
        feature_layers: List of feature layer names to analyze proximity to
        
    Returns:
        Dictionary with proximity analysis results
    """
    # Load erratic data
    erratic = load_erratic_by_id(erratic_id)
    if not erratic:
        return {"error": f"Erratic with ID {erratic_id} not found"}
    
    # Default feature layers if none specified
    if feature_layers is None:
        feature_layers = [
            "water_bodies", 
            "native_territories", 
            "settlements", 
            "colonial_settlements", 
            "roads", 
            "colonial_roads"
        ]
    
    # Extract location from erratic data
    try:
        longitude = erratic.get('longitude')
        latitude = erratic.get('latitude')
        erratic_point = Point(longitude, latitude)
        logger.info(f"Analyzing erratic at {longitude}, {latitude}")
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Invalid location data for erratic: {e}")
        return {"error": "Invalid location data for erratic"}
    
    # Verify the erratic is in North America
    if not (NORTH_AMERICA_BOUNDS["xmin"] <= longitude <= NORTH_AMERICA_BOUNDS["xmax"] and 
            NORTH_AMERICA_BOUNDS["ymin"] <= latitude <= NORTH_AMERICA_BOUNDS["ymax"]):
        logger.warning(f"Erratic appears to be outside North America bounds. Please verify coordinates.")
    
    # Initialize results dictionary
    results = {
        "in_north_america": (NORTH_AMERICA_BOUNDS["xmin"] <= longitude <= NORTH_AMERICA_BOUNDS["xmax"] and 
                             NORTH_AMERICA_BOUNDS["ymin"] <= latitude <= NORTH_AMERICA_BOUNDS["ymax"])
    }
    
    # Calculate elevation category if elevation is available
    if erratic.get('elevation') is not None:
        try:
            elevation = float(erratic.get('elevation'))
            elevation_category = get_elevation_category(elevation)
            results['elevation_category'] = elevation_category
            logger.info(f"Elevation category: {elevation_category} (from {elevation}m)")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not categorize elevation: {e}")
    
    # Establish database connection once if needed for DB queries
    conn = None
    if feature_layers and any(f in ['lakes', 'rivers'] for f in feature_layers):
        conn = get_db_connection()
        if not conn:
            logger.warning("Could not establish database connection for direct feature queries. Falling back to loading files.")

    # --- Water Bodies Analysis ---
    if not feature_layers or 'lakes' in feature_layers or 'rivers' in feature_layers:
        nearest_water_dist = float('inf')
        nearest_water_name = None
        nearest_water_type = None

        # Prefer DB query for large hydro datasets if connection available
        if conn and ('lakes' in feature_layers or not feature_layers):
            logger.info("Querying nearest lake from database...")
            # Assumes a table named 'HydroLAKES' with geometry col 'geom' and 'Hylak_id', 'Lake_name' attributes
            # *** Adjust table_name, geom_col, feature_id_col, attrs_to_select as per your actual DB schema ***
            lake_feature, lake_dist = find_nearest_feature_db(
                erratic_point, conn, 
                table_name='HydroLAKES', # <-- Replace with your actual table name for lakes
                geom_col='geom',        # <-- Replace with your actual geometry column
                feature_id_col='Hylak_id', # <-- Replace with your actual ID column
                attrs_to_select=['Lake_name'] # <-- Add other attributes if needed
            )
            if lake_dist < nearest_water_dist:
                 nearest_water_dist = lake_dist
                 nearest_water_name = lake_feature.get('Lake_name') if lake_feature else None
                 nearest_water_type = 'lake'
                 logger.info(f"Nearest lake (DB Query): {nearest_water_name} at {nearest_water_dist:.1f}m")
        elif 'lakes' in feature_layers or not feature_layers:
            # Fallback to loading file if DB connection failed or not specified
            logger.info("Loading lake data from file...")
            water_bodies_gdf = load_hydro_features('lakes')
            if not water_bodies_gdf.empty:
                logger.info(f"Loaded {len(water_bodies_gdf)} lakes")
                lake_feature, lake_dist = find_nearest_feature(erratic_point, water_bodies_gdf)
                if lake_dist < nearest_water_dist:
                     nearest_water_dist = lake_dist
                     nearest_water_name = lake_feature.get('Lake_name') if lake_feature else None
                     nearest_water_type = 'lake'
                     logger.info(f"Nearest lake (File Load): {nearest_water_name} at {nearest_water_dist:.1f}m")
            else:
                logger.warning("Could not load lake data.")

        # Add similar logic for rivers if needed, using find_nearest_feature_db
        # if conn and ('rivers' in feature_layers or not feature_layers):
        #     logger.info("Querying nearest river from database...")
        #     # river_feature, river_dist = find_nearest_feature_db(...) 
        #     # ... update nearest_water_dist etc.
        # elif 'rivers' in feature_layers or not feature_layers:
        #    # Fallback to load_hydro_features('rivers')
        
        results["nearest_water_body_dist"] = nearest_water_dist if nearest_water_dist != float('inf') else None
        results["nearest_water_body_name"] = nearest_water_name
        results["nearest_water_body_type"] = nearest_water_type

    # --- Native Territory Analysis ---
    if "native_territories" in feature_layers:
        logger.info("Loading Native American territory data...")
        try:
            native_territories = load_native_territories()
            if not native_territories.empty:
                logger.info(f"Loaded {len(native_territories)} native territories")
                territory, distance = find_nearest_feature(erratic_point, native_territories)
                
                if territory:
                    # These are extra informational fields, not directly in ErraticAnalysis model
                    results["nearest_native_territory_name"] = territory.get('Name', 'Unknown') 
                    results["nearest_native_territory_nation"] = territory.get('Nation', 'Unknown')
                    results["on_native_territory"] = distance < 100  # Within 100m
                    
                    logger.info(f"Nearest native territory: {results.get('nearest_native_territory_name')} ({results.get('nearest_native_territory_nation')}) at {results['nearest_native_territory_dist']:.1f}m")
                    if results.get("on_native_territory"):
                        logger.info(f"Erratic is on {results.get('nearest_native_territory_name')} territory")
            else:
                logger.warning("No native territory data available")
        except Exception as e:
            logger.error(f"Error processing native territories: {e}")
    
    # Process modern settlements
    if "settlements" in feature_layers:
        logger.info("Loading modern settlement data...")
        try:
            # Load settlements from North America
            settlements = load_settlements('north-america')
            if not settlements.empty:
                logger.info(f"Loaded {len(settlements)} settlements")
                settlement, distance = find_nearest_feature(erratic_point, settlements)
                
                if settlement:
                    results["nearest_settlement_dist"] = distance
                    results["nearest_settlement_name"] = settlement.get('name', 'Unknown')
                    results["nearest_settlement_type"] = settlement.get('place_type', 'Unknown')
                    
                    logger.info(f"Nearest settlement: {results['nearest_settlement_name']} ({results['nearest_settlement_type']}) at {distance:.1f}m")
            else:
                logger.warning("No settlement data available")
        except Exception as e:
            logger.error(f"Error processing settlements: {e}")
    
    # Process colonial settlements
    if "colonial_settlements" in feature_layers:
        logger.info("Loading colonial settlement data...")
        try:
            colonial_settlements = load_colonial_settlements()
            if not colonial_settlements.empty:
                logger.info(f"Loaded {len(colonial_settlements)} colonial settlements")
                settlement, distance = find_nearest_feature(erratic_point, colonial_settlements)
                
                if settlement:
                    results["nearest_colonial_settlement_dist"] = distance
                    results["nearest_colonial_settlement_name"] = settlement.get('name', 'Unknown')
                    
                    # Include founding date if available
                    if 'founded' in settlement:
                        results["nearest_colonial_settlement_founded"] = int(settlement.get('founded', 0))
                    
                    # Include colonial power if available
                    if 'colony' in settlement:
                        results["nearest_colonial_settlement_colony"] = settlement.get('colony', 'Unknown')
                    
                    logger.info(f"Nearest colonial settlement: {results['nearest_colonial_settlement_name']} at {distance:.1f}m")
            else:
                logger.warning("No colonial settlement data available")
        except Exception as e:
            logger.error(f"Error processing colonial settlements: {e}")
    
    # Process modern roads
    if "roads" in feature_layers:
        logger.info("Loading modern road data...")
        results['nearest_road_dist'] = None # Initialize
        try:
            # Load modern roads from North America
            roads = load_roads('north-america', include_historical=False) # This calls load_modern_roads
            if not roads.empty:
                logger.info(f"Loaded {len(roads)} modern roads")
                road, distance = find_nearest_feature(erratic_point, roads)
                
                results['nearest_road_dist'] = distance if distance != float('inf') else None
                if road:
                    # These are extra informational fields
                    results["nearest_road_name"] = road.get('name', 'Unknown')
                    results["nearest_road_type"] = road.get('highway', road.get('road_type', 'Unknown'))
                    
                    logger.info(f"Nearest modern road: {results.get('nearest_road_name')} ({results.get('nearest_road_type')}) at {results['nearest_road_dist']:.1f}m")
            else:
                logger.warning("No modern road data available")
        except Exception as e:
            logger.error(f"Error processing modern roads: {e}")
    
    # Process colonial roads
    if "colonial_roads" in feature_layers:
        logger.info("Loading colonial road data...")
        try:
            colonial_roads = load_colonial_roads()
            if not colonial_roads.empty:
                logger.info(f"Loaded {len(colonial_roads)} colonial roads")
                road, distance = find_nearest_feature(erratic_point, colonial_roads)
                
                if road:
                    results["nearest_colonial_road_dist"] = distance
                    results["nearest_colonial_road_name"] = road.get('name', 'Unknown')
                    
                    # Include historical period if available
                    if 'year' in road:
                        results["nearest_colonial_road_year"] = int(road.get('year', 0))
                    
                    logger.info(f"Nearest colonial road: {results['nearest_colonial_road_name']} at {distance:.1f}m")
            else:
                logger.warning("No colonial road data available")
        except Exception as e:
            logger.error(f"Error processing colonial roads: {e}")
    
    # --- Terrain Analysis ---
    dem_path = None # Initialize dem_path
    try:
        dem_path = load_dem_data()
    except Exception as e:
        logger.error(f"Failed to load DEM data: {e}")

    if dem_path:
        try:
            logger.info("Analyzing point elevation...")
            point_elevation = get_elevation_at_point(erratic_point, dem_path)
            if point_elevation is not None:
                # Update elevation in results if not already present or if more accurate
                # The Erratic model might already have an elevation, but DEM provides standardized one
                results['elevation_dem'] = point_elevation
                results['elevation_category'] = get_elevation_category(point_elevation)
                logger.info(f"Elevation from DEM: {point_elevation:.1f}m (Category: {results['elevation_category']})")
            else:
                logger.warning("Could not retrieve elevation from DEM for this point.")
        except Exception as e:
            logger.error(f"Error getting point elevation: {e}")
            
        try:
            logger.info("Analyzing terrain context (landscape metrics & geomorphology)...")
            # Use a moderate radius for landscape metrics, adjust as needed
            landscape_metrics = calculate_landscape_metrics(erratic_point, dem_path, radius_m=500) 
            geomorphological_context = determine_geomorphological_context(erratic_point, dem_path)
            
            # Add relevant metrics to results if they are calculated
            # Ensure these keys exist in ErraticAnalyses model or are for informational purposes only
            if not np.isnan(landscape_metrics.get('ruggedness_tri', np.nan)):
                results["ruggedness_tri"] = landscape_metrics['ruggedness_tri']
            if geomorphological_context.get('landform', 'unknown') != 'unknown':
                results["terrain_landform"] = geomorphological_context['landform']
            if geomorphological_context.get('slope_position', 'unknown') != 'unknown':
                 results["terrain_slope_position"] = geomorphological_context['slope_position']
            
            logger.info(f"Terrain context: Landform={results.get('terrain_landform')}, Slope Position={results.get('terrain_slope_position')}, TRI={results.get('ruggedness_tri')}")
        except Exception as e:
            logger.error(f"Error analyzing terrain context: {e}")
    else:
        logger.warning("DEM data not available, skipping terrain analysis.")

    # --- Other Calculated Fields ---

    # Estimate displacement distance (simplified example)
    try:
        # This is a simplified model - a real implementation would use actual glacial flow models
        # and geological data for the specific region in North America
        
        # Different displacement estimates based on region
        if latitude > 60:  # Northern Canada/Alaska
            displacement_est = 12000.0  # Typically longer in northern regions
        elif latitude > 45:  # Northern US/Southern Canada
            displacement_est = 8500.0   # Moderate in central regions
        else:  # Southern US
            displacement_est = 5000.0   # Typically shorter in southern extent of glaciation
            
        results["estimated_displacement_dist"] = displacement_est
        logger.info(f"Estimated displacement: {results['estimated_displacement_dist']}m")
    except Exception as e:
        logger.error(f"Error estimating displacement: {e}")
    
    # Calculate accessibility score (1-5 scale)
    try:
        # Base score on distance to nearest road and settlement
        road_dist = results.get("nearest_road_dist", float('inf'))
        settlement_dist = results.get("nearest_settlement_dist", float('inf'))
        
        if road_dist < 100 and settlement_dist < 5000:
            accessibility = 5  # Very accessible
        elif road_dist < 500 and settlement_dist < 10000:
            accessibility = 4  # Accessible
        elif road_dist < 2000 and settlement_dist < 20000:
            accessibility = 3  # Moderately accessible
        elif road_dist < 5000 and settlement_dist < 50000:
            accessibility = 2  # Remote
        else:
            accessibility = 1  # Very remote
        
        results["accessibility_score"] = accessibility
        logger.info(f"Accessibility score: {results['accessibility_score']}/5")
    except Exception as e:
        logger.error(f"Error calculating accessibility: {e}")
    
    # Combine all results
    analysis_result = {
        "erratic_id": erratic_id,
        "erratic_name": erratic.get('name', 'Unknown'),
        "location": {
            "longitude": longitude,
            "latitude": latitude
        },
        "proximity_analysis": results
    }
    
    logger.info(f"Analysis complete for erratic {erratic_id}")

    # Close DB connection if it was opened
    if conn:
        conn.close()

    return analysis_result

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description='Calculate proximity for a glacial erratic')
    parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze')
    parser.add_argument('--features', nargs='+', help='Feature layers to calculate proximity to')
    parser.add_argument('--update-db', action='store_true', help='Update database with results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting proximity analysis for erratic {args.erratic_id}")
    results = calculate_proximity(args.erratic_id, args.features)
    
    # Handle error
    if 'error' in results:
        logger.error(f"Analysis error: {results['error']}")
        print(json.dumps({"error": results['error']}))
        return 1
    
    # Update database if requested
    if args.update_db:
        logger.info(f"Updating database with analysis results for erratic {args.erratic_id}")
        update_data = results.get('proximity_analysis', {})
        success = update_erratic_analysis_data(args.erratic_id, update_data)
        results['database_updated'] = success
        logger.info(f"Database update {'succeeded' if success else 'failed'}")
    
    # Write to output file if specified
    if args.output:
        logger.info(f"Writing results to {args.output}")
        json_to_file(results, args.output)
    
    # Print results as JSON to stdout
    print(json.dumps(results, indent=2))
    logger.info("Analysis complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 