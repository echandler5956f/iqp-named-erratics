#!/usr/bin/env python3
"""
Proximity Analysis Script for Glacial Erratics

This script calculates distances from a specified erratic to various geographic features.
It can be run directly from the command line with an erratic ID or used as a module.
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional, Union

# Add the parent directory to sys.path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from python.utils.data_loader import load_erratic_by_id, update_erratic_analysis_data, json_to_file
from python.utils.geo_utils import (
    Point, 
    haversine_distance, 
    calculate_distances_to_features,
    categorize_elevation
)

def calculate_proximity(erratic_id: int, feature_layers: Optional[List[str]] = None) -> Dict:
    """
    Calculate proximity metrics for an erratic.
    
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
        feature_layers = ["water_bodies", "settlements", "trails", "roads"]
    
    # Extract location from erratic data
    try:
        longitude = erratic.get('longitude')
        latitude = erratic.get('latitude')
        point = Point(longitude, latitude)
    except (KeyError, TypeError):
        return {"error": "Invalid location data for erratic"}
    
    # Calculate elevation category if elevation is available
    results = {}
    if erratic.get('elevation') is not None:
        results['elevation_category'] = categorize_elevation(erratic.get('elevation'))
    
    # TODO: Load actual feature layers from GIS data
    # For now, we'll demonstrate with placeholder distances
    # This would be replaced with actual GIS data loading and analysis
    
    # Placeholder for demonstration - in a real implementation, these would be loaded from GIS data
    placeholder_distances = {
        "water_bodies": 1200,  # 1.2 km to nearest water body
        "settlements": 3500,   # 3.5 km to nearest settlement
        "trails": 800,         # 800 m to nearest trail
        "roads": 1500          # 1.5 km to nearest road
    }
    
    # Filter to requested feature layers
    for layer in feature_layers:
        if layer in placeholder_distances:
            if layer == "water_bodies":
                results["nearest_water_body_dist"] = placeholder_distances[layer]
            elif layer == "settlements":
                results["nearest_settlement_dist"] = placeholder_distances[layer]
            else:
                results[f"nearest_{layer}_dist"] = placeholder_distances[layer]
    
    return {
        "erratic_id": erratic_id,
        "erratic_name": erratic.get('name', 'Unknown'),
        "location": {
            "longitude": longitude,
            "latitude": latitude
        },
        "proximity_analysis": results
    }

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description='Calculate proximity for a glacial erratic')
    parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze')
    parser.add_argument('--features', nargs='+', help='Feature layers to calculate proximity to')
    parser.add_argument('--update-db', action='store_true', help='Update database with results')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Calculate proximity
    results = calculate_proximity(args.erratic_id, args.features)
    
    # Handle error
    if 'error' in results:
        print(json.dumps({"error": results['error']}))
        return 1
    
    # Update database if requested
    if args.update_db:
        update_data = results.get('proximity_analysis', {})
        success = update_erratic_analysis_data(args.erratic_id, update_data)
        results['database_updated'] = success
    
    # Write to output file if specified
    if args.output:
        json_to_file(results, args.output)
    
    # Print results as JSON to stdout
    print(json.dumps(results, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main()) 