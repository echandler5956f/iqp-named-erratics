#!/usr/bin/env python3
"""
Test script demonstrating manual data integration with the data pipeline.

This script shows how to add manually downloaded data to the pipeline
and use it alongside automatically downloaded data sources.
"""

import os
import tempfile
import geopandas as gpd
from shapely.geometry import Point
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the pipeline system
from data_pipeline import load_data, add_manual_data, registry

def create_sample_data(filepath):
    """Create a sample GeoJSON file for testing"""
    # Create some sample points using a simpler approach
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-73.5, 45.5]},
                "properties": {"id": 1, "name": "Special Site A"}
            },
            {
                "type": "Feature", 
                "geometry": {"type": "Point", "coordinates": [-74.0, 45.8]},
                "properties": {"id": 2, "name": "Special Site B"}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-73.8, 45.3]},
                "properties": {"id": 3, "name": "Special Site C"}
            }
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(geojson_data, f)
    print(f"Created sample data at: {filepath}")

def main():
    print("=== Testing Manual Data Integration ===\n")
    
    # Create a temporary file to simulate manually downloaded data
    with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
        sample_data_path = tmp.name
    
    try:
        # Create sample data file
        create_sample_data(sample_data_path)
        
        # 1. Add the manual data source to the pipeline
        print("1. Adding manual data source to pipeline...")
        add_manual_data(
            name='special_access_sites',
            local_path=sample_data_path,
            format='geojson',
            description='Special access sites requiring manual download',
            access_level='restricted',
            source_agency='Special Research Institute'
        )
        
        # 2. List all sources to verify it was added
        print(f"2. Available data sources: {registry.list_sources()}")
        print(f"   Manual sources: {registry.list_manual_sources()}")
        
        # 3. Load the manual data just like any other source
        print("\n3. Loading manual data...")
        special_data = load_data('special_access_sites')
        print(f"   Loaded {len(special_data)} features")
        print(f"   Columns: {list(special_data.columns)}")
        print(f"   CRS: {special_data.crs}")
        
        # 4. Show that it integrates with existing pipeline features
        print("\n4. Testing cache integration...")
        # Load again - should come from cache
        cached_data = load_data('special_access_sites')
        print(f"   Second load (from cache): {len(cached_data)} features")
        
        # 5. Test error handling for missing files
        print("\n5. Testing error handling...")
        try:
            add_manual_data(
                name='missing_data',
                local_path='/nonexistent/path/data.shp',
                format='shapefile'
            )
            missing_data = load_data('missing_data')
        except Exception as e:
            print(f"   Expected error for missing file: {type(e).__name__}: {e}")
        
        # 6. Show integration with automatic sources
        print("\n6. Integration with automatic sources...")
        print("   Manual data works alongside automatic downloads:")
        
        # Try to load an automatic source (if available)
        available_sources = [s for s in registry.list_sources() 
                           if s != 'special_access_sites' and s != 'missing_data']
        
        if available_sources:
            auto_source = available_sources[0]
            print(f"   Available automatic source: {auto_source}")
            # Note: We won't actually load it to avoid download time
            print(f"   Would load with: load_data('{auto_source}')")
        
        print("\n=== Manual Data Integration Test Complete ===")
        print("\nUsage Summary:")
        print("1. Download data manually to your local system")
        print("2. Register it: add_manual_data('name', '/path/to/file.ext', format='...')")
        print("3. Use it: data = load_data('name')")
        print("4. Enjoys all pipeline benefits: caching, format processing, etc.")
        
    finally:
        # Cleanup
        if os.path.exists(sample_data_path):
            os.unlink(sample_data_path)

if __name__ == '__main__':
    main() 