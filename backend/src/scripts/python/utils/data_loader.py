#!/usr/bin/env python3
"""
Data loading utilities for interacting with the PostgreSQL database.
Provides functions to load erratic data and related geographic information.
"""

import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Tuple, Union

def get_db_connection():
    """Establish a connection to the PostgreSQL database using environment variables."""
    # Try to get database configuration from environment variables
    try:
        db_config = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "database": os.environ.get("DB_NAME"),
            "user": os.environ.get("DB_USER"),
            "password": os.environ.get("DB_PASSWORD"),
            "port": os.environ.get("DB_PORT", 5432)
        }
        
        # Check if any required config is missing
        if not all([db_config["database"], db_config["user"], db_config["password"]]):
            raise ValueError("Database configuration incomplete. Check environment variables.")
        
        conn = psycopg2.connect(**db_config)
        return conn
    except (Exception, psycopg2.Error) as error:
        print(f"Error connecting to PostgreSQL database: {error}")
        return None

def load_erratics() -> gpd.GeoDataFrame:
    """
    Load all erratics from the database into a GeoDataFrame.
    
    Returns:
        GeoDataFrame containing all erratics with point geometry
    """
    conn = get_db_connection()
    if not conn:
        return gpd.GeoDataFrame()
    
    try:
        # Query to extract data with PostGIS ST_AsText to convert geometry to WKT
        query = """
        SELECT 
            id, name, 
            ST_AsText(location) AS geometry,
            elevation, size_meters, rock_type, 
            estimated_age, discovery_date, description,
            cultural_significance, historical_notes,
            usage_type, cultural_significance_score,
            has_inscriptions, accessibility_score,
            size_category, nearest_water_body_dist,
            nearest_settlement_dist, elevation_category,
            geological_type, estimated_displacement_dist
        FROM "Erratics"
        """
        
        # Load into pandas DataFrame first
        df = pd.read_sql_query(query, conn)
        
        # Convert WKT to shapely geometry
        df['geometry'] = df['geometry'].apply(lambda wkt: Point.from_wkt(wkt))
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        
        return gdf
    except Exception as e:
        print(f"Error loading erratics: {e}")
        return gpd.GeoDataFrame()
    finally:
        conn.close()

def load_erratic_by_id(erratic_id: int) -> Dict:
    """
    Load a single erratic by ID.
    
    Args:
        erratic_id: The ID of the erratic to load
        
    Returns:
        Dictionary containing erratic data
    """
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query to extract data with PostGIS ST_AsText to convert geometry to WKT
        query = """
        SELECT 
            id, name, 
            ST_AsText(location) AS geometry,
            ST_X(location) AS longitude,
            ST_Y(location) AS latitude,
            elevation, size_meters, rock_type, 
            estimated_age, discovery_date, description,
            cultural_significance, historical_notes,
            usage_type, cultural_significance_score,
            has_inscriptions, accessibility_score,
            size_category, nearest_water_body_dist,
            nearest_settlement_dist, elevation_category,
            geological_type, estimated_displacement_dist
        FROM "Erratics"
        WHERE id = %s
        """
        
        cursor.execute(query, (erratic_id,))
        result = cursor.fetchone()
        
        if result:
            # Convert to dictionary
            return dict(result)
        else:
            return {}
    except Exception as e:
        print(f"Error loading erratic {erratic_id}: {e}")
        return {}
    finally:
        conn.close()

def update_erratic_analysis_data(erratic_id: int, data: Dict) -> bool:
    """
    Update the analysis fields for an erratic.
    
    Args:
        erratic_id: The ID of the erratic to update
        data: Dictionary of fields to update
        
    Returns:
        Success status (True/False)
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    # Filter to only include valid fields
    valid_fields = [
        'usage_type', 'cultural_significance_score',
        'has_inscriptions', 'accessibility_score',
        'size_category', 'nearest_water_body_dist',
        'nearest_settlement_dist', 'elevation_category',
        'geological_type', 'estimated_displacement_dist'
    ]
    
    update_data = {k: v for k, v in data.items() if k in valid_fields}
    if not update_data:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Build update query
        fields = ", ".join([f'"{k}" = %s' for k in update_data.keys()])
        query = f'UPDATE "Erratics" SET {fields} WHERE id = %s'
        
        # Execute with values
        values = list(update_data.values()) + [erratic_id]
        cursor.execute(query, values)
        conn.commit()
        
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating erratic {erratic_id}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def json_to_file(data: Dict, output_file: str) -> bool:
    """
    Write JSON data to a file.
    
    Args:
        data: Dictionary to serialize as JSON
        output_file: Path to write the file
        
    Returns:
        Success status (True/False)
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        return False

def file_to_json(input_file: str) -> Dict:
    """
    Read JSON data from a file.
    
    Args:
        input_file: Path to the JSON file
        
    Returns:
        Dictionary containing the parsed JSON
    """
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading from {input_file}: {e}")
        return {}