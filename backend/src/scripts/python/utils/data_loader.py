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
import requests
import zipfile
import tempfile
import logging
import sqlite3
from pathlib import Path
from io import BytesIO
import hashlib
import pyarrow.feather as feather
from dotenv import load_dotenv, find_dotenv
try:
    from shapely.wkb import loads as wkb_loads, dumps as wkb_dumps
except ImportError:
    # For older shapely versions
    from shapely.wkb import loads as wkb_loads
    from shapely.wkb import dumps as wkb_dumps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for GIS data storage
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
GIS_DATA_DIR = os.path.join(DATA_DIR, 'gis')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

# URLs for authoritative GIS data sources
DATA_URLS = {
    # HydroSHEDS Global hydrological data (rivers, watersheds)
    'hydrosheds_rivers': 'https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip',
    'hydrosheds_lakes': 'https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip',
    'hydrosheds_basins': 'https://data.hydrosheds.org/file/hydrobasins/standard/hybas_na_lev05_v1c.zip',
    
    # North American OSM data extracts (settlements, roads) from Geofabrik
    'osm_north_america': 'https://download.geofabrik.de/north-america-latest.osm.pbf',
    'osm_us': 'https://download.geofabrik.de/north-america/us-latest.osm.pbf',
    'osm_canada': 'https://download.geofabrik.de/north-america/canada-latest.osm.pbf',
    
    # Native American settlement and territory data
    'native_territories': 'https://nativeland.info/api/index.php?maps=territories',
    'native_languages': 'https://nativeland.info/api/index.php?maps=languages',
    'native_treaties': 'https://nativeland.info/api/index.php?maps=treaties',
    
    # US National Historical GIS data (for historical settlements)
    'nhgis_historical': 'https://www.nhgis.org/sites/www.nhgis.org/files/nhgis_shapefiles.zip',
    
    # Colonial era road data from the Digital Archive of American Roads and Trails
    'colonial_roads': 'https://www.daart.online/data/roads_colonial_era.geojson',
    
    # Global Digital Elevation Model - North America subset
    'elevation_dem_na': 'https://opentopography.s3.sdsc.edu/raster/SRTM_GL1_Ellip/north_america_30m.tif',
}

# Ensure directories exist
os.makedirs(GIS_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
for subdir in ['hydro', 'settlements', 'roads', 'elevation', 'native', 'colonial']:
    os.makedirs(os.path.join(GIS_DATA_DIR, subdir), exist_ok=True)

# --- Caching Helper Functions ---

def _get_cache_path(source_filepath: str, params: Optional[Dict] = None) -> str:
    """Generates a consistent cache file path based on the source file and parameters."""
    filename = os.path.basename(source_filepath)
    # Create a hash based on filename and any relevant parameters to avoid collisions
    hasher = hashlib.md5()
    hasher.update(filename.encode())
    if params:
        hasher.update(json.dumps(params, sort_keys=True).encode())
    
    cache_filename = f"{Path(filename).stem}_{hasher.hexdigest()}.feather"
    return os.path.join(CACHE_DIR, cache_filename)

def _cache_gdf(gdf: gpd.GeoDataFrame, cache_filepath: str):
    """Saves a GeoDataFrame to a Feather cache file."""
    try:
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        
        # Create a copy to avoid modifying the original
        gdf_copy = gdf.copy()
        
        # Save CRS to restore later (store as a separate file for simplicity)
        crs_path = cache_filepath + ".crs"
        if gdf_copy.crs:
            with open(crs_path, 'w') as f:
                f.write(str(gdf_copy.crs))
        
        # Convert geometry to WKB bytes for storage
        if 'geometry' in gdf_copy.columns:
            # Store geometry column name
            geometry_col = gdf_copy._geometry_column_name
            with open(cache_filepath + ".geom_col", 'w') as f:
                f.write(geometry_col)
            
            # Convert geometry to WKB and store as bytes
            gdf_copy['_wkb'] = gdf_copy[geometry_col].apply(lambda geom: wkb_dumps(geom) if geom else None)
            
            # Drop the geometry column as it can't be serialized
            gdf_copy = gdf_copy.drop(columns=[geometry_col])
        
        # Feather requires string column names
        gdf_copy.columns = gdf_copy.columns.astype(str)
        
        # Write to feather
        feather.write_feather(gdf_copy, cache_filepath)
        logger.info(f"Cached GeoDataFrame to {cache_filepath}")
    except Exception as e:
        logger.error(f"Error saving GDF to cache {cache_filepath}: {e}")

def _load_gdf_from_cache(cache_filepath: str, source_filepath: str) -> Optional[gpd.GeoDataFrame]:
    """Loads a GeoDataFrame from cache if it exists and newer than the source file."""
    try:
        if os.path.exists(cache_filepath):
            # Check if cache is newer than the source file
            cache_mtime = os.path.getmtime(cache_filepath)
            source_mtime = os.path.getmtime(source_filepath) if os.path.exists(source_filepath) else 0
            
            if cache_mtime > source_mtime:
                logger.info(f"Loading GeoDataFrame from cache: {cache_filepath}")
                df = feather.read_feather(cache_filepath)
                
                # Restore CRS
                crs = None
                crs_path = cache_filepath + ".crs"
                if os.path.exists(crs_path):
                    with open(crs_path, 'r') as f:
                        crs = f.read().strip()
                
                # Check for WKB geometry data
                if '_wkb' in df.columns:
                    # Get geometry column name
                    geom_col = 'geometry'  # Default
                    geom_col_path = cache_filepath + ".geom_col"
                    if os.path.exists(geom_col_path):
                        with open(geom_col_path, 'r') as f:
                            geom_col = f.read().strip()
                    
                    # Convert WKB back to geometry objects
                    df[geom_col] = df['_wkb'].apply(lambda wkb: wkb_loads(wkb) if wkb else None)
                    df = df.drop(columns=['_wkb'])
                    
                    # Create GeoDataFrame with the right geometry column
                    gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs=crs)
                else:
                    # No geometry column found, return as regular DataFrame
                    gdf = gpd.GeoDataFrame(df, crs=crs)
                
                return gdf
            else:
                logger.info(f"Cache file {cache_filepath} is older than source {source_filepath}. Re-loading.")
        else:
            logger.info(f"No cache file found at {cache_filepath}")
    except Exception as e:
        logger.error(f"Error loading GDF from cache {cache_filepath}: {e}")
    
    return None

# --- End Caching Helper Functions ---

def get_db_connection():
    """Establish a connection to the PostgreSQL database using environment variables or a .env file."""
    
    # Load environment variables from .env file if it exists
    # Looks for .env in the current directory or parent directories
    dotenv_path = find_dotenv()
    if dotenv_path:
         logger.info(f"Loading environment variables from: {dotenv_path}")
         load_dotenv(dotenv_path=dotenv_path)
    else:
         logger.info("No .env file found, relying on system environment variables.")

    # Try to get database configuration from environment variables
    try:
        db_config = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "database": os.environ.get("DB_NAME"),
            "user": os.environ.get("DB_USER"),
            "password": os.environ.get("DB_PASSWORD"),
            # Ensure port is an integer if provided
            "port": int(os.environ.get("DB_PORT", 5432)) 
        }
        
        # Check if any required config is missing
        missing_vars = [k for k, v in db_config.items() if v is None and k in ['database', 'user', 'password']]
        if missing_vars:
            raise ValueError(f"Database configuration incomplete. Missing environment variables: {', '.join(missing_vars)}")
        
        logger.info(f"Attempting to connect to database '{db_config['database']}' on {db_config['host']}:{db_config['port']} as user '{db_config['user']}'")
        conn = psycopg2.connect(**db_config)
        logger.info("Database connection successful.")
        return conn
    except ValueError as ve:
        logger.error(f"Database configuration error: {ve}")
        return None
    except (Exception, psycopg2.Error) as error:
        logger.error(f"Error connecting to PostgreSQL database: {error}")
        return None

def load_erratics() -> gpd.GeoDataFrame:
    """
    Load all erratics from the database into a GeoDataFrame using geopandas.read_postgis.
    
    Returns:
        GeoDataFrame containing all erratics with point geometry
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to establish database connection for loading erratics.")
        return gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs="EPSG:4326") # Return empty GDF with schema
    
    try:
        # Use geopandas.read_postgis for efficient loading including geometry
        # Ensure the geometry column name in the query matches 'geom' expected by read_postgis by default,
        # or specify `geom_col` parameter. Using 'location' as the geometry column.
        query = """
        SELECT 
            id, name, 
            location, -- Select the geometry column directly
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
        
        gdf = gpd.read_postgis(query, conn, geom_col='location', crs='EPSG:4326') # Assuming source CRS is 4326
        
        # Ensure 'id' column exists and handle potential empty results
        if 'id' not in gdf.columns and not gdf.empty:
             logger.warning("Loaded erratics GeoDataFrame is missing 'id' column.")
        elif gdf.empty:
             logger.info("No erratics found in the database.")
             # Return empty GeoDataFrame with expected columns if needed
             return gpd.GeoDataFrame(columns=['id', 'name', 'location', 'elevation', 'size_meters', 'rock_type', 
                                             'estimated_age', 'discovery_date', 'description',
                                             'cultural_significance', 'historical_notes',
                                             'usage_type', 'cultural_significance_score',
                                             'has_inscriptions', 'accessibility_score',
                                             'size_category', 'nearest_water_body_dist',
                                             'nearest_settlement_dist', 'elevation_category',
                                             'geological_type', 'estimated_displacement_dist'], 
                                    geometry='location', crs="EPSG:4326")


        return gdf
    except Exception as e:
        logger.error(f"Error loading erratics using read_postgis: {e}")
        # Return empty GeoDataFrame with a minimal schema on error
        return gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs="EPSG:4326")
    finally:
        if conn:
            conn.close()

def load_erratic_by_id(erratic_id: int) -> Optional[Dict]:
    """
    Load a single erratic by ID from the database
    
    Args:
        erratic_id: ID of the erratic to load
        
    Returns:
        Dictionary with erratic data or None if not found
    """
    try:
        # Connect to the database
        from django.db import connection
        with connection.cursor() as cursor:
            # Query for the erratic with the given ID
            cursor.execute("""
                SELECT e.id, e.name, ST_X(e.location::geometry) as longitude, 
                       ST_Y(e.location::geometry) as latitude, e.elevation, 
                       e.description, e.cultural_significance, e.historical_notes
                FROM "Erratics" e
                WHERE e.id = %s
            """, [erratic_id])
            
            row = cursor.fetchone()
            
            if row:
                # Convert to dictionary
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
            else:
                return None
    except Exception as e:
        logger.error(f"Error loading erratic by ID: {e}")
        # Fallback to mock data for testing
        return {
            'id': erratic_id,
            'name': f'Test Erratic {erratic_id}',
            'longitude': -73.968285,  # Example coordinates (New York)
            'latitude': 40.785091,
            'elevation': 100.0,
            'description': 'A test erratic for development purposes.',
            'cultural_significance': 'None, this is test data.',
            'historical_notes': 'Created for testing the spatial analysis pipeline.'
        }

def update_erratic_analysis_data(erratic_id: int, data: Dict) -> bool:
    """
    Update or insert analysis results for an erratic in the ErraticAnalyses table.
    
    Args:
        erratic_id: ID of the erratic to update/insert analysis data for.
        data: Dictionary of analysis data to update/insert.
        
    Returns:
        True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        logger.error(f"Failed to get DB connection for updating ErraticAnalyses for erratic_id: {erratic_id}")
        return False

    try:
        with conn.cursor() as cursor:
            # Filter out keys not meant for ErraticAnalyses table or that are None
            # Define known fields of ErraticAnalyses to ensure only valid data is passed
            # This list should be kept in sync with the ErraticAnalysis model definition
            known_analysis_fields = [
                'usage_type', 'cultural_significance_score', 'has_inscriptions',
                'accessibility_score', 'size_category', 'nearest_water_body_dist',
                'nearest_settlement_dist', 'nearest_colonial_settlement_dist',
                'nearest_road_dist', 'nearest_colonial_road_dist',
                'nearest_native_territory_dist', 'elevation_category',
                'geological_type', 'estimated_displacement_dist',
                'vector_embedding', 'vector_embedding_data'
            ]
            
            update_values = {}
            for key, value in data.items():
                if key in known_analysis_fields and value is not None:
                    update_values[key] = value
            
            if not update_values:
                logger.info(f"No valid analysis data provided to update for erratic_id: {erratic_id}")
                return False # Or True, if no update needed is considered success

            # Add erraticId to the values for insertion
            update_values['erraticId'] = erratic_id

            # Prepare for UPSERT
            columns = update_values.keys()
            placeholders = [f"%({col})s" for col in columns]
            
            set_clauses = []
            for col in columns:
                if col != 'erraticId': # Don't try to update the PK itself in the SET part
                    set_clauses.append(f'"{col}" = EXCLUDED."{col}"')
            
            # Include updatedAt timestamp
            if 'updatedAt' not in update_values:
                update_values['updatedAt'] = 'NOW()' # Let PG handle this directly in query
                # For INSERT, we need to ensure it's part of the columns and placeholders
                # if it's not already handled by Sequelize's default value on the model/table.
                # Assuming table has default for createdAt and auto-updates updatedAt.
                # If not, this needs adjustment. For UPSERT, this is usually fine.

            # For INSERT part, ensure all columns in update_values are listed
            insert_columns_str = ", ".join([f'"{col}"' for col in update_values.keys()])
            insert_values_str = ", ".join([f"%({col})s" for col in update_values.keys()])

            # For UPDATE part (conflict)
            update_set_str = ", ".join(set_clauses)
            if 'updatedAt' not in [col for col in columns if col != 'erraticId']:
                 update_set_str += ", \"updatedAt\" = NOW()"
            else: # if updatedAt was explicitly in update_values
                 # it's already in set_clauses, but ensure it's EXCLUDED.updatedAt if needed
                 pass 

            sql = f"""
            INSERT INTO "ErraticAnalyses" ({insert_columns_str})
            VALUES ({insert_values_str})
            ON CONFLICT ("erraticId") DO UPDATE
            SET {update_set_str};
            """
            
            # psycopg2 uses %(key)s for named placeholders in execute method
            cursor.execute(sql, update_values)
            conn.commit()
            logger.info(f"Successfully upserted analysis data for erratic_id: {erratic_id}")
            return True
    except Exception as e:
        logger.error(f"Error upserting analysis data for erratic_id {erratic_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def json_to_file(data: Dict, filepath: str) -> bool:
    """
    Save JSON data to a file
    
    Args:
        data: Dictionary to save
        filepath: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to file: {e}")
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

def download_and_extract_data(data_key: str, region: Optional[str] = None, force_download: bool = False) -> Optional[str]:
    """
    Download and extract a GIS dataset if not already available
    
    Args:
        data_key: Key from DATA_URLS dictionary
        region: Optional region parameter for region-specific data
        force_download: Force re-download even if file exists
        
    Returns:
        Path to the extracted data directory or None if failed
    """
    # Build target path
    subdir = 'default'
    if 'hydro' in data_key:
        subdir = 'hydro'
    elif 'osm' in data_key:
        subdir = 'settlements'
    elif 'roads' in data_key or 'trails' in data_key:
        subdir = 'roads'
    elif 'elevation' in data_key or 'dem' in data_key:
        subdir = 'elevation'
    elif 'native' in data_key:
        subdir = 'native'
    elif 'colonial' in data_key or 'nhgis' in data_key:
        subdir = 'colonial'
    
    target_dir = os.path.join(GIS_DATA_DIR, subdir, data_key)
    if os.path.exists(target_dir) and not force_download:
        logger.info(f"Data already exists at {target_dir}")
        return target_dir
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Get URL, handle region-specific URLs
    url = DATA_URLS.get(data_key, "")
    if not url:
        logger.error(f"No URL found for data key: {data_key}")
        return None
    
    if '{region}' in url and region:
        url = url.format(region=region)
    
    # Download data
    logger.info(f"Downloading data from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract if it's a zip file
        if url.endswith('.zip'):
            logger.info(f"Extracting zip file to {target_dir}")
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.unlink(tmp_path)
        elif url.endswith('.pbf'):
            # For OSM PBF files, we need to convert them to GeoJSON or Shapefile
            # This would require osmium or a similar tool in a full implementation
            logger.warning(f"PBF files require additional processing. Storing as-is for now.")
            os.rename(tmp_path, os.path.join(target_dir, os.path.basename(url)))
        elif url.endswith('.geojson'):
            # Just move the GeoJSON file
            os.rename(tmp_path, os.path.join(target_dir, os.path.basename(url)))
        else:
            # Just move the file
            file_name = url.split('/')[-1]
            os.rename(tmp_path, os.path.join(target_dir, file_name))
        
        logger.info(f"Successfully downloaded and extracted to {target_dir}")
        return target_dir
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return None

def load_gis_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Load GIS data from a file into a GeoDataFrame, using cache if available.
    
    Args:
        filepath: Path to the GIS data file (supports various formats via geopandas)
        
    Returns:
        GeoDataFrame with the loaded data
    """
    cache_filepath = _get_cache_path(filepath)
    
    # Try loading from cache first
    cached_gdf = _load_gdf_from_cache(cache_filepath, filepath)
    if cached_gdf is not None:
        return cached_gdf
        
    # If cache miss or error, load from source
    logger.info(f"Loading GIS data from source: {filepath}")
    try:
        gdf = gpd.GeoDataFrame() # Initialize empty
        if not os.path.exists(filepath):
             logger.error(f"Source file not found: {filepath}")
             # Return empty GeoDataFrame with basic structure
             return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")

        if filepath.endswith('.geojson'):
            gdf = gpd.read_file(filepath)
        elif filepath.endswith('.shp'):
            gdf = gpd.read_file(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if 'longitude' in df.columns and 'latitude' in df.columns:
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs="EPSG:4326" # Assume WGS84 for CSV lon/lat
                )
            # Handle potential WKT column
            elif 'geometry' in df.columns and isinstance(df['geometry'].iloc[0], str):
                 try:
                     from shapely import wkt
                     gdf = gpd.GeoDataFrame(
                         df,
                         geometry=df['geometry'].apply(wkt.loads),
                         crs="EPSG:4326" # Assume WGS84 if CRS not specified
                     )
                 except ImportError:
                      logger.error("Shapely is required to parse WKT geometries from CSV.")
                      raise ValueError("CSV contains WKT geometry but Shapely is not installed.")
                 except Exception as parse_err:
                      logger.error(f"Error parsing WKT geometry from CSV {filepath}: {parse_err}")
                      raise ValueError("Could not parse WKT geometry column in CSV.")
            else:
                raise ValueError("CSV must contain longitude/latitude columns or a WKT geometry column.")
        else:
            # Attempt generic read_file for other geopandas supported formats
            try:
                gdf = gpd.read_file(filepath)
                logger.info(f"Loaded {filepath} using generic geopandas read_file.")
            except Exception as generic_read_error:
                 logger.error(f"Unsupported or invalid file format: {filepath}. Error: {generic_read_error}")
                 raise ValueError(f"Unsupported or invalid file format: {filepath}") from generic_read_error

        # Ensure CRS is set if possible
        if gdf.crs is None:
             logger.warning(f"Loaded data from {filepath} has no CRS defined. Assuming EPSG:4326.")
             gdf.crs = "EPSG:4326"
        
        # Cache the loaded data
        _cache_gdf(gdf.copy(), cache_filepath) # Use copy to avoid modifying original gdf

        return gdf
    except Exception as e:
        logger.error(f"Error loading GIS data from {filepath}: {e}")
        # Return empty GeoDataFrame with basic structure
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")

def load_hydro_features(feature_type: str = 'rivers', region: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load hydrological features (rivers, lakes, etc.) from HydroSHEDS database
    
    Args:
        feature_type: Type of feature ('rivers', 'lakes', 'basins')
        region: Optional region to filter by
        
    Returns:
        GeoDataFrame with the hydrological features
    """
    data_key = None
    if feature_type.lower() == 'rivers':
        data_key = 'hydrosheds_rivers'
    elif feature_type.lower() == 'lakes':
        data_key = 'hydrosheds_lakes'
    elif feature_type.lower() == 'basins':
        data_key = 'hydrosheds_basins'
    else:
        logger.error(f"Unsupported hydrological feature type: {feature_type}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
    
    # Download and extract if needed
    target_dir = download_and_extract_data(data_key, region)
    if not target_dir:
        logger.error(f"Failed to get data directory for {data_key}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
    
    # Find shapefile in the extracted directory
    shapefile = None
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.shp'):
                shapefile = os.path.join(root, file)
                break
        if shapefile:
            break
    
    if not shapefile:
        logger.error(f"No shapefile found in {target_dir}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
    
    # Load the shapefile
    return load_gis_data(shapefile)

def load_native_territories() -> gpd.GeoDataFrame:
    """
    Load Native American territory data from Native Land Digital API, using cache.
    
    Returns:
        GeoDataFrame with native territories
    """
    data_key = 'native_territories'
    # Determine the expected final file path *before* downloading
    # Assuming the API consistently provides data we save as territories.geojson
    # Need a placeholder target directory structure even if download doesn't happen yet
    # This seems slightly complex; let's refine the download logic to return the *final* file path
    # or ensure the filename is predictable.
    
    # Let's adjust download_and_extract_data slightly to handle this better
    # For now, assume the file will be named territories.geojson inside the target dir
    target_dir_base = os.path.join(GIS_DATA_DIR, 'native', data_key)
    geojson_path = os.path.join(target_dir_base, 'territories.geojson')

    # Check cache first using the expected final path
    cache_filepath = _get_cache_path(geojson_path)
    cached_gdf = _load_gdf_from_cache(cache_filepath, geojson_path)
    if cached_gdf is not None:
        return cached_gdf

    # If not cached or cache invalid, proceed to download/load logic
    logger.info(f"Cache miss or invalid for {geojson_path}. Attempting to load/download.")
    target_dir = download_and_extract_data(data_key) # This ensures data is downloaded if needed
    
    if not target_dir:
        logger.error("Failed to download or locate native territories data directory")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
    
    # Reconfirm the expected path after download ensures directory exists
    geojson_path = os.path.join(target_dir, 'territories.geojson') 

    # Check if the GeoJSON file exists (it might not if download_and_extract handled it differently)
    if not os.path.exists(geojson_path):
        # Attempt to fetch from API if the file wasn't created by download_and_extract_data
        # This handles cases where download_and_extract_data might just create the dir
        # or if the API URL wasn't a direct file link.
        logger.info(f"GeoJSON file {geojson_path} not found. Attempting to fetch from API.")
        try:
            api_url = DATA_URLS.get(data_key)
            if not api_url:
                 logger.error(f"API URL for {data_key} not found.")
                 return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")

            response = requests.get(api_url)
            response.raise_for_status()
            api_data = response.json()
            
            # Save the GeoJSON - Use file_to_json helper? No, that reads. Need a save helper.
            os.makedirs(os.path.dirname(geojson_path), exist_ok=True)
            with open(geojson_path, 'w') as f:
                json.dump(api_data, f)
            logger.info(f"Successfully fetched and saved native territories data to {geojson_path}")

        except Exception as e:
            logger.error(f"Error fetching or saving native territories data from API {api_url}: {e}")
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")

    # Now, load the data using the main loading function (which handles caching)
    return load_gis_data(geojson_path)

def load_colonial_settlements() -> gpd.GeoDataFrame:
    """
    Load historical colonial settlement data for North America
    
    Returns:
        GeoDataFrame with colonial settlements
    """
    # Try to load from NHGIS data
    data_key = 'nhgis_historical'
    target_dir = download_and_extract_data(data_key)
    
    if not target_dir:
        logger.warning("Failed to download NHGIS data, using fallback colonial settlement data")
        return _create_fallback_colonial_settlements()
    
    # Search for settlement shapefiles in the NHGIS data
    settlement_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.shp') and ('settlement' in file.lower() or 'town' in file.lower() or 'city' in file.lower()):
                settlement_files.append(os.path.join(root, file))
    
    if not settlement_files:
        logger.warning("No settlement shapefiles found in NHGIS data")
        return _create_fallback_colonial_settlements()
    
    # Load and combine shapefiles
    gdfs = []
    for file in settlement_files[:5]:  # Limit to first 5 to avoid loading too much
        try:
            gdf = gpd.read_file(file)
            gdfs.append(gdf)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not gdfs:
        logger.warning("No settlement data loaded from NHGIS files")
        return _create_fallback_colonial_settlements()
    
    try:
        # Combine all dataframes
        combined_gdf = pd.concat(gdfs)
        return combined_gdf
    except Exception as e:
        logger.error(f"Error combining settlement data: {e}")
        return _create_fallback_colonial_settlements()

def _create_fallback_colonial_settlements() -> gpd.GeoDataFrame:
    """
    Create a fallback GeoDataFrame with simplified historical colonial settlement data.
    
    Returns:
        GeoDataFrame with fallback colonial settlements
    """
    # Fallback to simplified historical data
    data = {
        'name': [
            'Jamestown', 'Plymouth', 'Quebec City', 'St. Augustine', 
            'New Amsterdam (New York)', 'Boston', 'Montreal', 'Philadelphia',
            'Charleston', 'New Orleans', 'Detroit', 'Baltimore'
        ],
        'founded': [
            1607, 1620, 1608, 1565, 
            1624, 1630, 1642, 1682,
            1670, 1718, 1701, 1729
        ],
        'colony': [
            'Virginia', 'Massachusetts', 'New France', 'Spanish Florida',
            'New Netherland', 'Massachusetts', 'New France', 'Pennsylvania',
            'Carolina', 'Louisiana', 'New France', 'Maryland'
        ],
        'longitude': [
            -76.779, -70.668, -71.208, -81.314,
            -74.010, -71.059, -73.588, -75.165,
            -79.931, -90.071, -83.045, -76.612
        ],
        'latitude': [
            37.321, 41.957, 46.813, 29.898,
            40.714, 42.361, 45.501, 39.952,
            32.776, 29.951, 42.331, 39.290
        ]
    }
    
    df = pd.DataFrame(data)
    try:
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        return gdf
    except Exception as e:
        logger.error(f"Error creating fallback colonial settlements: {e}")
        # Last resort - empty GeoDataFrame with the expected structure
        return gpd.GeoDataFrame(
            columns=['name', 'founded', 'colony', 'longitude', 'latitude', 'geometry'],
            geometry='geometry',
            crs="EPSG:4326"
        )

def load_colonial_roads() -> gpd.GeoDataFrame:
    """
    Load historical colonial road data for North America, using cache.
    
    Returns:
        GeoDataFrame with colonial roads
    """
    # Create a fallback colonial roads function in case loading fails
    def _create_fallback_colonial_roads() -> gpd.GeoDataFrame:
        # Simple fallback with major colonial roads
        roads_data = {
            'name': [
                'Boston Post Road', 'King\'s Highway', 'Wilderness Road', 
                'Natchez Trace', 'El Camino Real'
            ],
            'year': [1673, 1650, 1775, 1739, 1769],
            'wkt': [
                'LINESTRING(-71.06 42.36, -72.54 42.10, -73.76 42.65, -74.01 40.71)',
                'LINESTRING(-71.06 42.36, -74.01 40.71, -75.16 39.95, -77.03 38.90, -78.47 35.99, -79.93 32.78, -81.09 32.08)',
                'LINESTRING(-78.64 38.03, -81.63 38.35, -84.50 38.04, -85.76 38.25, -86.78 36.17)',
                'LINESTRING(-90.07 29.95, -90.10 32.30, -88.70 35.15, -86.78 36.17)',
                'LINESTRING(-106.48 31.76, -98.49 29.42, -95.36 29.76)'
            ]
        }
        
        df = pd.DataFrame(roads_data)
        try:
            # Ensure shapely is available for WKT parsing
            from shapely import wkt
            gdf = gpd.GeoDataFrame(
                df,
                geometry=df['wkt'].apply(wkt.loads),
                crs="EPSG:4326"
            )
            return gdf
        except Exception as fallback_err:
            logger.error(f"Error creating fallback colonial roads GDF: {fallback_err}")
            # Return empty GeoDataFrame with expected structure
            return gpd.GeoDataFrame(
                columns=['name', 'year', 'geometry'], 
                geometry='geometry',
                crs="EPSG:4326"
            )
    
    data_key = 'colonial_roads'
    # Try to determine the final path
    target_dir_base = os.path.join(GIS_DATA_DIR, 'roads', data_key)
    expected_filename = 'roads_colonial_era.geojson'
    expected_filepath = os.path.join(target_dir_base, expected_filename)
    
    # Check cache first - even if source doesn't exist yet
    cache_filepath = _get_cache_path(expected_filepath)
    cached_gdf = _load_gdf_from_cache(cache_filepath, expected_filepath)
    if cached_gdf is not None:
        return cached_gdf

    # Try to download/extract if needed
    logger.info(f"Attempting to download colonial roads data")
    target_dir = download_and_extract_data(data_key)
    if not target_dir:
        logger.warning("Failed to download colonial roads data, using fallback")
        fallback_gdf = _create_fallback_colonial_roads()
        # Cache the fallback data for future use
        _cache_gdf(fallback_gdf.copy(), cache_filepath)
        return fallback_gdf
    
    # Check for the expected file in the download directory
    geojson_path = os.path.join(target_dir, expected_filename)
    if not os.path.exists(geojson_path):
        # Look for any GeoJSON file
        found_files = [f for f in os.listdir(target_dir) if f.endswith('.geojson')]
        if len(found_files) == 1:
            geojson_path = os.path.join(target_dir, found_files[0])
            logger.info(f"Found colonial roads GeoJSON: {geojson_path}")
        elif len(found_files) > 1:
            # Multiple files found, use the first one
            geojson_path = os.path.join(target_dir, found_files[0])
            logger.warning(f"Multiple GeoJSON files found in {target_dir}, using {found_files[0]}")
        else:
            # No GeoJSON files found, use fallback
            logger.warning(f"No GeoJSON files found in {target_dir}, using fallback colonial roads data")
            fallback_gdf = _create_fallback_colonial_roads()
            # Cache the fallback data
            _cache_gdf(fallback_gdf.copy(), cache_filepath)
            return fallback_gdf
    
    # Load the GeoJSON file
    try:
        gdf = load_gis_data(geojson_path)
        if gdf.empty:
            logger.warning("Loaded empty colonial roads dataset, using fallback data")
            fallback_gdf = _create_fallback_colonial_roads()
            _cache_gdf(fallback_gdf.copy(), cache_filepath)
            return fallback_gdf
        return gdf
    except Exception as e:
        logger.error(f"Error loading colonial roads from {geojson_path}: {e}")
        fallback_gdf = _create_fallback_colonial_roads()
        _cache_gdf(fallback_gdf.copy(), cache_filepath)
        return fallback_gdf

def load_settlements(region: Optional[str] = 'north-america') -> gpd.GeoDataFrame:
    """
    Load settlement data for North America using OpenStreetMap data (or fallback).
    Attempts to use cached data first.
    
    Args:
        region: Region to load data for (default to 'north-america')
        
    Returns:
        GeoDataFrame with settlements
    """
    # Define a representative cache key/path for the generated fallback data based on region
    # Using a dummy file path in the cache dir structure
    fallback_cache_key = f"fallback_settlements_{region}.geojson" # Use .geojson suffix for consistency
    cache_filepath = _get_cache_path(os.path.join(CACHE_DIR, fallback_cache_key), params={'region': region})

    # Try loading from cache first
    # Note: For generated data, there's no real source file to compare mtime against.
    # We'll load from cache if it exists, assuming it's valid.
    # A more complex system could involve versioning or explicit cache clearing.
    if os.path.exists(cache_filepath):
         try:
             logger.info(f"Loading fallback settlements for region '{region}' from cache: {cache_filepath}")
             # Need to load without source file comparison
             gdf = feather.read_feather(cache_filepath)
             if gdf.crs is None: # Re-apply CRS if lost
                 gdf.crs = "EPSG:4326"
             # Ensure geometry column is set
             if 'geometry' in gdf.columns:
                 gdf = gdf.set_geometry('geometry')
             elif not gdf.empty:
                 geom_col = gdf.select_dtypes(include='geometry').columns
                 if len(geom_col) == 1:
                     gdf = gdf.set_geometry(geom_col[0])
                 else:
                     logger.warning(f"Could not determine geometry column in cached fallback settlements: {cache_filepath}")
             return gdf
         except Exception as e:
             logger.error(f"Error loading fallback settlements from cache {cache_filepath}: {e}. Regenerating.")

    # --- Production code would attempt PBF processing here --- 
    # data_key = ... (determine based on region)
    # target_dir = download_and_extract_data(data_key)
    # if target_dir:
    #     pbf_file = find_pbf_file(target_dir)
    #     if pbf_file:
    #         try:
    #             # Process PBF to GeoDataFrame (using osmium, etc.)
    #             gdf = process_osm_pbf(pbf_file, feature_type='settlements') 
    #             # Cache the processed result
    #             processed_cache_path = _get_cache_path(pbf_file, params={'feature':'settlements'})
    #             _cache_gdf(gdf, processed_cache_path)
    #             return gdf
    #         except Exception as pbf_error:
    #             logger.error(f"Error processing PBF file {pbf_file}: {pbf_error}. Using fallback.")
    # else: 
    #     logger.warning(f"Failed to download OSM data for {region}. Using fallback.")
    # --- End of hypothetical PBF processing --- 

    # If PBF processing is skipped, not implemented, or failed, generate fallback
    logger.warning(f"OSM PBF processing for settlements not implemented or failed for region '{region}'. Generating and caching fallback data.")
    
    # Generate the extended North American cities fallback data
    sample_data = {
        'osm_id': list(range(1, 51)),  # 50 cities
        'name': [
            # US Major Cities
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
            'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
            'Indianapolis', 'San Francisco', 'Seattle', 'Denver', 'Boston',
            # Canadian Major Cities
            'Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Edmonton',
            'Ottawa', 'Winnipeg', 'Quebec City', 'Hamilton', 'Kitchener',
            # Mexican Major Cities
            'Mexico City', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana',
            'León', 'Juárez', 'Culiacán', 'Mérida', 'Hermosillo',
            # Historical Settlements (Colonial/Native)
            'Cahokia', 'Jamestown', 'Plymouth', 'Quebec', 'St. Augustine',
            'Santa Fe', 'Albany', 'Tadoussac', 'Kaskaskia', 'Mobile'
        ],
        'place_type': ['city'] * 30 + ['city'] * 10 + ['historical_site', 'colonial', 'colonial', 'colonial', 'colonial', 'colonial', 'colonial', 'colonial', 'colonial', 'colonial'], # Adjusted types
        'longitude': [
            # US
            -74.006, -118.243, -87.630, -95.369, -112.074,
            -75.165, -98.491, -117.161, -96.797, -121.895,
            -97.733, -81.656, -97.330, -82.999, -80.843,
            -86.158, -122.419, -122.332, -104.991, -71.059,
            # Canada
            -79.347, -73.588, -123.116, -114.071, -113.323,
            -75.697, -97.139, -71.208, -79.877, -80.516,
            # Mexico
            -99.133, -103.349, -100.309, -98.190, -117.004,
            -101.686, -106.488, -107.394, -89.617, -110.966,
            # Historical
            -90.058, -76.779, -70.668, -71.208, -81.314,
            -105.944, -73.756, -69.719, -89.916, -88.040
        ],
        'latitude': [
            # US
            40.713, 34.052, 41.878, 29.760, 33.448,
            39.952, 29.424, 32.716, 32.778, 37.339,
            30.267, 30.332, 32.755, 39.961, 35.227,
            39.768, 37.774, 47.606, 39.739, 42.361,
            # Canada
            43.651, 45.501, 49.283, 51.045, 53.535,
            45.421, 49.896, 46.813, 43.256, 43.452,
            # Mexico
            19.432, 20.677, 25.677, 19.043, 32.514,
            21.122, 31.690, 24.799, 20.975, 29.088,
            # Historical
            38.657, 37.321, 41.957, 46.813, 29.898,
            35.691, 42.652, 48.143, 38.058, 30.697
        ],
        # Add historical periods (for filtering)
        'historical_period': [
            # US Cities (Modern)
            *( ['modern'] * 20 ),
            # Canadian Cities (Modern)
            *( ['modern'] * 10 ),
            # Mexican Cities (Modern)
            *( ['modern'] * 10 ),
            # Historical
            'pre-colonial', 'colonial', 'colonial', 'colonial', 'colonial',
            'colonial', 'colonial', 'colonial', 'colonial', 'colonial'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    # Cache the generated fallback data
    _cache_gdf(gdf.copy(), cache_filepath)
    
    return gdf

def load_roads(region: Optional[str] = 'north-america', include_historical: bool = True) -> gpd.GeoDataFrame:
    """
    Load road data for North America, optionally including historical roads
    
    Args:
        region: Region to load data for (default to 'north-america')
        include_historical: Whether to include historical roads
        
    Returns:
        GeoDataFrame with roads
    """
    # Load modern roads from OSM
    modern_roads = load_modern_roads(region)
    
    # If historical roads not needed, return just modern roads
    if not include_historical:
        return modern_roads
    
    # Load historical roads
    historical_roads = load_colonial_roads()
    
    # Combine the dataframes if both have data
    if not modern_roads.empty and not historical_roads.empty:
        # Ensure compatible schemas
        if 'road_type' not in historical_roads.columns:
            historical_roads['road_type'] = 'historical'
        if 'road_type' not in modern_roads.columns and 'highway' in modern_roads.columns:
            modern_roads['road_type'] = modern_roads['highway']
        
        # Add period identifier
        historical_roads['period'] = 'historical'
        modern_roads['period'] = 'modern'
        
        # Combine
        combined = pd.concat([modern_roads, historical_roads])
        return combined
    elif not historical_roads.empty:
        return historical_roads
    else:
        return modern_roads

def load_modern_roads(region: Optional[str] = 'north-america') -> gpd.GeoDataFrame:
    """
    Load modern road data from OSM (or fallback), using cache.
    
    Args:
        region: Region to load data for
        
    Returns:
        GeoDataFrame with modern roads
    """
    # Define cache key/path for fallback data
    fallback_cache_key = f"fallback_modern_roads_{region}.geojson"
    cache_filepath = _get_cache_path(os.path.join(CACHE_DIR, fallback_cache_key), params={'region': region})

    # Check cache first
    if os.path.exists(cache_filepath):
        try:
            logger.info(f"Loading fallback modern roads for region '{region}' from cache: {cache_filepath}")
            gdf = feather.read_feather(cache_filepath)
            if gdf.crs is None: gdf.crs = "EPSG:4326"
            if 'geometry' in gdf.columns:
                gdf = gdf.set_geometry('geometry')
            elif not gdf.empty:
                 geom_col = gdf.select_dtypes(include='geometry').columns
                 if len(geom_col) == 1:
                     gdf = gdf.set_geometry(geom_col[0])
                 else:
                     logger.warning(f"Could not determine geometry column in cached fallback roads: {cache_filepath}")
            return gdf
        except Exception as e:
            logger.error(f"Error loading fallback modern roads from cache {cache_filepath}: {e}. Regenerating.")
            
    # --- Production code would attempt PBF processing here ---
    # Similar logic as in load_settlements...
    # --- End of hypothetical PBF processing ---

    # If PBF processing skipped/failed, generate fallback
    logger.warning(f"OSM PBF processing for roads not implemented/failed for region '{region}'. Generating and caching fallback data.")
    
    # Major North American highways/roads sample
    roads_data = {
        'osm_id': list(range(1, 21)),
        'name': [
            'Interstate 95', 'Interstate 80', 'Interstate 10', 'Interstate 5', 
            'Trans-Canada Highway', 'Mexico Federal Highway 85',
            'US Route 1', 'US Route 66', 'US Route 101', 'Mexico Federal Highway 15',
            'I-90', 'I-70', 'I-40', 'I-35', 'Highway 401',
            'Interstate 25', 'US Route 2', 'Mexico Federal Highway 45', 'Canada Highway 16', 'Pan-American Highway'
        ],
        'highway': ['motorway'] * 5 + ['primary'] * 5 + ['motorway'] * 5 + ['primary'] * 5, # More varied types
        'road_type': ['motorway'] * 5 + ['primary'] * 5 + ['motorway'] * 5 + ['primary'] * 5, # Consistent column
        'period': ['modern'] * 20, # Add period
        'wkt': [
            # Simplified representations of major roads
            'LINESTRING(-67.0 45.0, -74.0 40.7, -80.0 25.8)',  # I-95
            'LINESTRING(-122.4 37.8, -111.9 41.3, -104.9 41.1, -95.7 41.2, -74.0 40.7)',  # I-80
            'LINESTRING(-118.4 34.0, -106.5 31.8, -98.5 29.4, -90.1 29.9, -84.3 30.4, -80.1 25.8)',  # I-10
            'LINESTRING(-122.4 37.8, -122.3 47.6, -123.1 49.3)',  # I-5
            'LINESTRING(-67.0 45.0, -76.5 44.2, -79.4 43.7, -97.1 49.9, -114.1 51.0, -123.1 49.3)',  # Trans-Canada
            'LINESTRING(-99.1 19.4, -100.3 25.7, -98.3 26.1)',  # Mexico 85
            'LINESTRING(-80.1 25.8, -82.5 28.1, -80.8 35.2, -77.0 38.9, -74.0 40.7, -71.1 42.4, -70.3 43.7, -67.0 45.0)',  # US-1
            'LINESTRING(-118.2 34.1, -117.3 34.1, -112.1 35.2, -105.9 35.1, -97.5 35.5, -90.2 38.6, -87.6 41.9)', # US-66 (Adjusted endpoint)
            'LINESTRING(-118.2 34.1, -122.4 37.8, -122.3 47.6)',  # US-101
            'LINESTRING(-99.1 19.4, -105.3 20.7, -103.3 20.7, -111.0 29.1, -110.3 32.5)',  # Mexico 15
            'LINESTRING(-71.1 42.4, -87.6 41.9, -97.5 44.9, -111.9 41.3, -122.3 47.6)',  # I-90 (Adjusted start)
            'LINESTRING(-75.1 39.9, -87.6 39.8, -92.2 38.6, -98.5 39.1, -104.9 39.7, -111.9 39.3)',  # I-70 (Adjusted ends)
            'LINESTRING(-80.8 35.2, -90.0 35.1, -97.5 35.5, -105.9 35.1, -117.3 34.1)',  # I-40 (Adjusted start)
            'LINESTRING(-97.1 49.9, -97.5 35.5, -97.3 30.3, -98.5 29.4)',  # I-35 (Adjusted ends)
            'LINESTRING(-83.0 42.3, -79.4 43.7, -76.5 44.2, -73.6 45.5)',  # Highway 401
            'LINESTRING(-105.9 35.1, -104.9 39.7, -106.0 40.6, -104.7 45.0)',  # I-25 (Adjusted end)
            'LINESTRING(-71.1 42.4, -79.0 43.0, -90.5 46.8, -97.1 49.9, -122.3 47.6)',  # US-2 (Adjusted start)
            'LINESTRING(-99.1 19.4, -101.7 21.1, -101.3 22.1, -101.7 23.2, -102.3 26.9, -106.5 31.8)',  # Mexico 45
            'LINESTRING(-123.1 49.3, -120.0 50.0, -113.3 53.5, -101.0 52.0, -97.1 49.9)',  # Canada 16 (Adjusted point)
            'LINESTRING(-80.1 25.8, -84.3 30.4, -90.1 29.9, -97.5 35.5, -106.5 31.8, -99.1 19.4, -75.0 0.0)'  # Pan-American (Simplified South America part)
        ]
    }
    
    df = pd.DataFrame(roads_data)
    try:
        # Ensure shapely is available
        from shapely import wkt
        gdf = gpd.GeoDataFrame(
            df,
            geometry=df['wkt'].apply(wkt.loads),
            crs="EPSG:4326"
        )
    except ImportError:
         logger.error("Shapely required for fallback modern roads data.")
         return gpd.GeoDataFrame(columns=['geometry', 'road_type', 'period'], geometry='geometry', crs="EPSG:4326")
    except Exception as fallback_err:
         logger.error(f"Error creating fallback modern roads GDF: {fallback_err}")
         return gpd.GeoDataFrame(columns=['geometry', 'road_type', 'period'], geometry='geometry', crs="EPSG:4326")

    # Cache the generated data
    _cache_gdf(gdf.copy(), cache_filepath)

    return gdf

def load_dem_data(bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[Dict]:
    """
    Load Digital Elevation Model data for an area
    
    Args:
        bounds: Optional bounding box as (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Dictionary with DEM data or None if failed
    """
    # This would ideally use USGS or similar services to get elevation data
    # For now, just return a simplified representation
    logger.warning("Using simplified DEM data - production would use proper DEM services")
    
    # Simple representation of a DEM
    dem_data = {
        'resolution': 30,  # 30m resolution
        'crs': 'EPSG:4326',
        'bounds': bounds or (-180, -90, 180, 90),
        'min_elevation': 0,
        'max_elevation': 8848,  # Height of Mount Everest
        'mean_elevation': 840,
        'data_source': 'Placeholder - would use SRTM or similar in production'
    }
    
    return dem_data