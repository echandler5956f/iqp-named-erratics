"""
Compatibility layer for the old data_loader API.

This module provides the same functions as the old monolithic data_loader.py
but uses the new modular pipeline system underneath.
"""

import os
import psycopg2
import geopandas as gpd
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, Any
from psycopg2.extras import RealDictCursor

from . import load_data, registry
from .sources import DataSource

# Import data sources to ensure they're registered
from . import data_sources


logger = logging.getLogger(__name__)


# Database functions (these remain largely unchanged)
def get_db_connection():
    """Establish a connection to the PostgreSQL database"""
    from dotenv import load_dotenv, find_dotenv
    
    dotenv_path = find_dotenv()
    if dotenv_path:
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    
    try:
        db_config = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "database": os.environ.get("DB_NAME"),
            "user": os.environ.get("DB_USER"),
            "password": os.environ.get("DB_PASSWORD"),
            "port": int(os.environ.get("DB_PORT", 5432))
        }
        
        missing_vars = [k for k, v in db_config.items() if v is None and k in ['database', 'user', 'password']]
        if missing_vars:
            raise ValueError(f"Database configuration incomplete. Missing: {', '.join(missing_vars)}")
        
        conn = psycopg2.connect(**db_config)
        logger.info("Database connection successful.")
        return conn
    except Exception as error:
        logger.error(f"Error connecting to PostgreSQL database: {error}")
        return None


def load_erratics() -> gpd.GeoDataFrame:
    """Load all erratics from the database into a GeoDataFrame"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to establish database connection for loading erratics.")
        return gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs="EPSG:4326")
    
    try:
        query = """
        SELECT 
            e.id, e.name, 
            e.location,
            e.elevation, e.size_meters, e.rock_type, 
            e.estimated_age, e.discovery_date, e.description,
            e.cultural_significance, e.historical_notes,
            ea.usage_type, ea.cultural_significance_score,
            ea.has_inscriptions, ea.accessibility_score,
            ea.size_category, ea.nearest_water_body_dist,
            ea.nearest_settlement_dist, ea.elevation_category,
            ea.vector_embedding
        FROM 
            "Erratics" e
        LEFT JOIN 
            "ErraticAnalyses" ea ON e.id = ea."erraticId"
        """
        
        gdf = gpd.read_postgis(query, conn, geom_col='location', crs='EPSG:4326')
        
        # Process vector_embedding column if it exists
        if 'vector_embedding' in gdf.columns and gdf['vector_embedding'].dtype == 'object':
            logger.info("Processing string-based vector_embedding column into lists of floats...")
            
            def parse_embedding_str(embedding_str):
                if pd.isna(embedding_str) or not isinstance(embedding_str, str):
                    return None
                try:
                    return [float(x) for x in embedding_str.strip('[]').split(',')]
                except ValueError:
                    return None
            
            gdf['vector_embedding'] = gdf['vector_embedding'].apply(parse_embedding_str)
        
        return gdf
    except Exception as e:
        logger.error(f"Error loading erratics: {e}")
        return gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs="EPSG:4326")
    finally:
        if conn:
            conn.close()


def load_erratic_by_id(erratic_id: int) -> Optional[Dict]:
    """Load a single erratic by ID from the database"""
    conn = get_db_connection()
    if not conn:
        # Fallback to mock data
        return {
            'id': erratic_id,
            'name': f'Test Erratic {erratic_id}',
            'longitude': -73.968285,
            'latitude': 40.785091,
            'elevation': 100.0,
            'description': 'A test erratic for development purposes.',
            'cultural_significance': 'None, this is test data.',
            'historical_notes': 'Created for testing.'
        }
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT e.id, e.name, ST_X(e.location::geometry) as longitude, 
                       ST_Y(e.location::geometry) as latitude, e.elevation, 
                       e.description, e.cultural_significance, e.historical_notes
                FROM "Erratics" e
                WHERE e.id = %s
            """, [erratic_id])
            
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error loading erratic by ID {erratic_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()


def update_erratic_analysis_data(erratic_id: int, data: Dict) -> bool:
    """Update or insert analysis results for an erratic"""
    conn = get_db_connection()
    if not conn:
        logger.error(f"Failed to get DB connection for updating ErraticAnalyses")
        return False
    
    try:
        with conn.cursor() as cursor:
            # Known fields from ErraticAnalyses table
            known_fields = [
                'usage_type', 'cultural_significance_score', 'has_inscriptions',
                'accessibility_score', 'size_category', 'nearest_water_body_dist',
                'nearest_settlement_dist', 'nearest_colonial_settlement_dist',
                'nearest_road_dist', 'nearest_colonial_road_dist',
                'nearest_native_territory_dist', 'elevation_category',
                'geological_type', 'estimated_displacement_dist',
                'vector_embedding', 'ruggedness_tri', 'terrain_landform', 
                'terrain_slope_position'
            ]
            
            update_values = {k: v for k, v in data.items() 
                           if k in known_fields and v is not None}
            
            if not update_values:
                logger.info(f"No valid analysis data to update for erratic_id: {erratic_id}")
                return False
            
            update_values['erraticId'] = erratic_id
            
            # Check if record exists
            cursor.execute('SELECT 1 FROM "ErraticAnalyses" WHERE "erraticId" = %s', [erratic_id])
            record_exists = cursor.fetchone() is not None
            
            if record_exists:
                # Update existing record
                set_clauses = [f'"{col}" = %s' for col in update_values if col != 'erraticId']
                set_clauses.append('"updatedAt" = NOW()')
                
                update_sql = f"""
                UPDATE "ErraticAnalyses" 
                SET {', '.join(set_clauses)}
                WHERE "erraticId" = %s
                """
                
                query_values = [v for k, v in update_values.items() if k != 'erraticId']
                query_values.append(erratic_id)
                
                cursor.execute(update_sql, query_values)
            else:
                # Insert new record
                columns = [f'"{col}"' for col in update_values if col not in ['createdAt', 'updatedAt']]
                placeholders = ['%s'] * len(columns)
                query_values = [v for k, v in update_values.items() if k not in ['createdAt', 'updatedAt']]
                
                columns.extend(['"createdAt"', '"updatedAt"'])
                placeholders.extend(['NOW()', 'NOW()'])
                
                insert_sql = f"""
                INSERT INTO "ErraticAnalyses" ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                """
                
                cursor.execute(insert_sql, query_values)
            
            conn.commit()
            logger.info(f"Successfully updated analysis data for erratic_id: {erratic_id}")
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
    """Save JSON data to a file"""
    try:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to file: {e}")
        return False


def file_to_json(input_file: str) -> Dict:
    """Read JSON data from a file"""
    try:
        import json
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading from {input_file}: {e}")
        return {}


# Data loading functions that use the new pipeline
def load_hydro_features(feature_type: str = 'rivers', region: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load hydrological features"""
    source_map = {
        'rivers': 'hydrosheds_rivers',
        'lakes': 'hydrosheds_lakes',
        'basins': 'hydrosheds_basins'
    }
    
    source_name = source_map.get(feature_type.lower())
    if not source_name:
        logger.error(f"Unsupported hydrological feature type: {feature_type}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
    
    try:
        return load_data(source_name)
    except Exception as e:
        logger.error(f"Error loading {feature_type}: {e}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")


def load_native_territories() -> gpd.GeoDataFrame:
    """Load Native American territory data"""
    try:
        return load_data('native_territories')
    except Exception as e:
        logger.error(f"Error loading native territories: {e}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")


def load_colonial_settlements() -> gpd.GeoDataFrame:
    """Load historical colonial settlement data"""
    try:
        # For now, return a simple fallback until we implement proper processing
        # of the NHGIS data to extract settlements
        return _create_fallback_colonial_settlements()
    except Exception as e:
        logger.error(f"Error loading colonial settlements: {e}")
        return _create_fallback_colonial_settlements()


def load_colonial_roads() -> gpd.GeoDataFrame:
    """Load historical colonial road data"""
    try:
        return load_data('colonial_roads')
    except Exception as e:
        logger.error(f"Error loading colonial roads: {e}")
        return _create_fallback_colonial_roads()


def load_settlements(region: Optional[str] = 'north-america') -> gpd.GeoDataFrame:
    """Load settlement data for a specified region"""
    source_map = {
        'us': 'osm_us',
        'canada': 'osm_canada',
        'north-america': 'osm_north_america'
    }
    
    source_name = source_map.get(region, 'osm_north_america')
    
    try:
        return load_data(source_name)
    except Exception as e:
        logger.error(f"Error loading settlements for {region}: {e}")
        return _create_fallback_settlements(region)


def load_roads(region: Optional[str] = 'north-america', include_historical: bool = True) -> gpd.GeoDataFrame:
    """Load road data for North America"""
    # For modern roads, we use OSM data with road filters
    source_map = {
        'us': 'osm_us',
        'canada': 'osm_canada', 
        'north-america': 'osm_north_america'
    }
    
    source_name = source_map.get(region, 'osm_north_america')
    
    try:
        # Need to adjust parameters for road extraction
        modern_roads = load_data(source_name, 
                               layer_type='lines',
                               sql_filter='highway IS NOT NULL')
        
        if not include_historical:
            return modern_roads
            
        # Load historical roads
        historical_roads = load_colonial_roads()
        
        # Combine if both have data
        if not modern_roads.empty and not historical_roads.empty:
            historical_roads['period'] = 'historical'
            modern_roads['period'] = 'modern'
            return pd.concat([modern_roads, historical_roads])
        elif not historical_roads.empty:
            return historical_roads
        else:
            return modern_roads
            
    except Exception as e:
        logger.error(f"Error loading roads: {e}")
        return _create_fallback_modern_roads(region)


def load_modern_roads(region: Optional[str] = 'north-america') -> gpd.GeoDataFrame:
    """Load modern road data"""
    return load_roads(region, include_historical=False)


def download_and_extract_data(data_key: str, region: Optional[str] = None, 
                            force_download: bool = False, point: Optional[Any] = None) -> Optional[str]:
    """Download and extract/prepare a GIS dataset if not already available"""
    # This function now just ensures the data is downloaded and returns the path
    try:
        # Map old data keys to new source names if needed
        key_map = {
            'elevation_srtm90_csi': 'elevation_srtm90_csi',
            'elevation_dem_na': 'elevation_dem_na',
            # Add more mappings as needed
        }
        
        source_name = key_map.get(data_key, data_key)
        
        # For elevation data with specific points, we'd need custom handling
        # For now, just ensure the data is available
        data = load_data(source_name, force_reload=force_download)
        
        # Return the directory where data is stored
        # This is a simplification - the real implementation would return
        # the actual file path
        return os.path.dirname(os.path.abspath(__file__))
        
    except Exception as e:
        logger.error(f"Error downloading {data_key}: {e}")
        return None


def load_dem_data(point: Optional[Any] = None) -> Optional[str]:
    """Load DEM data for a point"""
    # Simplified implementation - would need proper tile handling
    try:
        if point and hasattr(point, 'latitude') and -60 <= point.latitude <= 60:
            # Use SRTM for mid-latitudes
            data = load_data('elevation_srtm90_csi')
        else:
            # Use alternate DEM for high latitudes
            data = load_data('elevation_dem_na')
        
        # Return a path to the DEM file
        # This is simplified - real implementation would return actual file path
        return "path/to/dem.tif"
    except Exception as e:
        logger.error(f"Error loading DEM data: {e}")
        return None


# Fallback data generation functions
def _create_fallback_colonial_settlements() -> gpd.GeoDataFrame:
    """Create fallback colonial settlement data"""
    data = {
        'name': [
            'Jamestown', 'Plymouth', 'Quebec City', 'St. Augustine', 
            'New Amsterdam (New York)', 'Boston', 'Montreal', 'Philadelphia'
        ],
        'founded': [1607, 1620, 1608, 1565, 1624, 1630, 1642, 1682],
        'colony': [
            'Virginia', 'Massachusetts', 'New France', 'Spanish Florida',
            'New Netherland', 'Massachusetts', 'New France', 'Pennsylvania'
        ],
        'longitude': [-76.779, -70.668, -71.208, -81.314, -74.010, -71.059, -73.588, -75.165],
        'latitude': [37.321, 41.957, 46.813, 29.898, 40.714, 42.361, 45.501, 39.952]
    }
    
    df = pd.DataFrame(data)
    return gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )


def _create_fallback_colonial_roads() -> gpd.GeoDataFrame:
    """Create fallback colonial roads data"""
    from shapely import wkt
    
    roads_data = {
        'name': ['Boston Post Road', 'King\'s Highway', 'Wilderness Road'],
        'year': [1673, 1650, 1775],
        'wkt': [
            'LINESTRING(-71.06 42.36, -72.54 42.10, -73.76 42.65, -74.01 40.71)',
            'LINESTRING(-71.06 42.36, -74.01 40.71, -75.16 39.95, -77.03 38.90)',
            'LINESTRING(-78.64 38.03, -81.63 38.35, -84.50 38.04, -85.76 38.25)'
        ]
    }
    
    df = pd.DataFrame(roads_data)
    return gpd.GeoDataFrame(
        df,
        geometry=df['wkt'].apply(wkt.loads),
        crs="EPSG:4326"
    )


def _create_fallback_settlements(region: str) -> gpd.GeoDataFrame:
    """Create fallback settlement data"""
    sample_data = {
        'osm_id': list(range(1, 11)),
        'name': [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'
        ],
        'place_type': ['city'] * 10,
        'longitude': [-74.006, -118.243, -87.630, -95.369, -112.074,
                     -75.165, -98.491, -117.161, -96.797, -121.895],
        'latitude': [40.713, 34.052, 41.878, 29.760, 33.448,
                    39.952, 29.424, 32.716, 32.778, 37.339]
    }
    
    df = pd.DataFrame(sample_data)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )


def _create_fallback_modern_roads(region: str) -> gpd.GeoDataFrame:
    """Create fallback modern roads data"""
    from shapely import wkt
    
    roads_data = {
        'osm_id': list(range(1, 6)),
        'name': ['Interstate 95', 'Interstate 80', 'Interstate 10', 
                'Interstate 5', 'Trans-Canada Highway'],
        'highway': ['motorway'] * 5,
        'road_type': ['motorway'] * 5,
        'period': ['modern'] * 5,
        'wkt': [
            'LINESTRING(-67.0 45.0, -74.0 40.7, -80.0 25.8)',
            'LINESTRING(-122.4 37.8, -111.9 41.3, -104.9 41.1)',
            'LINESTRING(-118.4 34.0, -106.5 31.8, -98.5 29.4)',
            'LINESTRING(-122.4 37.8, -122.3 47.6, -123.1 49.3)',
            'LINESTRING(-67.0 45.0, -76.5 44.2, -79.4 43.7)'
        ]
    }
    
    df = pd.DataFrame(roads_data)
    return gpd.GeoDataFrame(
        df,
        geometry=df['wkt'].apply(wkt.loads),
        crs="EPSG:4326"
    ) 