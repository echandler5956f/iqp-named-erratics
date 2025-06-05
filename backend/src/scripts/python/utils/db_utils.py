# backend/src/scripts/python/utils/db_utils.py
import os
import json
import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import pathlib
import logging

logger = logging.getLogger(__name__)

def get_db_connection(autocommit=False):
    """Establish a connection to the PostgreSQL database using environment variables from the root .env file."""
    current_script_path = pathlib.Path(__file__).resolve()
    project_root = current_script_path.parent.parent.parent.parent.parent # utils -> python -> scripts -> src -> backend -> project_root
    dotenv_path = project_root / '.env'

    if dotenv_path.exists():
        logger.debug(f"Loading environment variables for DB connection from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logger.warning(f"Root .env file not found at {dotenv_path}. DB connection might fail or use system env vars.")

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
            raise ValueError(f"DB config incomplete. Missing: {missing_vars}")
        
        conn = psycopg2.connect(**db_config)
        if autocommit:
            conn.autocommit = True
        logger.info(f"DB connection to {db_config['database']} successful.")
        return conn
    except ValueError as ve:
        logger.error(f"DB configuration error: {ve}")
        raise
    except (Exception, psycopg2.Error) as error:
        logger.error(f"Error connecting to PostgreSQL: {error}")
        raise

def load_all_erratics_gdf() -> gpd.GeoDataFrame:
    """Load all erratics and their analysis data from the database into a GeoDataFrame."""
    logger.info("Loading all erratics with analysis data into GeoDataFrame...")
    conn = None
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            e.*, 
            ea.usage_type, ea.cultural_significance_score, ea.has_inscriptions, 
            ea.accessibility_score, ea.size_category, ea.nearest_water_body_dist,
            ea.nearest_settlement_dist, 
            ea.nearest_road_dist, 
            ea.nearest_native_territory_dist, ea.elevation_category,
            ea.geological_type, ea.estimated_displacement_dist,
            ea.ruggedness_tri, ea.terrain_landform, ea.terrain_slope_position,
            ea.vector_embedding,
            ea.nearest_natd_road_dist, 
            ea.nearest_forest_trail_dist 
        FROM "Erratics" e
        LEFT JOIN "ErraticAnalyses" ea ON e.id = ea."erraticId";
        """
        gdf = gpd.read_postgis(query, conn, geom_col='location', crs='EPSG:4326')
        logger.info(f"Loaded {len(gdf)} erratics with their analysis data.")
        if 'vector_embedding' in gdf.columns and gdf['vector_embedding'].dtype == 'object':
            logger.debug("Processing string-based vector_embedding column...")
            def parse_embedding_str(embedding_str):
                if pd.isna(embedding_str) or not isinstance(embedding_str, str): return None
                try: return [float(x) for x in embedding_str.strip('[]').split(',')]
                except ValueError: return None
            gdf['vector_embedding'] = gdf['vector_embedding'].apply(parse_embedding_str)
        return gdf
    except Exception as e:
        logger.error(f"Error loading all erratics into GeoDataFrame: {e}", exc_info=True)
        return gpd.GeoDataFrame() # Return empty GDF on error
    finally:
        if conn: conn.close()

def load_erratic_details_by_id(erratic_id: int) -> Optional[Dict]:
    """Load detailed information for a single erratic by its ID."""
    logger.info(f"Loading details for erratic ID: {erratic_id}")
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT e.id, e.name, ST_AsGeoJSON(e.location)::json->'coordinates'->>0 AS longitude, 
                       ST_AsGeoJSON(e.location)::json->'coordinates'->>1 AS latitude, 
                       e.elevation, e.size_meters, e.rock_type, e.estimated_age, e.discovery_date, 
                       e.description, e.cultural_significance, e.historical_notes, e.image_url
                FROM "Erratics" e
                WHERE e.id = %s;
            """, (erratic_id,))
            row = cursor.fetchone()
            if row is not None:
                # Convert numeric fields from string if necessary (RealDictCursor might return them as strings)
                for key in ['longitude', 'latitude', 'elevation', 'size_meters']:
                    if row.get(key) is not None:
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert {key}='{row[key]}' to float for erratic ID {erratic_id}")
                            # Decide on handling: keep as is, set to None, or raise error
                            # For now, let it pass, might result in string type if conversion fails
                            pass 
                logger.info(f"Details loaded for erratic ID: {erratic_id}")
                return dict(row)
            else:
                logger.warn(f"No erratic found with ID: {erratic_id}")
                return None
    except Exception as e:
        logger.error(f"Error loading details for erratic ID {erratic_id}: {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close()

def update_erratic_analysis_results(erratic_id: int, analysis_data: Dict[str, Any]) -> bool:
    """Update or insert analysis results for an erratic in the ErraticAnalyses table."""
    logger.info(f"Updating analysis results for erratic ID: {erratic_id}")
    conn = None
    
    # Define fields allowed in ErraticAnalyses table to prevent SQL injection or errors
    # This list should align with the columns in your ErraticAnalyses model (excluding primary/foreign keys and timestamps handled by DB)
    allowed_fields = [
        'usage_type', 'cultural_significance_score', 'has_inscriptions',
        'accessibility_score', 'size_category', 'nearest_water_body_dist',
        'nearest_settlement_dist', 
        'nearest_road_dist', 
        'nearest_native_territory_dist', 'elevation_category',
        'geological_type', 'estimated_displacement_dist',
        'vector_embedding', 
        'ruggedness_tri', 'terrain_landform', 'terrain_slope_position',
        'nearest_natd_road_dist', 
        'nearest_forest_trail_dist' 
    ]

    # Filter and prepare data
    db_data = {k: v for k, v in analysis_data.items() if k in allowed_fields and v is not None}

    if not db_data:
        logger.warn(f"No valid or updatable analysis data provided for erratic ID: {erratic_id}", extra={'original_data': analysis_data})
        return False

    try:
        conn = get_db_connection(autocommit=False) # Use autocommit=False for explicit transaction control
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Check if record exists
            cursor.execute('SELECT "erraticId" FROM "ErraticAnalyses" WHERE "erraticId" = %s;', (erratic_id,))
            exists_row = cursor.fetchone()

            # pgvector expects embeddings as a string like '[1,2,3]' or list if adapter is registered
            if 'vector_embedding' in db_data and isinstance(db_data['vector_embedding'], list):
                pass 

            if exists_row is not None:
                set_clause = ", ".join([f'\"{col}\" = %s' for col in db_data.keys()])
                sql = f'UPDATE "ErraticAnalyses" SET {set_clause}, "updatedAt" = NOW() WHERE "erraticId" = %s;'
                values = list(db_data.values()) + [erratic_id]
                logger.debug(f"Executing UPDATE for ErraticAnalyses ID {erratic_id}", extra={'sql': sql, 'values_count': len(values)})
            else:
                db_data['erraticId'] = erratic_id # Ensure erraticId is included for INSERT
                columns = '\", \"'.join(db_data.keys())
                placeholders = ", ".join(['%s'] * len(db_data))
                sql = f'INSERT INTO "ErraticAnalyses" (\"{columns}\", "createdAt", "updatedAt") VALUES ({placeholders}, NOW(), NOW());'
                values = list(db_data.values())
                logger.debug(f"Executing INSERT for ErraticAnalyses ID {erratic_id}", extra={'sql': sql, 'values_count': len(values)})
            
            cursor.execute(sql, values)
            conn.commit()
            logger.info(f"Successfully {'updated' if exists_row is not None else 'inserted'} analysis data for erratic ID: {erratic_id}")
            return True
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"Error updating/inserting analysis data for erratic ID {erratic_id}: {e}", extra={'original_data': analysis_data}, exc_info=True)
        return False
    finally:
        if conn: conn.close()

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing db_utils.py...")
    
    # Ensure your .env file is in the project root for this test to work
    # Test DB Connection
    conn_test = get_db_connection()
    if conn_test:
        print("DB Connection Test: SUCCESS")
        conn_test.close()
    else:
        print("DB Connection Test: FAILED")

    # Test loading all erratics
    # print("\nLoading all erratics GDF...")
    # all_erratics_gdf = load_all_erratics_gdf()
    # if not all_erratics_gdf.empty:
    #     print(f"Loaded {len(all_erratics_gdf)} erratics. First 5 IDs: {all_erratics_gdf['id'].head().tolist()}")
    #     print(all_erratics_gdf.head())
    # else:
    #     print("Failed to load erratics or table is empty.")

    # Test loading a single erratic (replace 1 with an existing ID in your DB)
    # test_erratic_id = 1 
    # print(f"\nLoading details for erratic ID: {test_erratic_id}...")
    # erratic_details = load_erratic_details_by_id(test_erratic_id)
    # if erratic_details:
    #     print("Erratic Details:", json.dumps(erratic_details, indent=2))
    # else:
    #     print(f"Erratic ID {test_erratic_id} not found or error loading.")

    # Test updating analysis data (replace 1 with an existing ID)
    # test_update_id = 1
    # sample_analysis_data = {
    #     'cultural_significance_score': 8,
    #     'accessibility_score': 4,
    #     'size_category': 'large',
    #     'nearest_water_body_dist': 150.75,
    #     'vector_embedding': [0.1] * 384 # Example embedding
    # }
    # print(f"\nUpdating analysis for erratic ID: {test_update_id}...")
    # update_success = update_erratic_analysis_results(test_update_id, sample_analysis_data)
    # print(f"Update {'SUCCESSFUL' if update_success else 'FAILED'}")
    pass 