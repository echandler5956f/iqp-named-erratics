"""
Cache management for processed data.

Provides caching functionality using Feather files for fast serialization
of GeoDataFrames.
"""

import os
import hashlib
import json
import logging
import geopandas as gpd
import pyarrow.feather as feather
from typing import Optional, Dict, Any
from pathlib import Path

from shapely.wkb import dumps as wkb_dumps, loads as wkb_loads


logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cached data using Feather format"""
    
    def __init__(self, cache_dir: str = None):
        """Initialize cache manager with cache directory"""
        if cache_dir is None:
            # Default to data/cache relative to this module
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'cache'
            )
        
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_key(self, source_name: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key based on source name and parameters"""
        hasher = hashlib.md5()
        hasher.update(source_name.encode())
        
        if params:
            # Sort parameters for consistent hashing
            params_str = json.dumps(params, sort_keys=True)
            hasher.update(params_str.encode())
            
        return hasher.hexdigest()
    
    def _get_cache_path(self, source_name: str, params: Optional[Dict] = None) -> str:
        """Get the full path for a cache file"""
        cache_key = self._get_cache_key(source_name, params)
        return os.path.join(self.cache_dir, f"{source_name}_{cache_key}.feather")
    
    def is_cached(self, source_name: str, params: Optional[Dict] = None, 
                  source_mtime: Optional[float] = None) -> bool:
        """Check if data is cached and up-to-date"""
        cache_path = self._get_cache_path(source_name, params)
        
        if not os.path.exists(cache_path):
            return False
            
        # Check if cache is newer than source
        if source_mtime is not None:
            cache_mtime = os.path.getmtime(cache_path)
            if cache_mtime < source_mtime:
                logger.info(f"Cache for {source_name} is older than source")
                return False
                
        return True
    
    def save(self, gdf: gpd.GeoDataFrame, source_name: str, 
             params: Optional[Dict] = None) -> bool:
        """Save GeoDataFrame to cache"""
        try:
            cache_path = self._get_cache_path(source_name, params)
            
            # Create a copy to avoid modifying the original
            gdf_copy = gdf.copy()
            
            # Save CRS information
            crs_path = cache_path + ".crs"
            if gdf_copy.crs:
                with open(crs_path, 'w') as f:
                    f.write(str(gdf_copy.crs))
            
            # Convert geometry to WKB for storage
            if 'geometry' in gdf_copy.columns:
                geometry_col = gdf_copy._geometry_column_name
                
                # Save geometry column name
                with open(cache_path + ".geom_col", 'w') as f:
                    f.write(geometry_col)
                
                # Handle WKB conversion
                if gdf_copy.empty or gdf_copy[geometry_col].apply(lambda geom: geom is None or geom.is_empty).all():
                    # If GDF is empty or all geometries are empty/None, create an empty _wkb column
                    gdf_copy['_wkb'] = None
                else:
                    # Convert actual geometries to WKB
                    gdf_copy['_wkb'] = gdf_copy[geometry_col].apply(
                        lambda geom: wkb_dumps(geom) if geom and not geom.is_empty else None
                    )
                
                # Drop the original geometry column
                gdf_copy = gdf_copy.drop(columns=[geometry_col])
            
            # Ensure string column names
            gdf_copy.columns = gdf_copy.columns.astype(str)
            
            # Write to feather
            feather.write_feather(gdf_copy, cache_path)
            logger.info(f"Cached {source_name} to {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching {source_name}: {e}")
            return False
    
    def load(self, source_name: str, params: Optional[Dict] = None) -> Optional[gpd.GeoDataFrame]:
        """Load GeoDataFrame from cache"""
        try:
            cache_path = self._get_cache_path(source_name, params)
            
            if not os.path.exists(cache_path):
                return None
                
            logger.info(f"Loading {source_name} from cache: {cache_path}")
            df = feather.read_feather(cache_path)
            
            # Restore CRS
            crs = None
            crs_path = cache_path + ".crs"
            if os.path.exists(crs_path):
                with open(crs_path, 'r') as f:
                    crs = f.read().strip()
            
            # Check for WKB geometry data
            if '_wkb' in df.columns:
                # Get geometry column name
                geom_col = 'geometry'
                geom_col_path = cache_path + ".geom_col"
                if os.path.exists(geom_col_path):
                    with open(geom_col_path, 'r') as f:
                        geom_col = f.read().strip()
                
                # Convert WKB back to geometry
                df[geom_col] = df['_wkb'].apply(
                    lambda wkb: wkb_loads(wkb) if wkb else None
                )
                df = df.drop(columns=['_wkb'])
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs=crs)
            else:
                gdf = gpd.GeoDataFrame(df, crs=crs)
                
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading from cache {source_name}: {e}")
            return None
    
    def clear(self, source_name: Optional[str] = None) -> None:
        """Clear cache for a specific source or all caches"""
        if source_name:
            # Clear specific source
            pattern = f"{source_name}_*.feather"
            for file in Path(self.cache_dir).glob(pattern):
                os.remove(file)
                # Also remove associated metadata files
                for ext in ['.crs', '.geom_col']:
                    meta_file = str(file) + ext
                    if os.path.exists(meta_file):
                        os.remove(meta_file)
            logger.info(f"Cleared cache for {source_name}")
        else:
            # Clear all caches
            for file in Path(self.cache_dir).glob("*.feather"):
                os.remove(file)
            for file in Path(self.cache_dir).glob("*.crs"):
                os.remove(file)
            for file in Path(self.cache_dir).glob("*.geom_col"):
                os.remove(file)
            logger.info("Cleared all caches") 