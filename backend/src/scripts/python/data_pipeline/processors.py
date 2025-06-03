"""
Data processors for different file formats.

Each processor is responsible for transforming raw data files into
the format needed by the application (typically GeoDataFrames).
"""

import os
import zipfile
import geopandas as gpd
import pandas as pd
import subprocess
import logging
from typing import Optional, Any
from abc import ABC, abstractmethod

from .sources import DataSource


logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Abstract base class for data processors"""
    
    @abstractmethod
    def process(self, source: DataSource, input_path: str) -> Any:
        """Process data from input_path according to source configuration"""
        pass


class ShapefileProcessor(BaseProcessor):
    """Processor for Shapefile format"""
    
    def process(self, source: DataSource, input_path: str) -> gpd.GeoDataFrame:
        """Load shapefile into GeoDataFrame"""
        try:
            # If input is a zip file, extract first
            if input_path.endswith('.zip'):
                extract_dir = input_path.replace('.zip', '_extracted')
                with zipfile.ZipFile(input_path, 'r') as zf:
                    zf.extractall(extract_dir)
                
                # Find .shp file in extracted contents
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.endswith('.shp'):
                            input_path = os.path.join(root, file)
                            break
            
            # Load the shapefile
            gdf = gpd.read_file(input_path)
            
            # Ensure CRS is set
            if gdf.crs is None:
                logger.warning(f"No CRS found for {source.name}, assuming EPSG:4326")
                gdf.crs = "EPSG:4326"
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error processing shapefile {source.name}: {e}")
            raise


class GeoJSONProcessor(BaseProcessor):
    """Processor for GeoJSON format"""
    
    def process(self, source: DataSource, input_path: str) -> gpd.GeoDataFrame:
        """Load GeoJSON into GeoDataFrame"""
        try:
            gdf = gpd.read_file(input_path)
            
            if gdf.crs is None:
                logger.warning(f"No CRS found for {source.name}, assuming EPSG:4326")
                gdf.crs = "EPSG:4326"
                
            return gdf
            
        except Exception as e:
            logger.error(f"Error processing GeoJSON {source.name}: {e}")
            raise


class PBFProcessor(BaseProcessor):
    """Processor for OpenStreetMap PBF format"""
    
    def process(self, source: DataSource, input_path: str) -> gpd.GeoDataFrame:
        """Convert PBF to GeoDataFrame using ogr2ogr"""
        try:
            # Extract parameters for PBF processing
            layer_type = source.params.get('layer_type', 'points')
            sql_filter = source.params.get('sql_filter', 'place IS NOT NULL')
            
            # Output path for GeoJSON
            output_path = input_path.replace('.pbf', f'_{layer_type}.geojson')
            
            # Build ogr2ogr command
            cmd = [
                'ogr2ogr',
                '-f', 'GeoJSON',
                output_path,
                input_path,
                layer_type,
                '-sql', f'SELECT * FROM "{layer_type}" WHERE {sql_filter}',
                '-lco', 'RFC7946=YES'
            ]
            
            logger.info(f"Converting PBF with ogr2ogr: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"ogr2ogr failed: {result.stderr}")
            
            # Load the resulting GeoJSON
            gdf = gpd.read_file(output_path)
            
            # Clean up intermediate file if requested
            if source.params.get('cleanup_intermediate', True):
                os.remove(output_path)
                
            return gdf
            
        except Exception as e:
            logger.error(f"Error processing PBF {source.name}: {e}")
            raise


class ProcessorFactory:
    """Factory for creating appropriate processors"""
    
    _processors = {
        'shapefile': ShapefileProcessor,
        'shp': ShapefileProcessor,
        'geojson': GeoJSONProcessor,
        'json': GeoJSONProcessor,
        'pbf': PBFProcessor,
    }
    
    @classmethod
    def get_processor(cls, format_type: str) -> BaseProcessor:
        """Get appropriate processor for format type"""
        processor_class = cls._processors.get(format_type.lower())
        if not processor_class:
            # Default to trying geopandas for unknown formats
            return GeoJSONProcessor()
        return processor_class() 