"""
Main data pipeline orchestrator.

The DataPipeline class coordinates the loading, processing, and caching
of data from registered sources.
"""

import os
import logging
from typing import Any, Optional, Dict
from pathlib import Path

from .sources import DataSource
from .registry import DataRegistry
from .loaders import LoaderFactory
from .processors import ProcessorFactory
from .cache import CacheManager


logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates the data loading pipeline"""
    
    def __init__(self, registry: DataRegistry, cache_dir: Optional[str] = None):
        """Initialize pipeline with a data registry"""
        self.registry = registry
        self.cache_manager = CacheManager(cache_dir)
        
        # Default directories - match the old data_loader structure
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, 'data')
        self.gis_data_dir = os.path.join(self.data_dir, 'gis')
        
        # Ensure directories exist
        os.makedirs(self.gis_data_dir, exist_ok=True)
        
    def load(self, source_name: str, force_reload: bool = False, **kwargs) -> Any:
        """
        Load data from a registered source.
        
        Args:
            source_name: Name of the registered data source
            force_reload: Force reload even if cached
            **kwargs: Additional parameters to override source defaults
            
        Returns:
            Processed data (typically a GeoDataFrame)
        """
        # Get source configuration
        source = self.registry.get(source_name)
        if not source:
            raise ValueError(f"Unknown data source: {source_name}")
        
        # Merge any runtime parameters
        if kwargs:
            # Create a new source with merged parameters
            merged_params = {**source.params, **kwargs}
            source = DataSource(
                name=source.name,
                source_type=source.source_type,
                url=source.url,
                path=source.path,
                format=source.format,
                processing_steps=source.processing_steps,
                output_dir=source.output_dir,
                cache_key=source.cache_key,
                params=merged_params
            )
        
        # Check cache first
        if not force_reload:
            cached_data = self.cache_manager.load(source.cache_key, source.params)
            if cached_data is not None:
                logger.info(f"Loaded {source_name} from cache")
                return cached_data
        
        # Download/acquire the data
        raw_data_path = self._acquire_data(source)
        if not raw_data_path:
            raise RuntimeError(f"Failed to acquire data for {source_name}")
        
        # Process the data
        processed_data = self._process_data(source, raw_data_path)
        
        # Cache the processed data
        if processed_data is not None:
            self.cache_manager.save(processed_data, source.cache_key, source.params)
        
        return processed_data
    
    def _acquire_data(self, source: DataSource) -> Optional[str]:
        """Acquire raw data from source"""
        # For local files, just return the path
        if source.source_type == 'file' and source.path:
            if os.path.exists(source.path):
                return source.path
            else:
                logger.error(f"Local file not found: {source.path}")
                return None
        
        # Determine download path
        if source.url:
            filename = os.path.basename(source.url)
        else:
            filename = f"{source.name}_data"
        
        # Build the full path using GIS data directory as base
        if source.output_dir:
            # output_dir is relative to gis_data_dir
            target_dir = os.path.join(self.gis_data_dir, source.output_dir, source.name)
        else:
            # Default to gis_data_dir/source_name
            target_dir = os.path.join(self.gis_data_dir, source.name)
            
        download_path = os.path.join(target_dir, filename)
        
        # Skip download if file already exists
        if os.path.exists(download_path):
            logger.info(f"Using existing file: {download_path}")
            return download_path
        
        # Get appropriate loader
        try:
            loader = LoaderFactory.get_loader(source.source_type)
        except ValueError as e:
            logger.error(f"No loader available: {e}")
            return None
        
        # Download the data
        success = loader.load(source, download_path)
        if success:
            return download_path
        else:
            return None
    
    def _process_data(self, source: DataSource, raw_path: str) -> Any:
        """Process raw data according to source configuration"""
        # Determine format
        if source.format == 'auto':
            # Try to infer from file extension
            ext = os.path.splitext(raw_path)[1].lower()
            format_map = {
                '.shp': 'shapefile',
                '.geojson': 'geojson',
                '.json': 'geojson',
                '.pbf': 'pbf',
                '.zip': 'shapefile',  # Assume zipped shapefiles
            }
            data_format = format_map.get(ext, 'geojson')
        else:
            data_format = source.format
        
        # Get appropriate processor
        processor = ProcessorFactory.get_processor(data_format)
        
        # Process the data
        try:
            return processor.process(source, raw_path)
        except Exception as e:
            logger.error(f"Error processing {source.name}: {e}")
            raise
    
    def clear_cache(self, source_name: Optional[str] = None) -> None:
        """Clear cache for a specific source or all sources"""
        if source_name:
            source = self.registry.get(source_name)
            if source:
                self.cache_manager.clear(source.cache_key)
        else:
            self.cache_manager.clear() 