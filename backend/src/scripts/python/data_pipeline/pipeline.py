"""
Main data pipeline orchestrator.

The DataPipeline class coordinates the loading, processing, and caching
of data from registered sources.
"""

import os
import logging
from typing import Any, Optional, Dict

from shapely.geometry import box
from .sources import DataSource
from .registry import DataRegistry
from .loaders import LoaderFactory
from .processors import ProcessorFactory
from .cache import CacheManager
from .download_cache import RawDownloadCache


logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates the data loading pipeline"""
    
    def __init__(self, registry: DataRegistry, cache_dir: Optional[str] = None):
        """Initialize pipeline with a data registry"""
        self.registry = registry
        self.cache_manager = CacheManager(cache_dir)
        self.download_cache = RawDownloadCache()
        
        # Default directories - match the old data_loader structure
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, 'data')
        self.gis_data_dir = os.path.join(self.data_dir, 'gis')
        
        # Ensure directories exist
        os.makedirs(self.gis_data_dir, exist_ok=True)
        
    def load(self, source_name: str, force_reload: bool = False, **kwargs) -> Any:
        """
        Load data from a registered source with smart caching and runtime filtering.
        
        Args:
            source_name: Name of the registered data source
            force_reload: Force reload even if cached
            **kwargs: Additional parameters including:
                     bbox: Optional (minx, miny, maxx, maxy) for spatial filtering
                     keep_cols: Optional list of columns to keep
            
        Returns:
            Processed data (typically a GeoDataFrame)
        """
        # Get source configuration
        source = self.registry.get(source_name)
        if not source:
            raise ValueError(f"Unknown data source: {source_name}")
        
        # Extract runtime hints (not part of cache key!)
        runtime_bbox = kwargs.pop('bbox', None)
        runtime_keep_cols = kwargs.pop('keep_cols', None)
        
        # Merge any remaining kwargs with source params
        if kwargs:
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
                params=merged_params,
                default_keep_cols=source.default_keep_cols
            )
        
        # Path 1: Try to load the canonical (full) version from CacheManager
        cached_full_gdf = None
        if not force_reload:
            cached_full_gdf = self.cache_manager.load(source.cache_key, source.params)

        if cached_full_gdf is not None:
            logger.debug(f"Cache hit for {source_name} (full). Applying runtime filters if any.")
            gdf_to_return = cached_full_gdf
            
            # Apply runtime bbox to the GDF loaded from the main cache
            if runtime_bbox and not gdf_to_return.empty:
                minx, miny, maxx, maxy = runtime_bbox
                try:
                    # Use spatial index for efficient querying
                    if hasattr(gdf_to_return, 'sindex') and len(gdf_to_return) > 100:
                        possible_matches_idx = list(gdf_to_return.sindex.intersection(runtime_bbox))
                        if not possible_matches_idx:
                            gdf_to_return = gdf_to_return.iloc[0:0]
                        else:
                            # Precise intersection
                            intersecting_mask = gdf_to_return.iloc[possible_matches_idx].intersects(box(*runtime_bbox))
                            gdf_to_return = gdf_to_return.iloc[possible_matches_idx][intersecting_mask]
                    elif not gdf_to_return.empty:
                        # Fallback for small datasets or no spatial index
                        gdf_to_return = gdf_to_return[gdf_to_return.intersects(box(*runtime_bbox))]
                except Exception as e_bbox_filter:
                    logger.warning(f"Error applying bbox filter to cached GDF for {source_name}: {e_bbox_filter}. Returning full cached GDF.")

            # Apply runtime keep_cols
            if runtime_keep_cols and not gdf_to_return.empty:
                cols_present = [col for col in runtime_keep_cols if col in gdf_to_return.columns]
                # Always keep the geometry column
                geom_col_name = gdf_to_return.geometry.name
                if geom_col_name not in cols_present:
                    cols_present.append(geom_col_name)
                gdf_to_return = gdf_to_return[list(set(cols_present))]
            
            logger.info(f"Returning {('filtered ' if runtime_bbox or runtime_keep_cols else '')}data for {source_name} from cache.")
            return gdf_to_return

        # Path 2: Canonical version not in CacheManager. Process full, cache full, then filter.
        logger.info(f"Cache miss for {source_name} (full). Processing from raw to create canonical cache entry.")
        raw_data_path = self._acquire_data(source)
        if not raw_data_path:
            logger.error(f"Failed to acquire raw data for {source_name}")
            raise RuntimeError(f"Failed to acquire data for {source_name}")

        # Process the data for canonical cache (use default_keep_cols if defined)
        default_cols_for_cache = source.default_keep_cols
        full_processed_gdf = self._process_data(source, raw_data_path, 
                                                bbox=None,  # Process full for cache
                                                keep_cols=default_cols_for_cache)

        if full_processed_gdf is None or full_processed_gdf.empty:
            logger.warning(f"Processing {source_name} from raw yielded no data. Caching empty.")
            import geopandas as gpd
            full_processed_gdf = gpd.GeoDataFrame([], columns=default_cols_for_cache or ['geometry'])

        # Save the full dataset to cache
        if full_processed_gdf is not None:
            self.cache_manager.save(full_processed_gdf, source.cache_key, source.params)
            logger.info(f"Saved {source_name} (full) to CacheManager.")
        
        # Now apply runtime filters to this newly processed full GDF
        gdf_to_return = full_processed_gdf
        if runtime_bbox and not gdf_to_return.empty:
            minx, miny, maxx, maxy = runtime_bbox
            try:
                if hasattr(gdf_to_return, 'sindex') and len(gdf_to_return) > 100:
                    possible_matches_idx = list(gdf_to_return.sindex.intersection(runtime_bbox))
                    if not possible_matches_idx:
                        gdf_to_return = gdf_to_return.iloc[0:0]
                    else:
                        intersecting_mask = gdf_to_return.iloc[possible_matches_idx].intersects(box(*runtime_bbox))
                        gdf_to_return = gdf_to_return.iloc[possible_matches_idx][intersecting_mask]
                elif not gdf_to_return.empty:
                    gdf_to_return = gdf_to_return[gdf_to_return.intersects(box(*runtime_bbox))]
            except Exception as e_bbox_filter_after_proc:
                logger.warning(f"Error applying runtime bbox filter to newly processed GDF for {source_name}: {e_bbox_filter_after_proc}. Returning full.")

        if runtime_keep_cols and not gdf_to_return.empty:
            cols_present = [col for col in runtime_keep_cols if col in gdf_to_return.columns]
            geom_col_name = gdf_to_return.geometry.name
            if geom_col_name not in cols_present:
                cols_present.append(geom_col_name)
            gdf_to_return = gdf_to_return[list(set(cols_present))]
        
        logger.info(f"Returning {('filtered ' if runtime_bbox or runtime_keep_cols else '')}data for {source_name} after processing from raw.")
        return gdf_to_return
    
    def _acquire_data(self, source: DataSource) -> Optional[str]:
        """Acquire raw data from source"""
        # Local file sources: just verify path exists
        if source.source_type == 'file' and source.path:
            if os.path.exists(source.path):
                return source.path
            logger.error(f"Local file not found: {source.path}")
            return None

        # Manual data sources: user-provided local files
        if source.source_type == 'manual':
            manual_path = source.params.get('local_path') or source.path
            if manual_path and os.path.exists(manual_path):
                return manual_path
            logger.error(f"Manual data file not found: {manual_path}")
            return None

        # Remote sources (http/https/ftp): use global raw cache with locking
        # ------------------------------------------------------------------
        # If remote source, delegate to RawDownloadCache which handles locking & reuse
        if source.source_type in {"http", "https", "ftp"} and source.url:
            def _dl(target_fp: str) -> bool:
                # Loader needs to stream into *target_fp* path.
                loader = LoaderFactory.get_loader(source.source_type)
                return loader.load(source, target_fp)

            cached_path = self.download_cache.ensure(source.url, source.name, _dl)
            return cached_path

        # Fallback to previous behaviour for other source types or when URL missing
        if source.url:
            filename = os.path.basename(source.url)
        else:
            filename = f"{source.name}_data"

        if source.output_dir:
            target_dir = os.path.join(self.gis_data_dir, source.output_dir, source.name)
        else:
            target_dir = os.path.join(self.gis_data_dir, source.name)

        os.makedirs(target_dir, exist_ok=True)
        download_path = os.path.join(target_dir, filename)

        if os.path.exists(download_path):
            logger.info(f"Using existing file: {download_path}")
            return download_path

        try:
            loader = LoaderFactory.get_loader(source.source_type)
        except ValueError as e:
            logger.error(f"No loader available: {e}")
            return None

        success = loader.load(source, download_path)
        return download_path if success else None
    
    def _process_data(self, source: DataSource, raw_path: str, *, bbox=None, keep_cols=None) -> Any:
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
            data = processor.process(source, raw_path, bbox=bbox, keep_cols=keep_cols)
            # If keep_cols specified, ensure geometry retained
            if keep_cols:
                keep_set = set(keep_cols)
                keep_set.add(data.geometry.name)
                data = data[[c for c in data.columns if c in keep_set]]
            return data
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