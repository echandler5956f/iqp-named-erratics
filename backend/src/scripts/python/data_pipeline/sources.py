"""
Data Source definitions for the pipeline system.

A DataSource encapsulates all the information needed to acquire and process
a specific dataset. This includes where to get it, how to process it, and
what the output should look like.
"""

from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class DataSource:
    """
    Represents a data source, encapsulating all information needed to acquire and process it.
    """
    name: str
    source_type: str  # Type of source (e.g., 'https', 'ftp', 'file', 'manual')
    format: str       # Data format (e.g., 'shapefile', 'geojson', 'pbf', 'geotiff')

    # Optional attributes for standard sources
    url: Optional[str] = None
    path: Optional[str] = None
    output_dir: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)  # Source-specific parameters
    description: Optional[str] = None
    cache_key: Optional[str] = None # Optional explicit cache key

    # New attributes for tiled datasets
    is_tiled: bool = False
    tile_paths: Optional[List[str]] = field(default_factory=list)
    tile_urls: Optional[List[str]] = field(default_factory=list)
    tile_centers: Optional[List[Tuple[float, float]]] = field(default_factory=list) # List of (lon, lat) tuples
    tile_size_degrees: Optional[float] = None
    
    # Optional list of processing step identifiers or descriptions
    processing_steps: Optional[List[str]] = field(default_factory=list)

    # Optional list of columns to keep in the main cache (memory optimization)
    default_keep_cols: Optional[List[str]] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("DataSource name cannot be empty.")
        if self.source_type not in ['https','http', 'ftp', 'file', 'manual']:
            raise ValueError(f"Invalid source_type: {self.source_type}")
        
        # Convert format to lowercase for consistency
        self.format = self.format.lower()

        # Default output_dir to name if not provided
        if self.output_dir is None:
            self.output_dir = self.name

        # Validation for standard (non-tiled) sources
        if not self.is_tiled:
            if self.source_type in ['https', 'http', 'ftp'] and not self.url:
                raise ValueError(f"URL must be provided for non-tiled source_type '{self.source_type}'.")
            if self.source_type == 'file' and self.path is None:
                # Allow deferred path resolution; log for awareness
                pass
        else: # Validations specific to tiled datasets
            if not self.tile_size_degrees or self.tile_size_degrees <= 0:
                raise ValueError("tile_size_degrees must be a positive number for tiled datasets.")
            
            if self.source_type == 'file':
                if not self.tile_paths:
                    raise ValueError("tile_paths must be provided for a tiled 'file' source.")
                if self.path: # Path should not be used for tiled file source, use tile_paths
                    raise ValueError("For a tiled 'file' source, use 'tile_paths' not 'path'.")
            elif self.source_type in ['https', 'http', 'ftp']:
                if not self.tile_urls:
                    raise ValueError("tile_urls must be provided for a tiled remote source.")
                if self.url: # URL should not be used for tiled remote source, use tile_urls
                    raise ValueError("For a tiled remote source, use 'tile_urls' not 'url'.")
            else: # 'manual' tiled source might not have tile_paths/urls if paths are directly referenced in geo_utils
                pass # Or add specific validation if manual tiled sources have particular requirements

            if not self.tile_centers:
                # If tile_centers missing, fill with None to match tile_paths/tile_urls length
                if self.source_type == 'file' and self.tile_paths:
                    self.tile_centers = [None] * len(self.tile_paths)
                elif self.source_type in ['https', 'http', 'ftp'] and self.tile_urls:
                    self.tile_centers = [None] * len(self.tile_urls)
                else:
                    self.tile_centers = []

            num_paths = len(self.tile_paths) if self.tile_paths else 0
            num_urls = len(self.tile_urls) if self.tile_urls else 0
            num_centers = len(self.tile_centers) if self.tile_centers else 0

            # Only raise if both are non-empty and lengths mismatch
            if self.source_type == 'file' and num_paths and num_centers and num_paths != num_centers:
                raise ValueError(f"Mismatch between number of tile_paths ({num_paths}) and tile_centers ({num_centers}).")
            if self.source_type in ['https', 'http', 'ftp'] and num_urls and num_centers and num_urls != num_centers:
                raise ValueError(f"Mismatch between number of tile_urls ({num_urls}) and tile_centers ({num_centers}).")

        # Set default cache_key if not provided
        if not self.cache_key:
            self.cache_key = self.name 

    def __repr__(self):
        return f"DataSource(name='{self.name}', type='{self.source_type}', format='{self.format}', is_tiled={self.is_tiled})" 

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def tile_id_for_point(self, lon: float, lat: float) -> Optional[int]:
        """Return index of tile covering (*lon*, *lat*) if this is a tiled DataSource.

        The method performs a simple bounds check based on *tile_centers* and
        *tile_size_degrees*.  It returns **None** when the DataSource is not
        tiled or when the point falls outside the declared tiles.
        """
        if not self.is_tiled or not self.tile_centers or not self.tile_size_degrees:
            return None
        half = self.tile_size_degrees / 2.0
        for idx, (clon, clat) in enumerate(self.tile_centers):
            if clon is None or clat is None:
                continue
            if abs(lon - clon) <= half and abs(lat - clat) <= half:
                return idx
        return None

    def url_or_path_for_tile(self, tile_index: int) -> Optional[str]:
        """Return URL/path for *tile_index* respecting source_type."""
        if not self.is_tiled:
            return None
        if self.source_type in {'http', 'https', 'ftp'}:
            if 0 <= tile_index < len(self.tile_urls):
                return self.tile_urls[tile_index]
        else:  # 'file' or 'manual'
            if 0 <= tile_index < len(self.tile_paths):
                return self.tile_paths[tile_index]
        return None 