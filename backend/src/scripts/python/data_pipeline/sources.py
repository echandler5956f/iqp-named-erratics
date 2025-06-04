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
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("DataSource name cannot be empty.")
        if self.source_type not in ['https', 'http', 'ftp', 'file', 'manual']:
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
            if self.source_type == 'file' and not self.path:
                raise ValueError("Path must be provided for non-tiled source_type 'file'.")
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
                raise ValueError("tile_centers must be provided for tiled datasets.")
            
            # Check consistency in lengths of tile attributes
            num_paths = len(self.tile_paths) if self.tile_paths else 0
            num_urls = len(self.tile_urls) if self.tile_urls else 0
            num_centers = len(self.tile_centers) if self.tile_centers else 0

            if self.source_type == 'file' and num_paths != num_centers:
                raise ValueError(f"Mismatch between number of tile_paths ({num_paths}) and tile_centers ({num_centers}).")
            if self.source_type in ['https', 'http', 'ftp'] and num_urls != num_centers:
                raise ValueError(f"Mismatch between number of tile_urls ({num_urls}) and tile_centers ({num_centers}).")

        # Set default cache_key if not provided
        if not self.cache_key:
            self.cache_key = self.name 

    def __repr__(self):
        return f"DataSource(name='{self.name}', type='{self.source_type}', format='{self.format}', is_tiled={self.is_tiled})" 