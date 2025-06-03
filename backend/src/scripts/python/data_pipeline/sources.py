"""
Data Source definitions for the pipeline system.

A DataSource encapsulates all the information needed to acquire and process
a specific dataset. This includes where to get it, how to process it, and
what the output should look like.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class DataSource:
    """Configuration for a data source"""
    
    name: str  # Unique identifier for the source
    source_type: str  # 'http', 'ftp', 'file', 'database', 'api'
    url: Optional[str] = None  # For remote sources
    path: Optional[str] = None  # For local files
    
    # Processing configuration
    format: str = 'auto'  # 'shapefile', 'geojson', 'pbf', 'geotiff', etc.
    processing_steps: List[str] = field(default_factory=list)  # e.g., ['extract_zip', 'convert_pbf']
    
    # Output configuration  
    output_dir: Optional[str] = None  # Where to save processed data
    cache_key: Optional[str] = None  # Custom cache key (defaults to name)
    
    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)  # Source-specific parameters
    
    def __post_init__(self):
        """Validate the data source configuration"""
        if not self.url and not self.path and self.source_type != 'database':
            raise ValueError(f"DataSource '{self.name}' must have either url or path")
        
        if not self.cache_key:
            self.cache_key = self.name 