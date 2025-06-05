"""
Registry for data sources.

The registry provides a central place to register and retrieve data sources.
This allows us to define data sources declaratively and access them by name.
"""

from typing import Dict, Optional
from .sources import DataSource


class DataRegistry:
    """Central registry for all data sources"""
    
    def __init__(self):
        self._sources: Dict[str, DataSource] = {}
    
    def register(self, source: DataSource) -> None:
        """Register a data source"""
        if source.name in self._sources:
            raise ValueError(f"Data source '{source.name}' already registered")
        
        self._sources[source.name] = source
    
    def add_manual_source(self, name: str, local_path: str, format: str = 'auto', 
                         description: str = '', **kwargs) -> None:
        """
        Convenience method to register a manual data source.
        
        Args:
            name: Unique name for the data source
            local_path: Path to the manually downloaded data file
            format: Data format ('shapefile', 'geojson', 'pbf', etc.)
            description: Optional description of the data source
            **kwargs: Additional parameters for the data source
        """
        source = DataSource(
            name=name,
            source_type='manual',
            path=local_path,
            format=format,
            params={
                'description': description,
                'local_path': local_path,
                **kwargs
            }
        )
        self.register(source)
    
    def get(self, name: str) -> Optional[DataSource]:
        """Get a data source by name"""
        return self._sources.get(name)
    
    def list_sources(self) -> list[str]:
        """List all registered source names"""
        return list(self._sources.keys())
    
    def list_manual_sources(self) -> list[str]:
        """List all manual data sources"""
        return [name for name, source in self._sources.items() 
                if source.source_type == 'manual']
    
    def get_all_sources(self) -> Dict[str, DataSource]:
        """Return internal mapping (shallow copy) of all sources for inspection."""
        return dict(self._sources)
    
    def clear(self) -> None:
        """Clear all registered sources (mainly for testing)"""
        self._sources.clear()

# Global singleton registry used throughout the pipeline
REGISTRY = DataRegistry()

# For tests that may reference DataRegistry.REGISTRY via patching path chains, expose as class attribute
setattr(DataRegistry, 'REGISTRY', REGISTRY) 