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
    
    def get(self, name: str) -> Optional[DataSource]:
        """Get a data source by name"""
        return self._sources.get(name)
    
    def list_sources(self) -> list[str]:
        """List all registered source names"""
        return list(self._sources.keys())
    
    def clear(self) -> None:
        """Clear all registered sources (mainly for testing)"""
        self._sources.clear() 