"""
Data Pipeline System for Glacial Erratics Project

A modular replacement for the monolithic data_loader.py that provides:
- Clean separation of concerns
- Easy extensibility for new data sources  
- Minimal complexity
- Backward compatibility with existing analysis scripts
"""

from .registry import DataRegistry
from .pipeline import DataPipeline
from .sources import DataSource

# Initialize the global registry
registry = DataRegistry()

# Import data sources to register them
from . import data_sources

# Main entry point for backward compatibility
def load_data(source_name: str, **kwargs):
    """Load data from a registered source"""
    pipeline = DataPipeline(registry)
    return pipeline.load(source_name, **kwargs)

# Convenience function for adding manual data sources
def add_manual_data(name: str, local_path: str, format: str = 'auto', 
                   description: str = '', **kwargs):
    """
    Add a manually downloaded data source to the pipeline.
    
    Args:
        name: Unique name for the data source
        local_path: Path to the manually downloaded data file
        format: Data format ('shapefile', 'geojson', 'pbf', etc.) 
        description: Optional description of the data source
        **kwargs: Additional parameters for the data source
        
    Example:
        add_manual_data(
            'special_dataset',
            '/path/to/manually/downloaded/data.shp',
            format='shapefile',
            description='Special access dataset requiring manual download'
        )
        
        # Then use it like any other data source
        data = load_data('special_dataset')
    """
    registry.add_manual_source(name, local_path, format, description, **kwargs)

__all__ = ['DataRegistry', 'DataPipeline', 'DataSource', 'registry', 'load_data', 'add_manual_data'] 