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

__all__ = ['DataRegistry', 'DataPipeline', 'DataSource', 'registry', 'load_data'] 