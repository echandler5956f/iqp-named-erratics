# Data Pipeline System

A modular replacement for the (old, deprecated) monolithic `data_loader.py` that provides clean separation of concerns and easy extensibility.

## Overview

The data pipeline system separates data loading into distinct components:

- **DataSource**: Configuration for where and how to get data, including support for tiled datasets.
- **DataRegistry**: Central registry of all data sources
- **Loaders**: Protocol-specific data acquisition (HTTP, FTP, file)
- **Processors**: Format-specific data processing (Shapefile, GeoJSON, PBF)
- **CacheManager**: Feather-based caching for processed data
- **DataPipeline**: Orchestrates the loading process

## Quick Start

### Using Existing Data Sources

```python
from data_pipeline import load_data

# Load hydrological features
rivers = load_data('hydrosheds_rivers')
lakes = load_data('hydrosheds_lakes')

# Load settlements
settlements = load_data('osm_north_america')

# Force reload (bypass cache)
fresh_data = load_data('native_territories', force_reload=True)
```

### Adding a New Data Source

1. Define the source in `data_sources.py`:

```python
from .sources import DataSource
from . import registry

# Define your data source
my_new_source = DataSource(
    name='my_data',
    source_type='https',  # or 'ftp', 'file'
    url='https://example.com/data.zip',
    format='shapefile',   # or 'geojson', 'pbf', etc.
    output_dir='custom'   # subdirectory for downloads
)

# Register it
registry.register(my_new_source)
```

2. Use it:

```python
my_data = load_data('my_data')
```

### Adding Manual Data Sources

For data that requires manual download (special access, authentication, etc.), use the manual data feature:

#### Simple Method

```python
from data_pipeline import add_manual_data, load_data

# Add manually downloaded data
add_manual_data(
    name='restricted_dataset',
    local_path='/path/to/manually/downloaded/data.shp',
    format='shapefile',
    description='Special access dataset requiring manual download'
)

# Use it like any other data source
data = load_data('restricted_dataset')
```

#### Advanced Method

```python
from data_pipeline import registry
from data_pipeline.sources import DataSource

# For more control, create DataSource directly
manual_source = DataSource(
    name='custom_manual_data',
    source_type='manual',
    path='/path/to/data.geojson',
    format='geojson',
    params={
        'description': 'Manually curated dataset',
        'access_level': 'restricted',
        'download_date': '2024-01-15',
        'source_agency': 'Research Institute'
    }
)

registry.register(manual_source)
data = load_data('custom_manual_data')
```

#### Registry Methods for Manual Data

```python
from data_pipeline import registry

# List all manual sources
manual_sources = registry.list_manual_sources()

# Add via registry directly
registry.add_manual_source(
    name='fieldwork_data',
    local_path='/data/fieldwork/sites.geojson',
    format='geojson',
    description='Field-collected GPS points',
    collection_date='2024-01-20',
    collector='Field Team A'
)
```

#### Benefits of Manual Data Integration

1. **Clean Integration**: Manual data works exactly like automatic sources
2. **Full Pipeline Benefits**: Caching, format processing, error handling
3. **Metadata Support**: Store additional information about the data source
4. **No Code Changes**: Existing analysis scripts work unchanged
5. **Easy Management**: List and track manual data sources

#### Manual Data Workflow

1. **Download**: Manually download data to your local system
2. **Register**: Use `add_manual_data()` or register a DataSource
3. **Load**: Use `load_data()` like any other source
4. **Benefit**: Automatic caching, format processing, etc.

Example complete workflow:
```python
# Step 1: You manually download special_data.shp to /data/special/

# Step 2: Register it
add_manual_data(
    'special_analysis_data',
    '/data/special/special_data.shp', 
    format='shapefile',
    description='Special access geological survey data'
)

# Step 3: Use in analysis (with caching, etc.)
geological_data = load_data('special_analysis_data')
analysis_results = perform_analysis(geological_data)

# Step 4: Re-run later - loads from cache instantly
geological_data = load_data('special_analysis_data')  # From cache
```

## Architecture

### Core Components

#### DataSource
Encapsulates all information needed to acquire and process data:
- Source location (URL, file path)
- Source type (HTTP, FTP, file, database, manual)
- Data format (Shapefile, GeoJSON, PBF, GeoTIFF, etc.)
- Processing parameters
- Output configuration
- For **tiled datasets** (like GMTED elevation data):
    - `is_tiled` (boolean): Marks the source as a collection of tiles.
    - `tile_paths` (list): List of file paths for each individual, locally available tile.
    - `tile_urls` (list): List of URLs if tiles are remote (less common for current setup).
    - `tile_centers` (list of tuples): List of (longitude, latitude) for the center of each tile.
    - `tile_size_degrees` (float): The side length of each square tile in decimal degrees.

#### DataRegistry
Central registry that manages all data sources:
- Register new sources
- Retrieve sources by name
- List available sources

#### Loaders
Protocol-specific classes for data acquisition:
- `HTTPLoader`: Downloads via HTTP/HTTPS
- `FTPLoader`: Downloads from FTP servers
- `FileLoader`: Handles local files
- `ManualLoader`: Validates and references manually downloaded data

#### Processors
Format-specific classes for data processing:
- `ShapefileProcessor`: Handles .shp files and zipped shapefiles
- `GeoJSONProcessor`: Processes GeoJSON files
- `PBFProcessor`: Converts OSM PBF files using ogr2ogr

#### CacheManager
Handles caching of processed data:
- Uses Feather format for fast serialization
- Supports geometry columns via WKB conversion
- Automatic cache invalidation based on file timestamps

#### DataPipeline
Orchestrates the entire process:
1. Check cache
2. Acquire raw data (if needed)
3. Process data
4. Cache results
5. Return processed data

## Backward Compatibility

The `compat.py` module provides the same API as the old `data_loader.py`:

```python
# Old code continues to work
from python.data_pipeline.compat import (
    load_erratics,
    load_hydro_features,
    load_settlements,
    load_roads,
    # ... etc
)
```

## Migration

To migrate existing scripts:

```bash
python migrate_to_pipeline.py
```

This will:
1. Update import statements
2. Create backups of original files
3. Maintain full compatibility

## Extending the System

### Adding a New Loader

```python
from .loaders import BaseLoader

class S3Loader(BaseLoader):
    def load(self, source: DataSource, target_path: str) -> bool:
        # Implement S3 download logic
        pass

# Register in LoaderFactory
LoaderFactory._loaders['s3'] = S3Loader
```

### Adding a New Processor

```python
from .processors import BaseProcessor

class NetCDFProcessor(BaseProcessor):
    def process(self, source: DataSource, input_path: str) -> Any:
        # Implement NetCDF processing
        pass

# Register in ProcessorFactory
ProcessorFactory._processors['netcdf'] = NetCDFProcessor
```

## Benefits

1. **Separation of Concerns**: Each component has a single responsibility
2. **Easy Extension**: Add new sources without modifying existing code
3. **Declarative Configuration**: Data sources are defined as data, not code
4. **Caching**: Automatic caching with smart invalidation
5. **Testability**: Each component can be tested in isolation
6. **Backward Compatible**: Existing scripts continue to work

## Example: Complete Workflow

```python
from data_pipeline import DataSource, registry, load_data

# 1. Define a new data source (non-tiled example)
earthquake_data = DataSource(
    name='usgs_earthquakes',
    source_type='https',
    url='https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson',
    format='geojson',
    output_dir='earthquakes',
    params={
        'min_magnitude': 4.0  # Custom parameter
    }
)
registry.register(earthquake_data)

# Example of a Tiled DataSource definition (conceptual, actual in data_sources.py):
# gmted_elevation_tiled = DataSource(
#     name='gmted_elevation_tiled',
#     source_type='file', # Assuming tiles are local
#     format='geotiff',
#     is_tiled=True,
#     tile_paths=['/path/to/tile1.tif', '/path/to/tile2.tif'],
#     tile_centers=[(-75.0, 45.0), (-70.0, 40.0)], # (lon, lat)
#     tile_size_degrees=30.0,
#     description='Tiled GMTED Elevation Data'
# )
# registry.register(gmted_elevation_tiled)

# 3. Load the data (for non-tiled sources directly, or specific tiles via custom logic)
# For non-tiled:
earthquakes = load_data('usgs_earthquakes')

# For tiled sources like 'gmted_elevation_tiled', scripts like geo_utils.py
# would typically access the registry to get the DataSource object,
# then use its tile_paths, tile_centers, and tile_size_degrees to find and 
# directly use the path to the relevant tile for a given point of interest.
# The main load_data('gmted_elevation_tiled') might not be used to load all tiles at once.

# 4. Use runtime parameters (for non-tiled sources)
big_quakes = load_data('usgs_earthquakes', min_magnitude=6.0)
```

## Directory Structure

```
data_pipeline/
├── __init__.py          # Main entry point
├── sources.py           # DataSource class
├── registry.py          # DataRegistry class
├── loaders.py           # Protocol-specific loaders
├── processors.py        # Format-specific processors
├── cache.py             # Cache management
├── pipeline.py          # Main orchestrator
├── data_sources.py      # All registered sources
├── compat.py            # Backward compatibility
└── README.md            # This file
```

## Future Enhancements

- [ ] Add database loader for PostGIS queries
- [ ] Support for parallel downloads
- [ ] Streaming processing for large files
- [ ] Plugin system for custom loaders/processors
- [ ] Configuration file support (YAML/JSON)
- [ ] Progress callbacks for long operations
- [ ] Automatic retry with exponential backoff
- [ ] Data validation framework 