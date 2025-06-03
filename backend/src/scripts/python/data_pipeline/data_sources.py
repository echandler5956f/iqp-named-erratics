"""
Data source configurations for the Glacial Erratics project.

This module registers all the GIS data sources used by the project.
Adding a new data source is as simple as creating a DataSource object
and registering it.
"""

from .sources import DataSource
from . import registry


# HydroSHEDS data sources
hydrosheds_rivers = DataSource(
    name='hydrosheds_rivers',
    source_type='https',
    url='https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip',
    format='shapefile',
    output_dir='hydro'
)

hydrosheds_lakes = DataSource(
    name='hydrosheds_lakes', 
    source_type='https',
    url='https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip',
    format='shapefile',
    output_dir='hydro'
)

hydrosheds_basins = DataSource(
    name='hydrosheds_basins',
    source_type='https',
    url='https://data.hydrosheds.org/file/hydrobasins/standard/hybas_na_lev05_v1c.zip',
    format='shapefile',
    output_dir='hydro'
)

# OpenStreetMap data sources
osm_north_america = DataSource(
    name='osm_north_america',
    source_type='https',
    url='https://download.geofabrik.de/north-america-latest.osm.pbf',
    format='pbf',
    output_dir='settlements',
    params={
        'layer_type': 'points',
        'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
    }
)

osm_us = DataSource(
    name='osm_us',
    source_type='https',
    url='https://download.geofabrik.de/north-america/us-latest.osm.pbf',
    format='pbf',
    output_dir='settlements',
    params={
        'layer_type': 'points',
        'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
    }
)

osm_canada = DataSource(
    name='osm_canada',
    source_type='https',
    url='https://download.geofabrik.de/north-america/canada-latest.osm.pbf',
    format='pbf',
    output_dir='settlements',
    params={
        'layer_type': 'points',
        'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
    }
)

# Native territories data
native_territories = DataSource(
    name='native_territories',
    source_type='https',
    url='https://nativeland.info/api/index.php?maps=territories',
    format='geojson',
    output_dir='native'
)

native_languages = DataSource(
    name='native_languages',
    source_type='https',
    url='https://nativeland.info/api/index.php?maps=languages',
    format='geojson',
    output_dir='native'
)

native_treaties = DataSource(
    name='native_treaties',
    source_type='https',
    url='https://nativeland.info/api/index.php?maps=treaties',
    format='geojson',
    output_dir='native'
)

# Historical data sources
nhgis_historical = DataSource(
    name='nhgis_historical',
    source_type='https',
    url='https://www.nhgis.org/sites/www.nhgis.org/files/nhgis_shapefiles.zip',
    format='shapefile',
    output_dir='colonial'
)

colonial_roads = DataSource(
    name='colonial_roads',
    source_type='https',
    url='https://www.daart.online/data/roads_colonial_era.geojson',
    format='geojson',
    output_dir='roads'
)

# Elevation data sources
elevation_srtm90 = DataSource(
    name='elevation_srtm90_csi',
    source_type='ftp',
    url='ftp://srtm.csi.cgiar.org/SRTM_V41/SRTM_Data_GeoTiff/',
    format='geotiff',
    output_dir='elevation',
    params={
        'tile_based': True,
        'resolution': 90
    }
)

elevation_dem_na = DataSource(
    name='elevation_dem_na',
    source_type='https',
    url='https://www.ngdc.noaa.gov/mgg/topo/gltiles/arctic/arctic.tgz',
    format='geotiff',
    output_dir='elevation'
)


def register_all_sources():
    """Register all data sources with the global registry"""
    sources = [
        hydrosheds_rivers,
        hydrosheds_lakes,
        hydrosheds_basins,
        osm_north_america,
        osm_us,
        osm_canada,
        native_territories,
        native_languages,
        native_treaties,
        nhgis_historical,
        colonial_roads,
        elevation_srtm90,
        elevation_dem_na
    ]
    
    for source in sources:
        registry.register(source)


# Auto-register when module is imported
register_all_sources() 