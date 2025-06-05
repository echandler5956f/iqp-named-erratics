"""
Data source configurations for the Glacial Erratics project.

This module registers all the GIS data sources used by the project.
Adding a new data source is as simple as creating a DataSource object
and registering it.
"""
import os
from .sources import DataSource
from . import registry

# Define a base path for locally stored GIS data to make paths relative
_LOCAL_GIS_DATA_BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), # Current dir: data_pipeline
    '..',                     # Up to python/
    'data',                   # Into data/
    'gis'                     # Into gis/
))

def _local_path(relative_path_to_gis_subfolder: str) -> str:
    """Constructs an absolute path to a file/folder within the project's local GIS data store."""
    return os.path.join(_LOCAL_GIS_DATA_BASE_PATH, relative_path_to_gis_subfolder)

# HydroSHEDS data sources
hydrosheds_rivers = DataSource(
    name='hydrosheds_rivers',
    source_type='https',
    url='https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip',
    format='shapefile',
    output_dir='hydro/HydroRIVERS_v10_na_shp'
)

hydrosheds_lakes = DataSource(
    name='hydrosheds_lakes', 
    source_type='https',
    url='https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip',
    format='shapefile',
    output_dir='hydro/HydroLAKES_polys_v10_shp'
)

hydrosheds_basins = DataSource(
    name='hydrosheds_basins',
    source_type='https',
    url='https://data.hydrosheds.org/file/hydrobasins/standard/hybas_na_lev05_v1c.zip',
    format='shapefile',
    output_dir='hydro/hybas_na_lev05_v1c'
)

# OpenStreetMap data sources

# Too big to download, so we'll use a local file for now
osm_north_america = DataSource(
    name='osm_north_america',
    source_type='file',
    path=_local_path('osm/north_america_pbf/north-america-latest.osm.pbf'),
    format='pbf',
    output_dir='osm/north_america_pbf', # Specific for the PBF file itself
    params={
        'layer_type': 'points',
        'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
    }
)

# osm_north_america = DataSource(
#     name='osm_north_america',
#     source_type='https',
#     url='https://download.geofabrik.de/north-america-latest.osm.pbf',
#     format='pbf',
#     output_dir='osm/north_america_pbf', # Specific for the PBF file itself
#     params={
#         'layer_type': 'points',
#         'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
#     }
# )

# osm_us = DataSource(
#     name='osm_us',
#     source_type='https',
#     url='https://download.geofabrik.de/north-america/us-latest.osm.pbf',
#     format='pbf',
#     output_dir='osm/us_pbf',
#     params={
#         'layer_type': 'points',
#         'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
#     }
# )

# osm_canada = DataSource(
#     name='osm_canada',
#     source_type='https',
#     url='https://download.geofabrik.de/north-america/canada-latest.osm.pbf',
#     format='pbf',
#     output_dir='osm/canada_pbf',
#     params={
#         'layer_type': 'points',
#         'sql_filter': "place IN ('city', 'town', 'village', 'hamlet')"
#     }
# )

# Native territories data
native_territories = DataSource(
    name='native_territories',
    source_type='file',
    path=_local_path('native/territories.geojson'),
    format='geojson',
    output_dir='native'
)

native_languages = DataSource(
    name='native_languages',
    source_type='file',
    path=_local_path('native/languages.geojson'),
    format='geojson',
    output_dir='native'
)

native_treaties = DataSource(
    name='native_treaties',
    source_type='file',
    path=_local_path('native/treaties.geojson'),
    format='geojson',
    output_dir='native'
)

# From US Department of Transportation
natd_roads = DataSource(
    name='natd_roads',
    source_type='file',
    path=_local_path('roads/North_American_Roads.shp'),
    format='shapefile',
    output_dir='roads'
)

# From US Forest Service
forest_trails = DataSource(
    name='forest_trails',
    source_type='file',
    path=_local_path('trails/National_Forest_System_Trails_(Feature_Layer).shp'),
    format='shapefile',
    output_dir='trails'
)

# Single DataSource for all GMTED Tiled Elevation Data
gmted_elevation_tiled = DataSource(
    name='gmted_elevation_tiled',
    source_type='file', # All tiles are local
    format='geotiff',
    is_tiled=True,
    tile_paths=[
        _local_path('elevation/GMTED2010N30W090_300/30n090w_20101117_gmted_mea300.tif'),
        _local_path('elevation/GMTED2010N30W120_300/30n120w_20101117_gmted_mea300.tif'),
        _local_path('elevation/GMTED2010N50W060_300/50n060w_20101117_gmted_mea300.tif'),
        _local_path('elevation/GMTED2010N50W090_300/50n090w_20101117_gmted_mea300.tif'),
        _local_path('elevation/GMTED2010N50W120_300/50n120w_20101117_gmted_mea300.tif'),
        _local_path('elevation/GMTED2010N50W150_300/50n150w_20101117_gmted_mea300.tif'),
        _local_path('elevation/GMTED2010N50W180_300/50n180w_20101117_gmted_mea300.tif'),
    ],
    tile_centers=[ # (lon, lat) for the CENTER of each tile
        (-75.0, 45.0),   # Center for n30w090 tile
        (-105.0, 45.0),  # Center for n30w120 tile
        (-45.0, 65.0),   # Center for n50w060 tile
        (-75.0, 65.0),   # Center for n50w090 tile
        (-105.0, 65.0),  # Center for n50w120 tile
        (-135.0, 65.0),  # Center for n50w150 tile
        (-165.0, 65.0),  # Center for n50w180 tile
    ],
    tile_size_degrees=30.0,
    output_dir='elevation/gmted_tiled_data', # General directory for this tiled source
    description='Tiled GMTED2010 Mean Elevation data at 300 arc-seconds resolution.'
)

glcc_nademl = DataSource(
    name='nademl',
    source_type='file',
    path=_local_path('glcc/nademl.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nabatsl20 = DataSource(
    name='nabatsl20',
    source_type='file',
    path=_local_path('glcc/nabatsl20.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_naigbpl20 = DataSource(
    name='naigbpl20',
    source_type='file',
    path=_local_path('glcc/naigbpl20.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nalulcl20 = DataSource(
    name='nalulcl20',
    source_type='file',
    path=_local_path('glcc/nalulcl20.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandviapr92l = DataSource(
    name='nandviapr92l',
    source_type='file',
    path=_local_path('glcc/nandviapr92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandviaug92l = DataSource(
    name='nandviaug92l',
    source_type='file',
    path=_local_path('glcc/nandviaug92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvidec92l = DataSource(
    name='nandvidec92l',
    source_type='file',
    path=_local_path('glcc/nandvidec92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvifeb93l = DataSource(
    name='nandvifeb93l',
    source_type='file',
    path=_local_path('glcc/nandvifeb93l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvijan93l = DataSource(
    name='nandvijan93l',
    source_type='file',
    path=_local_path('glcc/nandvijan93l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvijun92l = DataSource(
    name='nandvijun92l',
    source_type='file',
    path=_local_path('glcc/nandvijun92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvimar93l = DataSource(
    name='nandvimar93l',
    source_type='file',
    path=_local_path('glcc/nandvimar93l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvimay92l = DataSource(
    name='nandvimay92l',
    source_type='file',
    path=_local_path('glcc/nandvimay92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvinov92l = DataSource(
    name='nandvinov92l',
    source_type='file',
    path=_local_path('glcc/nandvinov92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvioct92l = DataSource(
    name='nandvioct92l',
    source_type='file',
    path=_local_path('glcc/nandvioct92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nandvisep92l = DataSource(
    name='nandvisep92l',
    source_type='file',
    path=_local_path('glcc/nandvisep92l.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_naogel20 = DataSource(
    name='naogel20',
    source_type='file',
    path=_local_path('glcc/naogel20.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_nasbm2l20 = DataSource(
    name='nasbm2l20',
    source_type='file',
    path=_local_path('glcc/nasbm2l20.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_naslcrl20 = DataSource(
    name='naslcrl20',
    source_type='file',
    path=_local_path('glcc/naslcrl20.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_naurbanl = DataSource(
    name='naurbanl',
    source_type='file',
    path=_local_path('glcc/naurbanl.tif'),
    format='geotiff',
    output_dir='glcc'
)

glcc_navll20 = DataSource(
    name='navll20',
    source_type='file',
    path=_local_path('glcc/navll20.tif'),
    format='geotiff',
    output_dir='glcc'
)


def register_all_sources():
    """Register all data sources with the global registry"""
    sources = [
        hydrosheds_rivers,
        hydrosheds_lakes,
        hydrosheds_basins,
        osm_north_america,
        # osm_us,
        # osm_canada,
        native_territories,
        native_languages,
        native_treaties,
        natd_roads,
        forest_trails,
        gmted_elevation_tiled,
        glcc_nademl,
        glcc_nabatsl20,
        glcc_naigbpl20,
        glcc_nalulcl20,
        glcc_nandviapr92l,
        glcc_nandviaug92l,
        glcc_nandvidec92l,
        glcc_nandvifeb93l,
        glcc_nandvijan93l,
        glcc_nandvijun92l,
        glcc_nandvimar93l,
        glcc_nandvimay92l,
        glcc_nandvinov92l,
        glcc_nandvioct92l,
        glcc_nandvisep92l,
        glcc_naogel20,
        glcc_nasbm2l20,
        glcc_naslcrl20,
        glcc_naurbanl,
        glcc_navll20
    ]
    
    for source in sources:
        registry.register(source)

    print(f"[data_sources.py] Registered {len(sources)} data sources.")

# Auto-register when module is imported
register_all_sources() 