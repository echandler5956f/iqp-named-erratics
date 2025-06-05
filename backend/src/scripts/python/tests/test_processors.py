import pytest
import json
import geopandas as gpd

from data_pipeline.sources import DataSource
from data_pipeline.processors import GeoJSONProcessor, ProcessorFactory, ShapefileProcessor, PBFProcessor


@pytest.fixture
def valid_geojson_content():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Test Point", "value": 123},
                "geometry": {"type": "Point", "coordinates": [-75.0, 45.0]}
            }
        ]
    }

@pytest.fixture
def geojson_with_crs_content():
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "CRS Point"},
                "geometry": {"type": "Point", "coordinates": [0, 0]}
            }
        ]
    }

class TestGeoJSONProcessor:
    def test_process_valid_geojson(self, tmp_path, valid_geojson_content):
        source_path = tmp_path / "test.geojson"
        with open(source_path, 'w') as f:
            json.dump(valid_geojson_content, f)
        
        source_ds = DataSource(name="test_geojson", source_type="file", format="geojson", path=str(source_path))
        processor = GeoJSONProcessor()
        
        gdf = processor.process(source_ds, str(source_path))
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.iloc[0]["name"] == "Test Point"
        assert gdf.iloc[0]["value"] == 123
        assert gdf.crs == "EPSG:4326" # Default CRS when none is in file

    def test_process_geojson_with_existing_crs(self, tmp_path, geojson_with_crs_content):
        source_path = tmp_path / "test_crs.geojson"
        with open(source_path, 'w') as f:
            json.dump(geojson_with_crs_content, f)
        
        source_ds = DataSource(name="test_geojson_crs", source_type="file", format="geojson", path=str(source_path))
        processor = GeoJSONProcessor()
        
        gdf = processor.process(source_ds, str(source_path))
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.crs.to_string() == "EPSG:3857"

    def test_process_invalid_geojson_string(self, tmp_path):
        source_path = tmp_path / "invalid.geojson"
        source_path.write_text("this is not valid json or geojson")
        
        source_ds = DataSource(name="test_invalid_geojson", source_type="file", format="geojson", path=str(source_path))
        processor = GeoJSONProcessor()
        
        with pytest.raises(Exception): # Geopandas might raise different errors, Exception is broad.
            processor.process(source_ds, str(source_path))

    def test_process_empty_geojson_file(self, tmp_path):
        source_path = tmp_path / "empty.geojson"
        source_path.touch() # Create an empty file
        
        source_ds = DataSource(name="test_empty_geojson", source_type="file", format="geojson", path=str(source_path))
        processor = GeoJSONProcessor()
        
        with pytest.raises(ValueError): # Changed to ValueError for empty file issues
            processor.process(source_ds, str(source_path))

class TestProcessorFactory:
    def test_get_geojson_processor(self):
        processor_geojson = ProcessorFactory.get_processor('geojson')
        assert isinstance(processor_geojson, GeoJSONProcessor)
        processor_json = ProcessorFactory.get_processor('json') # Alias
        assert isinstance(processor_json, GeoJSONProcessor)

    def test_get_shapefile_processor(self):
        processor_shp = ProcessorFactory.get_processor('shapefile')
        assert isinstance(processor_shp, ShapefileProcessor)
        processor_shp_alias = ProcessorFactory.get_processor('shp')
        assert isinstance(processor_shp_alias, ShapefileProcessor)

    def test_get_pbf_processor(self):
        processor = ProcessorFactory.get_processor('pbf')
        assert isinstance(processor, PBFProcessor)
    
    def test_get_default_processor_for_unknown(self):
        # The factory currently defaults to GeoJSONProcessor for unknown types
        processor = ProcessorFactory.get_processor('unknown_format')
        assert isinstance(processor, GeoJSONProcessor) 