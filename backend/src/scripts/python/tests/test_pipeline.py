import pytest
import os
from unittest import mock
import geopandas as gpd
from shapely.geometry import Point

from data_pipeline.sources import DataSource
from data_pipeline.pipeline import DataPipeline
from data_pipeline.registry import DataRegistry
from data_pipeline.cache import CacheManager
from data_pipeline.loaders import FileLoader # Assuming FileLoader for some tests
from data_pipeline.processors import GeoJSONProcessor # Assuming for some tests

@pytest.fixture
def mock_registry():
    return mock.MagicMock(spec=DataRegistry)

@pytest.fixture
def mock_cache_manager():
    return mock.MagicMock(spec=CacheManager)

@pytest.fixture
def mock_file_loader():
    loader = mock.MagicMock(spec=FileLoader)
    loader.load.return_value = True # Default to successful load
    return loader

@pytest.fixture
def mock_geojson_processor():
    processor = mock.MagicMock(spec=GeoJSONProcessor)
    # Default to returning a simple GeoDataFrame
    processor.process.return_value = gpd.GeoDataFrame({'id': [1], 'geometry': [Point(0,0)]}, crs="EPSG:4326")
    return processor

@pytest.fixture
def sample_source_def():
    return DataSource(
        name="sample_data",
        source_type="file",
        format="geojson",
        path="/fake/path/to/sample_data.geojson",
        output_dir="sample_output",
        params={"year": 2023}
    )

@pytest.fixture
def pipeline_instance(mock_registry, mock_cache_manager, tmp_path):
    # Ensure the pipeline uses our mocked cache manager by passing its dir
    # The pipeline will instantiate its own CacheManager, so we patch CacheManager directly for some tests
    # or ensure that if a cache_dir is passed, it's used, and then we can inspect that dir.
    # For simplicity in mocking, let's allow injecting the mock cache_manager if the class is changed to support it,
    # or patch CacheManager constructor.
    
    # Patching CacheManager within the pipeline's scope for these tests
    with mock.patch('data_pipeline.pipeline.CacheManager', return_value=mock_cache_manager) as patched_cache_mgr:
        pipeline = DataPipeline(registry=mock_registry, cache_dir=str(tmp_path / "pipe_cache"))
        # pipeline.cache_manager will be the instance of the patched_cache_mgr
        return pipeline

class TestDataPipeline:

    def test_load_cache_hit(self, pipeline_instance, mock_registry, mock_cache_manager, sample_source_def):
        mock_gdf = gpd.GeoDataFrame({'data': ['cached']})
        mock_registry.get.return_value = sample_source_def
        mock_cache_manager.load.return_value = mock_gdf
        # is_cached is not directly called by pipeline.load(), it relies on load() returning None for miss

        result_gdf = pipeline_instance.load(sample_source_def.name)

        mock_registry.get.assert_called_once_with(sample_source_def.name)
        mock_cache_manager.load.assert_called_once_with(sample_source_def.cache_key, sample_source_def.params)
        assert result_gdf.equals(mock_gdf)
        # Ensure loader and processor were not called
        # This requires knowing what LoaderFactory.get_loader and ProcessorFactory.get_processor would return
        # and asserting their methods were not called. For now, focusing on cache path.

    @mock.patch('data_pipeline.pipeline.LoaderFactory')
    @mock.patch('data_pipeline.pipeline.ProcessorFactory')
    @mock.patch('os.path.exists') # For checking existing raw file or source.path
    def test_load_cache_miss_successful_load_process_save(self, mock_os_exists, mock_proc_factory, mock_load_factory, 
                                                       pipeline_instance, mock_registry, mock_cache_manager, 
                                                       sample_source_def, mock_file_loader, mock_geojson_processor):
        mock_os_exists.return_value = True # Assume source.path exists for file type
        mock_registry.get.return_value = sample_source_def
        mock_cache_manager.load.return_value = None # Cache miss

        mock_load_factory.get_loader.return_value = mock_file_loader
        # _acquire_data will use source.path directly for 'file' type if it exists, so loader.load might not be called for this specific setup
        # Let's change source_type to http to force loader usage for this test.
        http_source_def = DataSource(**{**sample_source_def.__dict__, 'source_type': 'http', 'url': 'http://fake.url/data.geojson', 'path': None})
        mock_registry.get.return_value = http_source_def
        
        # _acquire_data specific mocks
        # It will try to see if download_path exists first. Let's make it not exist.
        def side_effect_os_exists(path_arg):
            if "fake.url" in path_arg or "sample_data.geojson" in path_arg: # a bit specific, better to make it more robust
                 # for the generated download_path, pretend it doesn't exist to force download
                if http_source_def.name in path_arg: return False 
            return True # for other general checks if any
        mock_os_exists.side_effect = side_effect_os_exists

        processed_gdf = gpd.GeoDataFrame({'data': ['processed']})
        mock_geojson_processor.process.return_value = processed_gdf
        mock_proc_factory.get_processor.return_value = mock_geojson_processor
        
        mock_cache_manager.save.return_value = True

        result_gdf = pipeline_instance.load(http_source_def.name)

        mock_registry.get.assert_called_once_with(http_source_def.name)
        mock_cache_manager.load.assert_called_once_with(http_source_def.cache_key, http_source_def.params)
        
        mock_load_factory.get_loader.assert_called_once_with(http_source_def.source_type)
        # The actual path for loader.load will be constructed inside _acquire_data
        # We need to ensure the path passed to loader.load is what we expect.
        # For now, checking it was called is a good start.
        mock_file_loader.load.assert_called_once() 

        mock_proc_factory.get_processor.assert_called_once_with(http_source_def.format) # or inferred format
        # The raw_path for processor.process is the result from _acquire_data
        mock_geojson_processor.process.assert_called_once_with(http_source_def, mock.ANY) # mock.ANY for raw_path

        mock_cache_manager.save.assert_called_once_with(processed_gdf, http_source_def.cache_key, http_source_def.params)
        assert result_gdf.equals(processed_gdf)

    @mock.patch('data_pipeline.pipeline.LoaderFactory')
    @mock.patch('os.path.exists')
    def test_load_loader_fails(self, mock_os_exists, mock_load_factory, pipeline_instance, mock_registry, mock_cache_manager, sample_source_def):
        mock_os_exists.return_value = True # for source.path if file type
        http_source_def = DataSource(**{**sample_source_def.__dict__, 'source_type': 'http', 'url': 'http://fake.url/data.geojson', 'path': None})
        mock_registry.get.return_value = http_source_def
        mock_cache_manager.load.return_value = None # Cache miss

        mock_failing_loader = mock.MagicMock()
        mock_failing_loader.load.return_value = False # Simulate loader failure
        mock_load_factory.get_loader.return_value = mock_failing_loader
        
        # For _acquire_data, ensure download_path doesn't exist to trigger loader
        mock_os_exists.side_effect = lambda p: False if http_source_def.name in p else True 

        with pytest.raises(RuntimeError, match=f"Failed to acquire data for {http_source_def.name}"):
            pipeline_instance.load(http_source_def.name)
        
        mock_cache_manager.save.assert_not_called()

    @mock.patch('data_pipeline.pipeline.LoaderFactory')
    @mock.patch('data_pipeline.pipeline.ProcessorFactory')
    @mock.patch('os.path.exists')
    def test_load_processor_fails(self, mock_os_exists, mock_proc_factory, mock_load_factory, pipeline_instance, 
                                  mock_registry, mock_cache_manager, sample_source_def, mock_file_loader):
        mock_os_exists.return_value = True # for source.path if file type
        # Use file type to simplify _acquire_data path, assumes source.path exists from mock_os_exists
        mock_registry.get.return_value = sample_source_def 
        mock_cache_manager.load.return_value = None # Cache miss

        # _acquire_data will return sample_source_def.path for file type
        # No need to mock LoaderFactory if source_type is 'file' and path exists

        mock_failing_processor = mock.MagicMock()
        mock_failing_processor.process.side_effect = ValueError("Processing failed miserably")
        mock_proc_factory.get_processor.return_value = mock_failing_processor

        with pytest.raises(ValueError, match="Processing failed miserably"):
            pipeline_instance.load(sample_source_def.name)
        
        mock_cache_manager.save.assert_not_called()

    def test_load_unknown_source(self, pipeline_instance, mock_registry):
        mock_registry.get.return_value = None
        with pytest.raises(ValueError, match="Unknown data source: unknown_source"):
            pipeline_instance.load("unknown_source")

    def test_load_with_runtime_param_override(self, pipeline_instance, mock_registry, mock_cache_manager, sample_source_def):
        mock_gdf = gpd.GeoDataFrame({'data': ['cached_override']})
        mock_registry.get.return_value = sample_source_def
        mock_cache_manager.load.return_value = mock_gdf

        runtime_params = {"year": 2024, "new_param": True}
        expected_merged_params = {**sample_source_def.params, **runtime_params}

        result_gdf = pipeline_instance.load(sample_source_def.name, **runtime_params)

        # Assert registry was called with original name
        mock_registry.get.assert_called_once_with(sample_source_def.name)
        # Assert cache was checked with merged params
        # The source object passed to cache_manager.load will be the *new* one with merged params.
        # So cache_key and params for this call should reflect the merged state.
        
        # Reconstruct the expected source name used for cache key from the original name
        # (cache_key defaults to name if not specified in DataSource)
        # The params passed to cache_manager.load will be the merged ones.
        mock_cache_manager.load.assert_called_once_with(sample_source_def.name, expected_merged_params)
        assert result_gdf.equals(mock_gdf)

    @mock.patch('os.path.exists', return_value=True)
    def test_acquire_data_file_source_exists(self, mock_exists, pipeline_instance, sample_source_def):
        # source_type is 'file' and path exists
        result_path = pipeline_instance._acquire_data(sample_source_def)
        assert result_path == sample_source_def.path
        mock_exists.assert_called_once_with(sample_source_def.path)

    @mock.patch('os.path.exists', return_value=False)
    def test_acquire_data_file_source_not_exists(self, mock_exists, pipeline_instance, sample_source_def):
        # source_type is 'file' and path does NOT exist
        result_path = pipeline_instance._acquire_data(sample_source_def)
        assert result_path is None
        mock_exists.assert_called_once_with(sample_source_def.path)

    @mock.patch('data_pipeline.pipeline.LoaderFactory')
    @mock.patch('os.path.exists') # Mock os.path.exists for _acquire_data logic
    def test_acquire_data_remote_source_download(self, mock_os_path_exists, mock_loader_factory, pipeline_instance, sample_source_def, mock_file_loader):
        http_source = DataSource(
            name="remote_source", source_type="http", format="geojson", 
            url="http://test.com/data.geojson", output_dir="remote_output"
        )
        # Simulate that the target download path does NOT exist initially
        mock_os_path_exists.return_value = False 
        mock_loader_factory.get_loader.return_value = mock_file_loader # mock_file_loader is a MagicMock
        mock_file_loader.load.return_value = True # Simulate successful download by loader

        expected_filename = "data.geojson"
        expected_target_dir = os.path.join(pipeline_instance.gis_data_dir, http_source.output_dir, http_source.name)
        expected_download_path = os.path.join(expected_target_dir, expected_filename)

        result_path = pipeline_instance._acquire_data(http_source)
        
        assert result_path == expected_download_path
        mock_loader_factory.get_loader.assert_called_once_with("http")
        mock_file_loader.load.assert_called_once_with(http_source, expected_download_path)
        # os.path.exists would be called for the download_path
        mock_os_path_exists.assert_any_call(expected_download_path)

    @mock.patch('data_pipeline.pipeline.LoaderFactory')
    @mock.patch('os.path.exists')
    def test_acquire_data_remote_source_already_exists(self, mock_os_path_exists, mock_loader_factory, pipeline_instance, sample_source_def):
        http_source = DataSource(
            name="remote_source_exists", source_type="https", format="json", 
            url="http://test.com/data_exists.json", output_dir="remote_exists_output"
        )
        # Simulate that the target download path *DOES* exist
        mock_os_path_exists.return_value = True
        
        expected_filename = "data_exists.json"
        expected_target_dir = os.path.join(pipeline_instance.gis_data_dir, http_source.output_dir, http_source.name)
        expected_download_path = os.path.join(expected_target_dir, expected_filename)

        result_path = pipeline_instance._acquire_data(http_source)
        
        assert result_path == expected_download_path
        mock_loader_factory.get_loader.assert_not_called() # Loader should not be called
        mock_os_path_exists.assert_called_once_with(expected_download_path) # Only check for existing download

    @mock.patch('data_pipeline.pipeline.ProcessorFactory')
    def test_process_data_with_processor(self, mock_processor_factory, pipeline_instance, sample_source_def, mock_geojson_processor):
        mock_processor_factory.get_processor.return_value = mock_geojson_processor
        mock_raw_path = "/fake/raw/data.geojson"
        
        pipeline_instance._process_data(sample_source_def, mock_raw_path)
        
        mock_processor_factory.get_processor.assert_called_once_with(sample_source_def.format)
        mock_geojson_processor.process.assert_called_once_with(sample_source_def, mock_raw_path)

    def test_clear_cache_specific_source(self, pipeline_instance, mock_registry, mock_cache_manager, sample_source_def):
        mock_registry.get.return_value = sample_source_def
        pipeline_instance.clear_cache(sample_source_def.name)
        mock_cache_manager.clear.assert_called_once_with(sample_source_def.cache_key)

    def test_clear_cache_all(self, pipeline_instance, mock_cache_manager):
        pipeline_instance.clear_cache()
        mock_cache_manager.clear.assert_called_once_with() # Called with no args 