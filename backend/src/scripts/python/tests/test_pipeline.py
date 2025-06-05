import pytest
import os
from unittest import mock
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

from data_pipeline.sources import DataSource
from data_pipeline.pipeline import DataPipeline
from data_pipeline.registry import DataRegistry
from data_pipeline.cache import CacheManager
from data_pipeline.loaders import FileLoader  # Assuming FileLoader for some tests
from data_pipeline.processors import GeoJSONProcessor  # Assuming for some tests
from data_pipeline.download_cache import RawDownloadCache

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
    """Instantiate DataPipeline with mocked CacheManager and isolate RawDownloadCache root in tmp_path."""
    with mock.patch('data_pipeline.pipeline.CacheManager', return_value=mock_cache_manager):
        # Patch RawDownloadCache used *inside* DataPipeline to avoid touching real filesystem
        with mock.patch('data_pipeline.pipeline.RawDownloadCache') as MockRDC:
            # Configure ensure to just execute its download_fn and return a deterministic path
            def _ensure(url, source_name, download_fn):
                dest = tmp_path / "raw_cache" / source_name / "dummy.dat"
                dest.parent.mkdir(parents=True, exist_ok=True)
                ok = download_fn(str(dest))
                return str(dest) if ok else None
            MockRDC.return_value.ensure.side_effect = _ensure
            pipeline = DataPipeline(registry=mock_registry, cache_dir=str(tmp_path / "pipe_cache"))
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
    def test_load_cache_miss_successful_load_process_save(self, mock_proc_factory, mock_load_factory, 
                                                       pipeline_instance, mock_registry, mock_cache_manager, 
                                                       sample_source_def, mock_file_loader, mock_geojson_processor):
        http_source_def = DataSource(**{**sample_source_def.__dict__, 'source_type': 'https', 'url': 'https://fake.url/data.geojson', 'path': None})
        mock_registry.get.return_value = http_source_def
        mock_cache_manager.load.return_value = None  # Cache miss

        mock_load_factory.get_loader.return_value = mock_file_loader
        mock_file_loader.load.return_value = True

        processed_gdf = gpd.GeoDataFrame({'data': ['processed']})
        mock_geojson_processor.process.return_value = processed_gdf
        mock_proc_factory.get_processor.return_value = mock_geojson_processor

        result_gdf = pipeline_instance.load(http_source_def.name)

        mock_registry.get.assert_called_once_with(http_source_def.name)
        mock_load_factory.get_loader.assert_called_once_with(http_source_def.source_type)
        mock_file_loader.load.assert_called_once()  # via RawDownloadCache.ensure
        mock_cache_manager.save.assert_called_once()
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
    def test_acquire_data_remote_source_download(self, mock_loader_factory, pipeline_instance, sample_source_def, mock_file_loader, tmp_path):
        https_source = DataSource(
            name="remote_source", source_type="https", format="geojson", 
            url="https://test.com/data.geojson", output_dir="remote_output"
        )
        mock_loader_factory.get_loader.return_value = mock_file_loader
        mock_file_loader.load.return_value = True

        result_path = pipeline_instance._acquire_data(https_source)

        mock_loader_factory.get_loader.assert_called_once_with("https")
        mock_file_loader.load.assert_called_once()
        # Path should lie inside tmp_path/raw_cache directory per patched ensure side_effect
        assert str(tmp_path / "raw_cache" / https_source.name) in result_path

    @mock.patch('data_pipeline.pipeline.ProcessorFactory')
    def test_process_data_with_processor(self, mock_processor_factory, pipeline_instance, sample_source_def, mock_geojson_processor):
        mock_processor_factory.get_processor.return_value = mock_geojson_processor
        mock_raw_path = "/fake/raw/data.geojson"
        
        pipeline_instance._process_data(sample_source_def, mock_raw_path)
        
        mock_processor_factory.get_processor.assert_called_once_with(sample_source_def.format)
        mock_geojson_processor.process.assert_called_once_with(sample_source_def, mock_raw_path, bbox=None, keep_cols=None)

    def test_clear_cache_specific_source(self, pipeline_instance, mock_registry, mock_cache_manager, sample_source_def):
        mock_registry.get.return_value = sample_source_def
        pipeline_instance.clear_cache(sample_source_def.name)
        mock_cache_manager.clear.assert_called_once_with(sample_source_def.cache_key)

    def test_clear_cache_all(self, pipeline_instance, mock_cache_manager):
        pipeline_instance.clear_cache()
        mock_cache_manager.clear.assert_called_once_with() # Called with no args 

class TestRawDownloadCache:
    def test_ensure_caches_file(self, tmp_path):
        cache = RawDownloadCache(root_dir=tmp_path / "rdc")
        url = "https://example.com/data/file.bin"

        # download_fn writes a marker file
        def dl(dest):
            Path(dest).write_text("content"); return True

        first_path = cache.ensure(url, "example", dl)
        assert Path(first_path).exists()

        # Second call should not invoke dl (simulate by failing if called)
        def dl_fail(dest):
            pytest.fail("download_fn should not be called on cache hit")

        second_path = cache.ensure(url, "example", dl_fail)
        assert second_path == first_path

    def test_trim(self, tmp_path):
        cache = RawDownloadCache(root_dir=tmp_path / "rdc_trim")
        url_base = "https://example.com/data/"

        # Create 5 files ~1MB each
        for i in range(5):
            def _dl(dest, idx=i):
                Path(dest).write_bytes(b"0" * 1024 * 1024)  # 1 MB
                return True
            cache.ensure(url_base + f"f{i}.bin", "trim", _dl)

        from data_pipeline.cache import CacheManager
        cm = CacheManager(cache_dir=tmp_path / "feather_cache")
        cm.trim(max_size_gb=0.001)  # 1 MB limit triggers deletion
        # Total feathers initially zero, this just checks no exception. 